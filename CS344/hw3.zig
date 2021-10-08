const std = @import("std");
const log = std.log;
const math = std.math;
const assert = std.debug.assert;

const zigimg = @import("zigimg");

const cudaz = @import("cudaz");
const Cuda = cudaz.Cuda;
const cu = cudaz.cu;

const png = @import("png.zig");
const utils = @import("utils.zig");

const resources_dir = "CS344/hw3_resources/";

pub fn main() !void {
    var general_purpose_allocator = std.heap.GeneralPurposeAllocator(.{}){};
    const allocator = &general_purpose_allocator.allocator;
    var cuda = try Cuda.init(0);
    defer cuda.deinit();
    log.info("***** HW3 ******", .{});

    // float *d_luminance;
    // unsigned int *d_cdf;

    // size_t numRows, numCols;
    // unsigned int numBins;

    // double perPixelError = 0.0;
    // double globalError = 0.0;
    // bool useEpsCheck = false;

    const img = try zigimg.Image.fromFilePath(allocator, resources_dir ++ "/memorial_exr.png");
    const numRows = img.height;
    const numCols = img.width;
    const rgb = try asFloat32(allocator, img);
    // std.log.debug("rgb image: {d:.3}", .{rgb});

    img.deinit();

    // load the image and convert it to xyY format
    const d_rgb = try cuda.allocAndCopy(cu.float3, rgb);
    const d_xyY = try cuda.alloc(cu.float3, numCols * numRows);

    const threads = cudaz.Dim3.init(32, 16, 1);
    const blocks = cudaz.Dim3.init((numCols + threads.x - 1) / threads.x, (numRows + threads.y - 1) / threads.y, 1);
    const grid = cudaz.Grid{ .blocks = blocks, .threads = threads };
    const rgb_to_xyY = try cudaz.Function("rgb_to_xyY").init(&cuda);
    try rgb_to_xyY.launch(grid, .{
        d_rgb.ptr,
        d_xyY.ptr,
        0.0001,
        @intCast(c_int, numRows),
        @intCast(c_int, numCols),
    });

    try cuda.synchronize();
    const h_xyY = try allocator.alloc(cu.float3, numCols * numRows);
    try cuda.memcpyDtoH(cu.float3, h_xyY, d_xyY);

    // allocate memory for the cdf of the histogram
    const numBins: usize = 1024;
    var d_cdf = try cuda.alloc(f32, numBins);
    defer cuda.free(d_cdf);

    var timer = cudaz.GpuTimer.init(&cuda);
    defer timer.deinit();
    // float min_logLum, max_logLum;
    timer.start();
    // TODO
    const min_max_lum = try histogram_and_prefixsum(&cuda, d_xyY, d_cdf, numRows, numCols, numBins);
    var lum_min = min_max_lum.x;
    var lum_range = min_max_lum.y - min_max_lum.x;

    assert(lum_range > 0);
    timer.stop();
    try cuda.synchronize();
    std.log.info("Your code ran in: {d:.1} msecs.", .{timer.elapsed() * 1000});
    std.log.info("Found a lum range of: {d:.5}", .{min_max_lum});
    var h_cdf = try allocator.alloc(f32, numBins);
    try cuda.memcpyDtoH(f32, h_cdf, d_cdf);
    std.log.info("Lum cdf: {d:.3}", .{h_cdf});

    const tone_map = try cudaz.Function("tone_map").init(&cuda);
    try tone_map.launch(
        grid,
        .{
            d_xyY.ptr,
            d_cdf.ptr,
            d_rgb.ptr,
            lum_min,
            lum_range,
            numBins,
            @intCast(c_int, numRows),
            @intCast(c_int, numCols),
        },
    );
    // check results and output the tone-mapped image
    // postProcess(output_file, numRows, numCols, min_logLum, max_logLum);
    try cuda.memcpyDtoH(cu.float3, rgb, d_rgb);
    var out_img = try fromFloat32(allocator, rgb, numCols, numRows);
    defer out_img.deinit();
    try png.writePngToFilePath(out_img, resources_dir ++ "output.png");
    try utils.validate_output(allocator, resources_dir);

    // TODO? referenceCalculation

}

fn asFloat32(allocator: *std.mem.Allocator, img: zigimg.Image) ![]cu.float3 {
    var rgb = try allocator.alloc(cu.float3, img.width * img.height);
    var pixels = img.iterator();
    var i: usize = 0;
    while (pixels.next()) |color| : (i += 1) {
        rgb[i] = .{
            .x = color.R,
            .y = color.G,
            .z = color.B,
        };
    }
    return rgb;
}

pub inline fn toColorIntClamp(comptime T: type, value: f32) T {
    if (math.isNan(value)) return 0;
    var val = value;
    var max_val = @intToFloat(f32, math.maxInt(T));
    var min_val = @intToFloat(f32, math.minInt(T));
    val = math.max(min_val, math.min(max_val, val * max_val));

    return @floatToInt(T, math.round(val));
}

fn fromFloat32(allocator: *std.mem.Allocator, rgb: []cu.float3, width: usize, height: usize) !zigimg.Image {
    var img = try zigimg.Image.create(allocator, width, height, .Rgb24, .Png);
    var pixels = img.pixels.?.Rgb24;
    for (rgb) |value, i| {
        pixels[i] = zigimg.color.Rgb24{
            .R = toColorIntClamp(u8, value.x),
            .G = toColorIntClamp(u8, value.y),
            .B = toColorIntClamp(u8, value.z),
        };
        // if (i % 100 == 0) {
        //     log.debug("{} -> {}", .{ value, pixels[i] });
        // }
    }
    return img;
}

fn histogram_and_prefixsum(
    cuda: *Cuda,
    d_xyY: []const cu.float3,
    d_cdf: []f32,
    numRows: usize,
    numCols: usize,
    numBins: usize,
) !cu.float2 {
    // Here are the steps you need to implement
    //   1) find the minimum and maximum value in the input logLuminance channel
    //      store in min_logLum and max_logLum
    //   2) subtract them to find the range
    //   3) generate a histogram of all the values in the logLuminance channel using
    //      the formula: bin = (lum[i] - lumMin) / lumRange * numBins
    // TODO: faster Cdf
    //   4) Perform an exclusive scan (prefix sum) on the histogram to get
    //      the cumulative distribution of luminance values (this should go in the
    //      incoming d_cdf pointer which already has been allocated for you)
    var num_pixels = numRows * numCols;
    var min_max_lum = try reduceMinMaxLum(cuda, d_xyY);

    const lum_histo = try cudaz.Function("lum_histo").init(cuda);
    var d_histo = try cuda.alloc(c_uint, numBins);
    try cuda.memset(c_uint, d_histo, 0);
    try lum_histo.launch(
        cudaz.Grid.init1D(num_pixels, 1024),
        .{
            d_histo.ptr,
            d_xyY.ptr,
            min_max_lum.x,
            min_max_lum.y - min_max_lum.x,
            @intCast(c_int, numBins),
            @intCast(c_int, num_pixels),
        },
    );
    var histo = try cuda.arena.allocator.alloc(c_uint, numBins);
    defer cuda.arena.allocator.free(histo);
    try cuda.memcpyDtoH(c_uint, histo, d_histo);
    std.log.info("Lum histo: {any}", .{histo});

    const computeCdf = try cudaz.Function("computeCdf").init(cuda);
    try computeCdf.launch(
        cudaz.Grid.init1D(numBins, numBins),
        .{ d_cdf.ptr, d_histo.ptr, @intCast(c_int, numBins) },
    );
    // var histo = try cuda.allocator.alloc(c_uint, numBins);
    // defer cuda.allocator.free(histo);
    // try cuda.memcpyDtoH(c_uint, histo, d_cdf);
    // std.log.info("Lum histo: {any}", .{histo});
    return min_max_lum;
}

fn reduceMinMaxLum(
    cuda: *Cuda,
    d_xyY: []const cu.float3,
) !cu.float2 {
    const num_pixels = d_xyY.len;
    const reduce_minmax_lum = try cudaz.Function("reduce_minmax_lum").init(cuda);

    const grid = cudaz.Grid.init1D(num_pixels, 1024);
    var d_buff = try cuda.alloc(cu.float2, grid.blocks.x);
    defer cuda.free(d_buff);
    var d_min_max_lum = try cuda.alloc(cu.float2, 1);
    defer cuda.free(d_min_max_lum);

    try reduce_minmax_lum.launchWithSharedMem(
        grid,
        grid.threads.x * @sizeOf(cu.float2),
        .{ d_xyY.ptr, d_buff.ptr, @intCast(c_int, num_pixels) },
    );
    try cuda.synchronize();

    const one_block = cudaz.Grid.init1D(d_buff.len, 0);
    const reduce_minmax = try cudaz.Function("reduce_minmax").init(cuda);
    try reduce_minmax.launchWithSharedMem(
        one_block,
        d_buff.len * @sizeOf(cu.float2),
        .{ d_buff.ptr, d_min_max_lum.ptr },
    );
    try cuda.synchronize();
    var min_max_lum = try cuda.readResult(cu.float2, d_min_max_lum);

    try std.testing.expect(min_max_lum.x < min_max_lum.y);
    return min_max_lum;
}
// void postProcess(const std::string &output_file, size_t numRows, size_t numCols,
//                  float min_log_Y, float max_log_Y) {
//   const int numPixels = numRows__ * numCols__;

//   const int numThreads = 192;

//   float *d_cdf_normalized;

//   checkCudaErrors(cudaMalloc(&d_cdf_normalized, sizeof(float) * numBins));

//   // first normalize the cdf to a maximum value of 1
//   // this is how we compress the range of the luminance channel
//   normalize_cdf<<<(numBins + numThreads - 1) / numThreads, numThreads>>>(
//       d_cdf__, d_cdf_normalized, numBins);

//   cudaDeviceSynchronize();
//   checkCudaErrors(cudaGetLastError());

//   // allocate memory for the output RGB channels
//   float *h_red, *h_green, *h_blue;
//   float *d_red, *d_green, *d_blue;

//   h_red = new float[numPixels];
//   h_green = new float[numPixels];
//   h_blue = new float[numPixels];

//   checkCudaErrors(cudaMalloc(&d_red, sizeof(float) * numPixels));
//   checkCudaErrors(cudaMalloc(&d_green, sizeof(float) * numPixels));
//   checkCudaErrors(cudaMalloc(&d_blue, sizeof(float) * numPixels));

//   float log_Y_range = max_log_Y - min_log_Y;

//   const dim3 blockSize(32, 16, 1);
//   const dim3 gridSize((numCols + blockSize.x - 1) / blockSize.x,
//                       (numRows + blockSize.y - 1) / blockSize.y);
//   // next perform the actual tone-mapping
//   // we map each luminance value to its new value
//   // and then transform back to RGB space
//   tonemap<<<gridSize, blockSize>>>(d_x__, d_y__, d_logY__, d_cdf_normalized,
//                                    d_red, d_green, d_blue, min_log_Y, max_log_Y,
//                                    log_Y_range, numBins, numRows, numCols);

//   cudaDeviceSynchronize();
//   checkCudaErrors(cudaGetLastError());

//   checkCudaErrors(cudaMemcpy(h_red, d_red, sizeof(float) * numPixels,
//                              cudaMemcpyDeviceToHost));
//   checkCudaErrors(cudaMemcpy(h_green, d_green, sizeof(float) * numPixels,
//                              cudaMemcpyDeviceToHost));
//   checkCudaErrors(cudaMemcpy(h_blue, d_blue, sizeof(float) * numPixels,
//                              cudaMemcpyDeviceToHost));

//   // recombine the image channels
//   float *imageHDR = new float[numPixels * 3];

//   for (int i = 0; i < numPixels; ++i) {
//     imageHDR[3 * i + 0] = h_blue[i];
//     imageHDR[3 * i + 1] = h_green[i];
//     imageHDR[3 * i + 2] = h_red[i];
//   }

//   saveImageHDR(imageHDR, numRows, numCols, output_file);

//   delete[] imageHDR;
//   delete[] h_red;
//   delete[] h_green;
//   delete[] h_blue;

//   // cleanup
//   checkCudaErrors(cudaFree(d_cdf_normalized));
// }

fn z(_z: f32) cu.float3 {
    return cu.float3{ .x = 0.0, .y = 0.0, .z = _z };
}

test "histogram" {
    var cuda = try Cuda.init(0);
    var img = [_]cu.float3{
        z(0.0),
        z(0.0),
        z(1.0),
        z(2.0),
        z(3.0),
        z(4.0),
        z(6.0),
        z(7.0),
        z(8.0),
        z(9.0),
        z(3.0),
        z(3.0),
        z(3.0),
        z(9.0),
        z(10.0),
    };
    const lum_histo = try cudaz.Function("lum_histo").init(&cuda);
    var bins = [_]c_uint{0} ** 10;
    var d_img = try cuda.allocAndCopy(cu.float3, &img);
    var d_bins = try cuda.allocAndCopy(c_uint, &bins);
    try lum_histo.launch(
        cudaz.Grid.init1D(img.len, 3),
        .{
            d_bins.ptr,
            d_img.ptr,
            0,
            9,
            @intCast(c_int, bins.len),
            @intCast(c_int, img.len),
        },
    );
    try cuda.memcpyDtoH(c_uint, &bins, d_bins);

    std.log.warn("bins: {any}", .{bins});
    try std.testing.expectEqual(
        [10]c_uint{ 2, 1, 1, 4, 1, 0, 1, 1, 1, 3 },
        bins,
    );
}

// test "min_max_lum" {
//     var cuda = try Cuda.init(0);
//     var img = [_]cu.float3{
//         z(0.0),
//         z(0.0),
//         z(1.0),
//         z(2.0),
//         z(3.0),
//         z(4.0),
//         z(6.0),
//         z(7.0),
//         z(8.0),
//         z(9.0),
//         z(3.0),
//         z(3.0),
//         z(3.0),
//         z(9.0),
//         z(10.0),
//         z(3.0),
//     };
//     const reduce_minmax_lum = try cudaz.Function("reduce_minmax_lum").init(&cuda);
//     var minmax = [_]cu.float2{.{ .x = 0.0, .y = 0.0 }} ** 8;
//     var d_img = try cuda.allocAndCopy(cu.float3, &img);
//     var d_minmax = try cuda.allocAndCopy(cu.float2, &minmax);
//     try reduce_minmax_lum.launch(
//         cudaz.Grid.init1D(img.len, minmax.len),
//         .{
//             d_img.ptr,
//             d_minmax.ptr,
//             0,
//             9,
//             @intCast(c_int, minmax.len),
//             @intCast(c_int, img.len),
//         },
//     );
//     try cuda.memcpyDtoH(c_uint, &minmax, d_minmax);

//     std.log.warn("minmax: {any}", .{minmax});
//     try std.testing.expectEqual(
//         [10]c_uint{ 2, 1, 1, 4, 1, 0, 1, 1, 1, 3 },
//         minmax,
//     );
// }

test "cdf" {
    var cuda = try Cuda.init(0);
    var bins = [_]c_uint{ 2, 1, 1, 4, 1, 0, 1, 1, 1, 3 };
    const computeCdf = try cudaz.Function("computeCdf").init(&cuda);
    var cdf = [_]f32{0.0} ** 10;
    var d_bins = try cuda.allocAndCopy(c_uint, &bins);
    var d_cdf = try cuda.allocAndCopy(f32, &cdf);
    try computeCdf.launch(
        cudaz.Grid.init1D(bins.len, 0),
        .{
            d_cdf.ptr,
            d_bins.ptr,
            @intCast(c_int, bins.len),
        },
    );
    try cuda.memcpyDtoH(f32, &cdf, d_cdf);

    // bins: { 2, 1, 1, 4, 1, 0, 1, 1, 1, 3 }
    var t: f32 = 15.0;
    const tgt_cdf = [10]f32{ 0, 2 / t, 3 / t, 4 / t, 8 / t, 9 / t, 9 / t, 10 / t, 11 / t, 12 / t };
    std.log.warn("tgt_cdf: {d:.3}", .{tgt_cdf});
    std.log.warn("cdf: {d:.3}", .{cdf});
    var i: u8 = 0;
    while (i < cdf.len) : (i += 1) {
        try std.testing.expectApproxEqRel(tgt_cdf[i], cdf[i], 0.0001);
    }
}
