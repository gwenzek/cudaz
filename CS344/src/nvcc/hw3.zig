const std = @import("std");
const log = std.log;
const math = std.math;
const testing = std.testing;
const assert = std.debug.assert;

const cuda = @import("cudaz");
const cu = cuda.cu;

const png = @import("png.zig");
const utils = @import("utils.zig");

const resources_dir = "resources/hw3_resources/";

pub fn main() !void {
    var general_purpose_allocator = std.heap.GeneralPurposeAllocator(.{}){};
    const allocator = general_purpose_allocator.allocator();
    var stream = try cuda.Stream.init(0);
    defer stream.deinit();
    log.info("***** HW3 ******", .{});

    const img = try png.Image.fromFilePath(allocator, resources_dir ++ "/memorial_exr.png");
    const numRows = img.height;
    const numCols = img.width;
    const rgb = try asFloat32(allocator, img);

    img.deinit();

    // load the image and convert it to xyY format
    const d_rgb = try cuda.allocAndCopy(cu.float3, rgb);
    const d_xyY = try cuda.alloc(cu.float3, numCols * numRows);

    const threads = cuda.Dim3.init(32, 16, 1);
    const blocks = cuda.Dim3.init((numCols + threads.x - 1) / threads.x, (numRows + threads.y - 1) / threads.y, 1);
    const grid = cuda.Grid{ .blocks = blocks, .threads = threads };
    const rgb_to_xyY = try cuda.CudaKernel("rgb_to_xyY").init();
    try rgb_to_xyY.launch(&stream, grid, .{
        d_rgb.ptr,
        d_xyY.ptr,
        0.0001,
        @intCast(c_int, numRows),
        @intCast(c_int, numCols),
    });

    const h_xyY = try allocator.alloc(cu.float3, numCols * numRows);
    try cuda.memcpyDtoH(cu.float3, h_xyY, d_xyY);

    // allocate memory for the cdf of the histogram
    const numBins: usize = 1024;
    var d_cdf = try cuda.alloc(f32, numBins);
    defer cuda.free(d_cdf);

    var timer = cuda.GpuTimer.start(&stream);
    errdefer timer.deinit();

    const min_max_lum = try histogram_and_prefixsum(&stream, d_xyY, d_cdf, numRows, numCols, numBins);
    var lum_min = min_max_lum.x;
    var lum_range = min_max_lum.y - min_max_lum.x;

    timer.stop();
    stream.synchronize();
    std.log.info("Your code ran in: {d:.1} msecs.", .{timer.elapsed() * 1000});
    std.log.info("Found a lum range of: ({d:.5}, {d:.5})", min_max_lum);

    var h_cdf = try cuda.allocAndCopyResult(f32, allocator, d_cdf);
    std.log.info("Lum cdf: {d:.3}", .{h_cdf});

    const tone_map = try cuda.CudaKernel("tone_map").init();
    try tone_map.launch(
        &stream,
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
    try cuda.memcpyDtoH(cu.float3, rgb, d_rgb);
    var out_img = try fromFloat32(allocator, rgb, numCols, numRows);
    defer out_img.deinit();
    try out_img.writeToFilePath(resources_dir ++ "output.png");
    try utils.validate_output(allocator, resources_dir, 2.0);

    try std.testing.expect(lum_range > 0);
}

fn asFloat32(allocator: std.mem.Allocator, img: png.Image) ![]cu.float3 {
    var rgb = try allocator.alloc(cu.float3, img.width * img.height);
    var pixels = img.iterator();
    var i: usize = 0;
    while (pixels.next()) |color| : (i += 1) {
        rgb[i] = .{
            .x = color.r,
            .y = color.g,
            .z = color.b,
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

fn fromFloat32(allocator: std.mem.Allocator, rgb: []cu.float3, width: usize, height: usize) !png.Image {
    var pixels = try allocator.alloc(png.Rgb24, width * height);
    for (rgb) |value, i| {
        pixels[i] = png.Rgb24{
            .r = toColorIntClamp(u8, value.x),
            .g = toColorIntClamp(u8, value.y),
            .b = toColorIntClamp(u8, value.z),
        };
        // if (i % 100 == 0) {
        //     log.debug("{} -> {}", .{ value, pixels[i] });
        // }
    }
    return png.Image{
        .allocator = allocator,
        .width = @intCast(u32, width),
        .height = @intCast(u32, height),
        .px = .{ .rgb24 = pixels },
    };
}

fn histogram_and_prefixsum(
    stream: *cuda.Stream,
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
    //   4) Perform an exclusive scan (prefix sum) on the histogram to get
    //      the cumulative distribution of luminance values (this should go in the
    //      incoming d_cdf pointer which already has been allocated for you)
    var num_pixels = numRows * numCols;
    var min_max_lum = try reduceMinMaxLum(stream, d_xyY);

    const lumHisto = try cuda.CudaKernel("lum_histo").init();
    var d_histo = try cuda.alloc(c_uint, numBins);
    try cuda.memset(c_uint, d_histo, 0);
    try lumHisto.launch(
        stream,
        cuda.Grid.init1D(num_pixels, 1024),
        .{
            d_histo.ptr,
            d_xyY.ptr,
            min_max_lum.x,
            min_max_lum.y - min_max_lum.x,
            @intCast(c_int, numBins),
            @intCast(c_int, num_pixels),
        },
    );

    const computeCdf = try cuda.CudaKernel("blellochCdf").init();
    try computeCdf.launch(
        stream,
        cuda.Grid.init1D(numBins, numBins),
        .{ d_cdf.ptr, d_histo.ptr, @intCast(c_int, numBins) },
    );
    stream.synchronize();

    return min_max_lum;
}

fn reduceMinMaxLum(
    stream: *cuda.Stream,
    d_xyY: []const cu.float3,
) !cu.float2 {
    // TODO: the results seems to change between runs
    const num_pixels = d_xyY.len;
    const reduce_minmax_lum = try cuda.CudaKernel("reduce_minmax_lum").init();

    const grid = cuda.Grid.init1D(num_pixels, 1024);
    var d_buff = try cuda.alloc(cu.float2, grid.blocks.x);
    defer cuda.free(d_buff);
    var d_min_max_lum = try cuda.alloc(cu.float2, 1);
    try cuda.memsetD8(cu.float2, d_min_max_lum, 0xaa);
    defer cuda.free(d_min_max_lum);

    try reduce_minmax_lum.launchWithSharedMem(
        stream,
        grid,
        grid.threads.x * @sizeOf(cu.float2),
        .{ d_xyY.ptr, d_buff.ptr, @intCast(c_int, num_pixels) },
    );

    const one_block = cuda.Grid.init1D(d_buff.len, 0);
    const reduce_minmax = try cuda.CudaKernel("reduce_minmax").init();
    try reduce_minmax.launchWithSharedMem(
        stream,
        one_block,
        one_block.threads.x * @sizeOf(cu.float2),
        .{ d_buff.ptr, d_min_max_lum.ptr, @intCast(c_int, num_pixels) },
    );
    var min_max_lum = stream.copyResult(cu.float2, &d_min_max_lum[0]);

    try std.testing.expect(min_max_lum.x < min_max_lum.y);
    return min_max_lum;
}

fn z(_z: f32) cu.float3 {
    return cu.float3{ .x = 0.0, .y = 0.0, .z = _z };
}

test "histogram" {
    var stream = try cuda.Stream.init(0);
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
    const lumHisto = try cuda.CudaKernel("lum_histo").init();
    var bins = [_]c_uint{0} ** 10;
    var d_img = try cuda.allocAndCopy(cu.float3, &img);
    var d_bins = try cuda.allocAndCopy(c_uint, &bins);
    try lumHisto.launch(
        &stream,
        cuda.Grid.init1D(img.len, 3),
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

const ReduceMinmaxLum = cuda.CudaKernel("reduce_minmax_lum");

fn test_min_max_lum(stream: *cuda.Stream, f: ReduceMinmaxLum, d_img: []cu.float3, expected: []const cu.float2) !void {
    var num_blocks = expected.len;
    var d_minmax = try cuda.alloc(cu.float2, num_blocks);
    defer cuda.free(d_minmax);

    var grid1D = cuda.Grid{
        .blocks = .{ .x = @intCast(c_uint, num_blocks) },
        .threads = .{ .x = @intCast(c_uint, std.math.divCeil(usize, d_img.len, num_blocks) catch unreachable) },
    };
    try f.launchWithSharedMem(
        stream,
        grid1D,
        @sizeOf(cu.float2) * grid1D.threads.x,
        .{
            d_img.ptr,
            d_minmax.ptr,
            @intCast(c_int, d_img.len),
        },
    );
    var minmax = try cuda.allocAndCopyResult(cu.float2, testing.allocator, d_minmax);
    defer testing.allocator.free(minmax);
    std.log.warn("minmax ({}x{}): {any}", .{ grid1D.blocks.x, grid1D.threads.x, minmax });
    for (expected) |exp, index| {
        std.testing.expectEqual(exp, minmax[index]) catch |err| {
            switch (err) {
                error.TestExpectedEqual => log.err("At index {} expected {d:.0} got {d:.0}", .{ index, exp, minmax[index] }),
                else => {},
            }
            return err;
        };
    }
}

test "min_max_lum" {
    var stream = try cuda.Stream.init(0);
    var img = [_]cu.float3{
        z(0),
        z(3),
        z(0),
        z(1),
        z(2),
        z(4),
        z(6),
        z(7),
        z(8),
        z(9),
        z(10),
        z(3),
        z(3),
        z(3),
        z(9),
        z(3),
    };
    const f = try ReduceMinmaxLum.init();
    var d_img = try cuda.allocAndCopy(cu.float3, &img);

    try test_min_max_lum(&stream, f, d_img, &[_]cu.float2{
        .{ .x = 0, .y = 7 },
        .{ .x = 3, .y = 10 },
    });
    try test_min_max_lum(&stream, f, d_img, &[_]cu.float2{
        .{ .x = 0, .y = 3 },
        .{ .x = 2, .y = 7 },
        .{ .x = 3, .y = 10 },
        .{ .x = 3, .y = 9 },
    });
}

test "cdf" {
    var stream = try cuda.Stream.init(0);
    inline for ([_][:0]const u8{ "computeCdf", "blellochCdf" }) |variant| {
        log.warn("Testing Cdf implementation: {s}", .{variant});
        var bins = [_]c_uint{ 2, 1, 1, 4, 1, 0, 1, 1, 1, 3 };
        const computeCdf = try cuda.CudaKernel(variant).init();
        var cdf = [_]f32{0.0} ** 10;
        var d_bins = try cuda.allocAndCopy(c_uint, &bins);
        var d_cdf = try cuda.allocAndCopy(f32, &cdf);
        try computeCdf.launch(
            &stream,
            cuda.Grid.init1D(bins.len, 0),
            .{
                d_cdf.ptr,
                d_bins.ptr,
                @intCast(c_int, bins.len),
            },
        );
        try cuda.memcpyDtoH(f32, &cdf, d_cdf);
        try cuda.memcpyDtoH(c_uint, &bins, d_bins);

        var t: f32 = 15.0;
        const tgt_cdf = [10]f32{ 0, 2 / t, 3 / t, 4 / t, 8 / t, 9 / t, 9 / t, 10 / t, 11 / t, 12 / t };
        log.warn("bins: {d}", .{bins});
        log.warn("tgt_cdf: {d:.3}", .{tgt_cdf});
        log.warn("cdf: {d:.3}", .{cdf});
        var i: u8 = 0;
        while (i < cdf.len) : (i += 1) {
            try std.testing.expectApproxEqRel(tgt_cdf[i], cdf[i], 0.0001);
        }
    }
}
