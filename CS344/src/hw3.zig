const std = @import("std");
const log = std.log;
const math = std.math;
const testing = std.testing;
const assert = std.debug.assert;

const cuda = @import("cudaz");
const cu = cuda.cu;

const png = @import("png.zig");
const utils = @import("utils.zig");
const kernels = @import("hw3_kernel.zig");

const resources_dir = "resources/hw3_resources/";

pub fn main() !void {
    var general_purpose_allocator = std.heap.GeneralPurposeAllocator(.{}){};
    const allocator = general_purpose_allocator.allocator();
    var stream = try cuda.Stream.init(0);
    defer stream.deinit();
    log.info("***** HW3 ******", .{});

    // load the image and convert it to f32 format
    // TODO: load the image to f32 directly from exr format
    const img = try png.Image.fromFilePath(allocator, resources_dir ++ "/memorial_exr.png");
    const rgb = try asFloat32(allocator, img);
    img.deinit();
    defer allocator.free(rgb);

    const d_rgb = try cuda.allocAndCopy(f32, rgb);
    const d_xyY = try cuda.alloc(f32, img.len() * 3);

    const grid = cuda.Grid.init1D(img.len(), 32);
    const rgb2xyY = try cuda.ZigKernel(kernels, "rgb2xyY").init();
    try rgb2xyY.launch(&stream, grid, .{ d_rgb, d_xyY, 0.0001 });
    // const h_xyY = try stream.allocAndCopyResult(f32, allocator, d_xyY);

    // allocate memory for the cdf of the histogram
    const numBins: usize = 1024;
    var d_cdf = try cuda.alloc(f32, numBins);
    defer cuda.free(d_cdf);

    var timer = cuda.GpuTimer.start(&stream);
    errdefer timer.deinit();

    const numRows = img.width;
    const numCols = img.height;
    const minmax_lum = try histogram_and_prefixsum(&stream, d_xyY, d_cdf, numRows, numCols, numBins);
    try std.testing.expect(minmax_lum.max > minmax_lum.min);

    timer.stop();
    stream.synchronize();
    std.log.info("Your code ran in: {d:.1} msecs.", .{timer.elapsed() * 1000});
    std.log.info("Found a lum range of: ({d:.5}, {d:.5})", minmax_lum);

    var h_cdf = try cuda.allocAndCopyResult(f32, allocator, d_cdf);
    std.log.info("Lum cdf: {d:.3}", .{h_cdf});

    const tone_map = try cuda.ZigKernel(kernels, "toneMap").init();
    try tone_map.launch(
        &stream,
        grid,
        .{
            d_xyY,
            d_cdf,
            d_rgb,
            minmax_lum,
            numBins,
        },
    );
    try cuda.memcpyDtoH(f32, rgb, d_rgb);
    var out_img = try fromFloat32(allocator, rgb, numCols, numRows);
    defer out_img.deinit();
    try out_img.writeToFilePath(resources_dir ++ "output.png");
    try utils.validate_output(allocator, resources_dir, 2.0);
}

fn asFloat32(allocator: std.mem.Allocator, img: png.Image) ![]f32 {
    var rgb = try allocator.alloc(f32, img.width * img.height * 3);
    var pixels = img.iterator();
    var i: usize = 0;
    while (pixels.next()) |color| : (i += 1) {
        rgb[3 * i] = color.r;
        rgb[3 * i + 1] = color.g;
        rgb[3 * i + 2] = color.b;
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

fn fromFloat32(allocator: std.mem.Allocator, rgb: []f32, width: usize, height: usize) !png.Image {
    var pixels = try allocator.alloc(png.Rgb24, width * height);
    for (rgb) |_, i| {
        pixels[i] = png.Rgb24{
            .r = toColorIntClamp(u8, rgb[3 * i + 0]),
            .g = toColorIntClamp(u8, rgb[3 * i + 1]),
            .b = toColorIntClamp(u8, rgb[3 * i + 2]),
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
    d_xyY: []const f32,
    d_cdf: []f32,
    numRows: usize,
    numCols: usize,
    numBins: u32,
) !kernels.MinMax {
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
    var minmax_lum = try reduceMinMaxLum(stream, d_xyY);

    const lumHisto = try cuda.ZigKernel(kernels, "lumHisto").init();
    var d_histo = try cuda.alloc(c_uint, numBins);
    try cuda.memset(c_uint, d_histo, 0);
    try lumHisto.launch(
        stream,
        cuda.Grid.init1D(num_pixels, 1024),
        .{ d_histo, d_xyY, minmax_lum },
    );

    const computeCdf = try cuda.ZigKernel(kernels, "blellochCdf").init();
    try computeCdf.launch(
        stream,
        cuda.Grid.init1D(numBins, numBins),
        .{ d_cdf, d_histo },
    );
    stream.synchronize();

    return minmax_lum;
}

fn reduceMinMaxLum(
    stream: *cuda.Stream,
    d_xyY: []const f32,
) !kernels.MinMax {
    // TODO: the results seems to change between runs
    const num_pixels = d_xyY.len;
    const reduce_minmax_lum = try cuda.ZigKernel(kernels, "reduceMinmaxLum").init();

    const grid = cuda.Grid.init1D(num_pixels, 1024);
    var d_buff = try cuda.alloc(kernels.MinMax, grid.blocks.x);
    defer cuda.free(d_buff);
    var d_min_max_lum = try cuda.alloc(kernels.MinMax, 1);
    // try cuda.memsetD8(kernels.MinMax, d_min_max_lum, 0xaa);
    defer cuda.free(d_min_max_lum);

    try reduce_minmax_lum.launchWithSharedMem(
        stream,
        grid,
        grid.threads.x * @sizeOf(kernels.MinMax),
        .{ d_xyY, d_buff },
    );

    const one_block = cuda.Grid.init1D(d_buff.len, 0);
    const reduce_minmax = try cuda.ZigKernel(kernels, "reduceMinmax").init();
    try reduce_minmax.launchWithSharedMem(
        stream,
        one_block,
        one_block.threads.x * @sizeOf(kernels.MinMax),
        .{ d_buff, d_min_max_lum },
    );
    var minmax_lum = stream.copyResult(kernels.MinMax, &d_min_max_lum[0]);

    try std.testing.expect(minmax_lum.min < minmax_lum.max);
    return minmax_lum;
}

test "histogram" {
    var stream = try cuda.Stream.init(0);
    var img = [_]f32{
        0.0, 0.0, -1.0,
        0.0, 0.0, 0.0,
        0.0, 0.0, 1.0,
        0.0, 0.0, 2.0,
        0.0, 0.0, 3.0,
        0.0, 0.0, 4.0,
        0.0, 0.0, 6.0,
        0.0, 0.0, 7.0,
        0.0, 0.0, 8.0,
        0.0, 0.0, 9.0,
        0.0, 0.0, 3.0,
        0.0, 0.0, 3.0,
        0.0, 0.0, 3.0,
        0.0, 0.0, 9.0,
        0.0, 0.0, 10.0,
    };
    const lumHisto = try cuda.ZigKernel(kernels, "lumHisto").init();
    var bins = [_]c_uint{0} ** 10;
    const d_img = try cuda.allocAndCopy(f32, &img);
    var d_bins = try cuda.allocAndCopy(c_uint, &bins);
    try lumHisto.launch(
        &stream,
        cuda.Grid.init1D(img.len, 3),
        .{ d_bins, d_img, .{ .min = 0, .max = 9 } },
    );
    try cuda.memcpyDtoH(c_uint, &bins, d_bins);

    std.log.warn("bins: {any}", .{bins});
    try std.testing.expectEqual(
        [10]c_uint{ 2, 1, 1, 4, 1, 0, 1, 1, 1, 3 },
        bins,
    );
}

const ReduceMinmaxLum = cuda.ZigKernel(kernels, "reduceMinmaxLum");

fn test_min_max_lum(stream: *cuda.Stream, f: ReduceMinmaxLum, d_img: []f32, expected: []const kernels.MinMax) !void {
    var num_blocks = expected.len;
    var d_minmax = try cuda.alloc(kernels.MinMax, num_blocks);
    defer cuda.free(d_minmax);

    var grid1D = cuda.Grid{
        .blocks = .{ .x = @intCast(c_uint, num_blocks) },
        .threads = .{ .x = @intCast(c_uint, std.math.divCeil(usize, d_img.len, 3 * num_blocks) catch unreachable) },
    };
    try f.launchWithSharedMem(
        stream,
        grid1D,
        @sizeOf(kernels.MinMax) * grid1D.threads.x,
        .{ d_img, d_minmax },
    );
    var minmax = try cuda.allocAndCopyResult(kernels.MinMax, testing.allocator, d_minmax);
    defer testing.allocator.free(minmax);
    std.log.warn("minmax ({}x{}): {any}", .{ grid1D.blocks.x, grid1D.threads.x, minmax });
    for (expected) |exp, index| {
        std.testing.expectEqual(exp, minmax[index]) catch |err| {
            log.err("At index {} expected {any} got ({any})", .{ index, exp, minmax[index] });
            return err;
        };
    }
}

test "minmax_lum" {
    var stream = try cuda.Stream.init(0);
    var img = [_]f32{
        -100, 100, 0,
        0,    0,   3,
        0,    0,   0,
        0,    0,   1,
        0,    0,   2,
        0,    0,   4,
        0,    0,   6,
        0,    0,   7,
        0,    0,   8,
        0,    0,   9,
        0,    0,   10,
        0,    0,   3,
        0,    0,   3,
        0,    0,   3,
        0,    0,   9,
        0,    0,   3,
    };
    const f = try ReduceMinmaxLum.init();
    var d_img = try cuda.allocAndCopy(f32, &img);

    try test_min_max_lum(&stream, f, d_img, &[_]kernels.MinMax{
        .{ .min = 0, .max = 7 },
        .{ .min = 3, .max = 10 },
    });
    try test_min_max_lum(&stream, f, d_img, &[_]kernels.MinMax{
        .{ .min = 0, .max = 3 },
        .{ .min = 2, .max = 7 },
        .{ .min = 3, .max = 10 },
        .{ .min = 3, .max = 9 },
    });
}

test "cdf" {
    var stream = try cuda.Stream.init(0);
    inline for ([_][:0]const u8{"blellochCdf"}) |variant| {
        log.warn("Testing Cdf implementation: {s}", .{variant});
        var bins = [_]c_uint{ 2, 1, 1, 4, 1, 0, 1, 1, 1, 3 };
        log.warn("original bins: {d}", .{bins});
        // var bins = [_]c_uint{ 2, 1 };
        const computeCdf = try cuda.ZigKernel(kernels, variant).init();
        var cdf = [_]f32{0.0} ** 10;
        var d_bins = try stream.allocAndCopy(c_uint, &bins);
        var d_cdf = try stream.allocAndCopy(f32, &cdf);
        try computeCdf.launch(&stream, cuda.Grid.init1D(bins.len, 0), .{ d_cdf, d_bins });
        stream.memcpyDtoH(f32, &cdf, d_cdf);
        stream.memcpyDtoH(c_uint, &bins, d_bins);

        var t: f32 = 15.0;
        const tgt_cdf = [10]f32{ 0, 2 / t, 3 / t, 4 / t, 8 / t, 9 / t, 9 / t, 10 / t, 11 / t, 12 / t };
        log.warn("bins: {d}", .{bins});
        log.warn("tgt_cdf: {d:.3}", .{tgt_cdf});
        log.warn("cdf: {d:.3}", .{cdf});
        var i: u8 = 0;
        while (i < bins.len) : (i += 1) {
            try std.testing.expectApproxEqRel(tgt_cdf[i], cdf[i], 0.0001);
        }
    }
}
