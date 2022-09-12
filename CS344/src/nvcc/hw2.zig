const std = @import("std");
const log = std.log;
const math = std.math;
const assert = std.debug.assert;

const cuda = @import("cudaz");
const cu = cuda.cu;

const png = @import("png.zig");
const utils = @import("utils.zig");

const resources_dir = "resources/hw2_resources/";

const Rgb24 = png.Rgb24;
const Gray8 = png.Gray8;

pub fn main() anyerror!void {
    log.info("***** HW2 ******", .{});

    var general_purpose_allocator = std.heap.GeneralPurposeAllocator(.{}){};
    const alloc = general_purpose_allocator.allocator();
    try all_tests(alloc);
    const args = try std.process.argsAlloc(alloc);
    defer std.process.argsFree(alloc, args);

    var stream = try cuda.Stream.init(0);
    defer stream.deinit();

    const img = try png.Image.fromFilePath(alloc, resources_dir ++ "cinque_terre_small.png");
    defer img.deinit();
    assert(img.px == .rgb24);
    var max_show: usize = 10;
    log.info("Loaded img {}x{}: ({any}...)", .{ img.width, img.height, std.mem.sliceAsBytes(img.px.rgb24[200 .. 200 + max_show]) });

    var d_img = try cuda.allocAndCopy(Rgb24, img.px.rgb24);
    defer cuda.free(d_img);

    var d_red = try cuda.alloc(Gray8, img.width * img.height);
    defer cuda.free(d_red);
    var d_green = try cuda.alloc(Gray8, img.width * img.height);
    defer cuda.free(d_green);
    var d_blue = try cuda.alloc(Gray8, img.width * img.height);
    defer cuda.free(d_blue);

    var d_red_blured = try cuda.alloc(Gray8, img.width * img.height);
    defer cuda.free(d_red_blured);
    var d_green_blured = try cuda.alloc(Gray8, img.width * img.height);
    defer cuda.free(d_green_blured);
    var d_blue_blured = try cuda.alloc(Gray8, img.width * img.height);
    defer cuda.free(d_blue_blured);
    const d_buffers = [_][2]@TypeOf(d_red){
        .{ d_red, d_red_blured },
        .{ d_green, d_green_blured },
        .{ d_blue, d_blue_blured },
    };

    var d_out = try cuda.alloc(Rgb24, img.width * img.height);
    defer cuda.free(d_out);

    const separateChannels = try cuda.CudaKernel("separateChannels").init();
    const gaussianBlur = try cuda.CudaKernel("gaussian_blur").init();
    const recombineChannels = try cuda.CudaKernel("recombineChannels").init();

    var d_filter = try cuda.allocAndCopy(f32, &blurFilter());
    defer cuda.free(d_filter);

    var grid2D = cuda.Grid.init2D(img.height, img.width, 32, 32);
    try separateChannels.launch(
        &stream,
        grid2D,
        .{
            @ptrCast([*c]const cu.uchar3, d_img.ptr),
            @intCast(c_int, img.height),
            @intCast(c_int, img.width),
            @ptrCast([*c]u8, d_red.ptr),
            @ptrCast([*c]u8, d_green.ptr),
            @ptrCast([*c]u8, d_blue.ptr),
        },
    );

    var timer = cuda.GpuTimer.start(&stream);
    for (d_buffers) |d_src_tgt| {
        try gaussianBlur.launch(
            &stream,
            grid2D,
            .{
                @ptrCast([*c]const u8, d_src_tgt[0].ptr),
                @ptrCast([*c]u8, d_src_tgt[1].ptr),
                @intCast(c_uint, img.height),
                @intCast(c_uint, img.width),
                @ptrCast([*c]const f32, d_filter),
                @intCast(c_int, blurKernelWidth),
            },
        );
    }
    timer.stop();
    var h_red = try png.grayscale(alloc, img.width, img.height);
    try cuda.memcpyDtoH(Gray8, h_red.px.gray8, d_red_blured);
    try h_red.writeToFilePath(resources_dir ++ "output_red.png");

    try recombineChannels.launch(
        &stream,
        grid2D,
        .{
            @ptrCast([*c]const u8, d_red_blured.ptr),
            @ptrCast([*c]const u8, d_green_blured.ptr),
            @ptrCast([*c]const u8, d_blue_blured.ptr),
            @ptrCast([*c]cu.uchar3, d_out.ptr),
            @intCast(c_int, img.height),
            @intCast(c_int, img.width),
        },
    );

    try cuda.memcpyDtoH(Rgb24, img.px.rgb24, d_out);
    try img.writeToFilePath(resources_dir ++ "output.png");
    try utils.validate_output(alloc, resources_dir, 2.0);
}

const blurKernelWidth = 9;
fn blurFilter() [blurKernelWidth * blurKernelWidth]f32 {
    const blurKernelSigma = 2.0;

    // create and fill the filter we will convolve with
    var filter: [blurKernelWidth * blurKernelWidth]f32 = undefined;
    var filterSum: f32 = 0.0; // for normalization

    const halfWidth: i8 = @divTrunc(blurKernelWidth, 2);
    var r: i8 = -halfWidth;
    while (r <= halfWidth) : (r += 1) {
        var c: i8 = -halfWidth;
        while (c <= halfWidth) : (c += 1) {
            const filterValue: f32 = math.exp(-@intToFloat(f32, c * c + r * r) /
                (2.0 * blurKernelSigma * blurKernelSigma));
            filter[@intCast(usize, (r + halfWidth) * blurKernelWidth + c + halfWidth)] = filterValue;
            filterSum += filterValue;
        }
    }

    const normalizationFactor = 1.0 / filterSum;
    var result: f32 = 0.0;
    for (filter) |*v| {
        v.* *= normalizationFactor;
        result += v.*;
    }
    return filter;
}

fn all_tests(alloc: std.mem.Allocator) !void {
    var arena_alloc = std.heap.ArenaAllocator.init(alloc);
    defer arena_alloc.deinit();
    try test_gaussianBlur(arena_alloc.allocator());
}

fn test_gaussianBlur(alloc: std.mem.Allocator) !void {
    // const blur_kernel_size = blurKernelWidth * blurKernelWidth;
    var cols: c_uint = 50;
    var rows: c_uint = 100;
    var img: []u8 = try alloc.alloc(u8, @intCast(usize, rows * cols));
    std.mem.set(u8, img, 100);
    var out: []u8 = try alloc.alloc(u8, @intCast(usize, rows * cols));
    std.mem.set(u8, out, 0);
    cu.blockDim = cu.dim3{ .x = cols, .y = rows, .z = 1 };
    cu.blockIdx = cu.dim3{ .x = 0, .y = 0, .z = 0 };
    cu.threadIdx = cu.dim3{ .x = 10, .y = 10, .z = 0 };
    cu.gaussian_blur(img.ptr, out.ptr, rows, cols, &blurFilter(), blurKernelWidth);
    try std.testing.expectEqual(out[cols * 10 + 10], 100);

    cu.threadIdx = cu.dim3{ .x = cols - 1, .y = 10, .z = 0 };
    cu.gaussian_blur(img.ptr, out.ptr, rows, cols, &blurFilter(), blurKernelWidth);
    try std.testing.expectEqual(out[cols * 11 - 1], 100);
    try std.testing.expectEqual(out[cols * 11 - 1], 100);
}

fn recombine(
    img: png.Image,
    red: []const Gray8,
    green: []const Gray8,
    blue: []const Gray8,
) !void {
    for (img.px) |_, i| {
        img.px.rgb24[i] = Rgb24(red[i], green[i], blue[i]);
    }
}
