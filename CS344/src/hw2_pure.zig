const std = @import("std");
const log = std.log;
const math = std.math;
const assert = std.debug.assert;

const zigimg = @import("zigimg");
const cuda = @import("cudaz");
const cu = cuda.cu;

const png = @import("png.zig");
const utils = @import("utils.zig");
const kernels = @import("hw2_pure_kernel.zig");

const resources_dir = "resources/hw2_resources/";

const Rgb24 = zigimg.color.Rgb24;
const Gray8 = zigimg.color.Grayscale8;

pub fn main() anyerror!void {
    log.info("***** HW2 ******", .{});

    var general_purpose_allocator = std.heap.GeneralPurposeAllocator(.{}){};
    const alloc = &general_purpose_allocator.allocator;
    const args = try std.process.argsAlloc(alloc);
    defer std.process.argsFree(alloc, args);

    var stream = try cuda.Stream.init(0);
    defer stream.deinit();

    const img = try zigimg.Image.fromFilePath(alloc, resources_dir ++ "cinque_terre_small.png");
    defer img.deinit();
    assert(img.image_format == .Png);
    var max_show: usize = 10;
    log.info("Loaded img {}x{}: ({any}...)", .{ img.width, img.height, std.mem.sliceAsBytes(img.pixels.?.Rgb24[200 .. 200 + max_show]) });

    var d_img = try cuda.allocAndCopy(Rgb24, img.pixels.?.Rgb24);
    defer cuda.free(d_img);

    var d_out = try cuda.alloc(Rgb24, img.width * img.height);
    defer cuda.free(d_out);

    var timer = cuda.GpuTimer.init(&stream);
    const gaussianBlur = try cuda.FnStruct("gaussianBlur", kernels.gaussianBlur).init();

    var d_filter = try cuda.allocAndCopy(f32, &blurFilter());
    defer cuda.free(d_filter);

    var grid3D = cuda.Grid.init3D(img.width, img.height, 3, 32, 32, 1);
    try gaussianBlur.launch(
        &stream,
        grid3D,
        .{
            std.mem.sliceAsBytes(d_img),
            std.mem.sliceAsBytes(d_out),
            @intCast(i32, img.width),
            @intCast(i32, img.height),
            d_filter,
            @intCast(i32, blur_kernel_width),
        },
    );
    timer.stop();
    try cuda.memcpyDtoH(Rgb24, img.pixels.?.Rgb24, d_out);
    try png.writePngToFilePath(img, resources_dir ++ "output.png");
    try utils.validate_output(alloc, resources_dir, 2.0);
}

const blur_kernel_width = 9;
fn blurFilter() [blur_kernel_width * blur_kernel_width]f32 {
    const blurKernelSigma = 2.0;

    // create and fill the filter we will convolve with
    var filter: [blur_kernel_width * blur_kernel_width]f32 = undefined;
    var filterSum: f32 = 0.0; // for normalization

    const halfWidth: i8 = @divTrunc(blur_kernel_width, 2);
    var r: i8 = -halfWidth;
    while (r <= halfWidth) : (r += 1) {
        var c: i8 = -halfWidth;
        while (c <= halfWidth) : (c += 1) {
            const filterValue: f32 = math.exp(-@intToFloat(f32, c * c + r * r) /
                (2.0 * blurKernelSigma * blurKernelSigma));
            filter[@intCast(usize, (r + halfWidth) * blur_kernel_width + c + halfWidth)] = filterValue;
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
