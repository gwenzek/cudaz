const std = @import("std");
const log = std.log;
const math = std.math;
const assert = std.debug.assert;

const cuda = @import("cuda");
const cu = cuda.cu;
const ptx = @import("nvptx");
pub const panic = ptx.panic;

const kernels = @import("hw2_pure_kernel.zig");
const Mat3 = kernels.Mat3;
const Mat2Float = kernels.Mat2Float;
const png = @import("png.zig");
const utils = @import("utils.zig");

const hw2_ptx = @embedFile("hw2_pure_ptx");

const resources_dir = "resources/hw2_resources/";

pub fn main() anyerror!void {
    log.info("***** HW2 ******", .{});

    var general_purpose_allocator = std.heap.GeneralPurposeAllocator(.{}){};
    const alloc = general_purpose_allocator.allocator();
    const args = try std.process.argsAlloc(alloc);
    defer std.process.argsFree(alloc, args);

    var stream = try cuda.Stream.init(0);
    defer stream.deinit();

    const module: *cuda.Module = .initFromData(hw2_ptx);
    defer module.deinit();
    const gaussianBlurVerbose: cuda.Kernel(kernels, "gaussianBlurVerbose") = try .init(module);
    const gaussianBlur: cuda.Kernel(kernels, "gaussianBlur") = try .init(module);

    var img = try png.fromFilePath(alloc, resources_dir ++ "cinque_terre_small.png");
    defer img.deinit(alloc);
    log.info("Loaded img {d}x{d} {t}", .{ img.width, img.height, img.pixels });

    const d_img = try stream.allocAndCopy(png.Rgb24, img.pixels.rgb24);
    defer stream.free(d_img);

    const d_out = try stream.alloc(png.Rgb24, img.width * img.height);
    // stream.memset(png.Rgb24, d_out, .{ .r = 0xff, .g = 0, .b = 0 });
    defer stream.free(d_out);

    const d_filter = Mat2Float{
        .data = (try stream.allocAndCopy(f32, &blurFilter())).ptr,
        .shape = .{ blur_kernel_width, blur_kernel_width },
    };
    defer stream.free(d_filter.data[0..@intCast(d_filter.shape[0])]);
    const img_mat: Mat3 = .{
        .data = std.mem.sliceAsBytes(d_img).ptr,
        .shape = .{ @intCast(img.height), @intCast(img.width), 3 },
    };
    const grid3D: cuda.Grid = .init3D(.{ img.height, img.width, 3 }, .{ 32, 32, 1 });

    var timer = cuda.GpuTimer.start(stream);
    try gaussianBlurVerbose.launch(
        stream,
        grid3D,
        .{
            img_mat.data,
            img_mat.shape[0],
            img_mat.shape[1],
            d_filter.data[0 .. blur_kernel_width * blur_kernel_width],
            @intCast(blur_kernel_width),
            std.mem.sliceAsBytes(d_out).ptr,
        },
    );
    timer.stop();
    stream.memcpyDtoH(png.Rgb24, img.pixels.rgb24, d_out);
    stream.synchronize();

    // std.log.debug("out: {any}", .{img.pixels.rgb24[0..24]});
    try png.writeToFilePath(img, resources_dir ++ "output.png");
    try utils.validate_output(alloc, resources_dir, 2.0);

    try gaussianBlur.launch(
        stream,
        grid3D,
        .{
            img_mat,
            d_filter,
            std.mem.sliceAsBytes(d_out),
        },
    );
    stream.memcpyDtoH(png.Rgb24, img.pixels.rgb24, d_out);
    stream.synchronize();
    try png.writeToFilePath(img, resources_dir ++ "output.png");
    try utils.validate_output(alloc, resources_dir, 2.0);
}

const blur_kernel_width: i32 = 9;
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
            const filterValue: f32 = math.exp(-@as(f32, @floatFromInt(c * c + r * r)) /
                (2.0 * blurKernelSigma * blurKernelSigma));
            filter[@intCast((r + halfWidth) * blur_kernel_width + c + halfWidth)] = filterValue;
            filterSum += filterValue;
        }
    }

    const normalizationFactor = 1.0 / filterSum;
    var result: f32 = 0.0;
    for (filter[0..]) |*v| {
        v.* *= normalizationFactor;
        result += v.*;
    }
    return filter;
}
