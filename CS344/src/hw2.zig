const std = @import("std");
const log = std.log;
const math = std.math;
const assert = std.debug.assert;

const cuda = @import("cudaz");
const cu = cuda.cu;

const png = @import("png.zig");
const utils = @import("utils.zig");
const kernels = @import("hw2_kernel.zig");
const Mat3 = kernels.Mat3;
const Mat2Float = kernels.Mat2Float;

const resources_dir = "resources/hw2_resources/";

const Rgb24 = png.Rgb24;
const Gray8 = png.Grayscale8;

pub fn main() anyerror!void {
    log.info("***** HW2 ******", .{});

    var general_purpose_allocator = std.heap.GeneralPurposeAllocator(.{}){};
    const alloc = general_purpose_allocator.allocator();
    const args = try std.process.argsAlloc(alloc);
    defer std.process.argsFree(alloc, args);

    var stream = try cuda.Stream.init(0);
    defer stream.deinit();

    const img = try png.Image.fromFilePath(alloc, resources_dir ++ "cinque_terre_small.png");
    defer img.deinit();
    log.info("Loaded {}", .{img});

    var d_img = try cuda.allocAndCopy(Rgb24, img.px.rgb24);
    defer cuda.free(d_img);

    var d_out = try cuda.alloc(Rgb24, img.width * img.height);
    defer cuda.free(d_out);

    const d_filter = Mat2Float{
        .data = (try cuda.allocAndCopy(f32, &blurFilter())).ptr,
        .shape = [_]i32{ blur_kernel_width, blur_kernel_width },
    };
    defer cuda.free(d_filter.data[0..@intCast(usize, d_filter.shape[0])]);
    var img_mat = Mat3{
        .data = std.mem.sliceAsBytes(d_img).ptr,
        .shape = [3]u32{ @intCast(u32, img.height), @intCast(u32, img.width), 3 },
    };
    var grid3D = cuda.Grid.init3D(img.height, img.width, 3, 32, 32, 1);
    var timer = cuda.GpuTimer.start(&stream);

    // Here, we compares 3 ways of making a gaussianBlur kernel:
    // by using a c-like API, by passing one struct with all args,
    // and a more fluent version that uses 3 "Matrix" struct.
    const gaussianBlurVerbose = try cuda.ZigKernel(kernels, "gaussianBlurVerbose").init();
    try gaussianBlurVerbose.launch(
        &stream,
        grid3D,
        .{
            img_mat.data,
            img_mat.shape[0],
            img_mat.shape[1],
            d_filter.data[0 .. blur_kernel_width * blur_kernel_width],
            @intCast(i32, blur_kernel_width),
            std.mem.sliceAsBytes(d_out).ptr,
        },
    );
    // const blur_args = kernels.GaussianBlurArgs{
    //     .img = img_mat,
    //     .filter = d_filter.data[0 .. blur_kernel_width * blur_kernel_width],
    //     .filter_width = @intCast(i32, blur_kernel_width),
    //     .output = std.mem.sliceAsBytes(d_out).ptr,
    // };

    // log.info("arg.img.data={*}", .{blur_args.img.data});
    // log.info("arg.img.shape={any}", .{blur_args.img.shape});
    // log.info("arg.filter={*}", .{blur_args.filter});
    // log.info("arg.filter_width={}", .{blur_args.filter_width});
    // log.info("arg.output={*}", .{blur_args.output});

    // const gaussianBlurStruct = try cuda.ZigKernel(kernels, "gaussianBlurStruct").init();
    // try gaussianBlurStruct.launch(
    //     &stream,
    //     grid3D,
    //     .{blur_args},
    // );
    // stream.synchronize();
    // const gaussianBlur = try cuda.ZigKernel(kernels, "gaussianBlur").init();
    // try gaussianBlur.launch(
    //     &stream,
    //     grid3D,
    //     .{
    //         img_mat,
    //         d_filter,
    //         std.mem.sliceAsBytes(d_out),
    //     },
    // );
    timer.stop();
    try cuda.memcpyDtoH(Rgb24, img.px.rgb24, d_out);
    try img.writeToFilePath(resources_dir ++ "output.png");
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
