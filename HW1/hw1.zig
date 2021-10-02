const std = @import("std");
const log = std.log;
const assert = std.debug.assert;
const zigimg = @import("zigimg");
const png = @import("png.zig");

const cuda_module = @import("cuda");
const Cuda = cuda_module.Cuda;
const cu = cuda_module.cu;
const Image = zigimg.Image;

const resources_dir = "HW1/hw1_resources/";

pub fn main() anyerror!void {
    var general_purpose_allocator = std.heap.GeneralPurposeAllocator(.{}){};
    const alloc = &general_purpose_allocator.allocator;
    const args = try std.process.argsAlloc(alloc);
    defer std.process.argsFree(alloc, args);

    var cuda = try Cuda.init(0);
    defer cuda.deinit();

    const img = try Image.fromFilePath(alloc, "HW1/cinque_terre_small.png");
    defer img.deinit();
    assert(img.image_format == .Png);
    var max_show: usize = 10;
    log.info("Loaded img {}x{}: ({any}...)", .{ img.width, img.height, std.mem.sliceAsBytes(img.pixels.?.Rgb24[200 .. 200 + max_show]) });
    // try img.writeToFilePath("HW1/output.pbm", .Pbm, .{ .pbm = .{ .binary = false } });

    const Rgb24 = zigimg.color.Rgb24;
    var d_img = try cuda.alloc(Rgb24, img.width * img.height);
    defer cuda.free(d_img);
    try cuda.memcpyHtoD(Rgb24, d_img, img.pixels.?.Rgb24);

    const Gray8 = zigimg.color.Grayscale8;
    var gray = try grayscale(alloc, img.width, img.height);
    defer gray.deinit();
    var d_gray = try cuda.alloc(Gray8, img.width * img.height);
    defer cuda.free(d_gray);
    try cuda.memset(Gray8, d_gray, Gray8{ .value = 0 });

    var timer = cuda_module.GpuTimer.init(&cuda);
    const kernel = try cuda_module.KernelSignature("./cudaz/kernel.ptx", "rgba_to_greyscale").init(&cuda);
    timer.start();
    try kernel.launch(
        .{ .x = @intCast(c_uint, img.width), .y = @intCast(c_uint, img.height) },
        .{},
        .{
            .@"0" = @ptrCast([*c]const cu.uchar3, d_img.ptr),
            .@"1" = @ptrCast([*c]u8, d_gray.ptr),
            .@"2" = @intCast(c_int, img.width),
            .@"3" = @intCast(c_int, img.height),
        },
    );
    timer.stop();

    try cuda.memcpyDtoH(Gray8, gray.pixels.?.Grayscale8, d_gray);
    try png.writePngToFilePath(gray, resources_dir ++ "output.png");
    try validate_output(alloc);
}

fn grayscale(allocator: *std.mem.Allocator, width: usize, height: usize) !Image {
    var img = Image{
        .allocator = allocator,
        .width = width,
        .height = height,
        .pixels = try zigimg.color.ColorStorage.init(allocator, .Grayscale8, width * height),
        .image_format = .Png,
    };

    return img;
}

fn validate_output(alloc: *std.mem.Allocator) !void {
    const output = try Image.fromFilePath(alloc, resources_dir ++ "output.png");
    const reference = try Image.fromFilePath(alloc, resources_dir ++ "reference.png");

    log.info("Loaded output image and reference image for comparison", .{});
    std.debug.assert(output.width == reference.width);
    std.debug.assert(output.height == reference.height);
    std.debug.assert(output.image_format == reference.image_format);
    std.debug.assert(output.pixels.?.Grayscale8.len == reference.pixels.?.Grayscale8.len);

    const img_match = try eq_and_show_diff(alloc, output, reference);
    if (img_match) {
        log.info("*** The image matches, Congrats ! ***", .{});
    } else {
        std.os.exit(1);
    }
}

fn eq_and_show_diff(alloc: *std.mem.Allocator, output: Image, reference: Image) !bool {
    var diff = try grayscale(alloc, reference.width, reference.height);
    var diff_pxls = diff.pixels.?.Grayscale8;
    var out_pxls = output.pixels.?.Grayscale8;
    var ref_pxls = reference.pixels.?.Grayscale8;

    const num_pixels = ref_pxls.len;
    var i: usize = 0;
    var min_val: i16 = 255;
    var max_val: i16 = -255;
    while (i < diff_pxls.len) : (i += 1) {
        var d: i16 = @intCast(i16, ref_pxls[i].value) - @intCast(i16, out_pxls[i].value);
        // d = try std.math.absInt(d);
        min_val = std.math.min(min_val, d);
        max_val = std.math.max(max_val, d);
    }
    i = 0;
    while (i < diff_pxls.len) : (i += 1) {
        var d: i16 = @intCast(i16, ref_pxls[i].value) - @intCast(i16, out_pxls[i].value);
        // d = try std.math.absInt(d);
        const centered_d = (255.0 * @intToFloat(f32, d - min_val)) / @intToFloat(f32, max_val - min_val);
        diff_pxls[i] = .{ .value = @floatToInt(u8, centered_d) };
    }

    try png.writePngToFilePath(diff, resources_dir ++ "output_diff.png");
    if (min_val != 0 or max_val != 0) {
        std.log.warn("Found diffs between two images, ranging from {} to {} pixel value.", .{ min_val, max_val });
        return false;
    }
    return true;
}
