const std = @import("std");
const log = std.log;
const assert = std.debug.assert;

const zigimg = @import("zigimg");
const cuda_module = @import("cuda");
const Cuda = cuda_module.Cuda;
const cu = cuda_module.cu;

const png = @import("png.zig");
const utils = @import("utils.zig");

const resources_dir = "HW1/hw1_resources/";

pub fn main() anyerror!void {
    var general_purpose_allocator = std.heap.GeneralPurposeAllocator(.{}){};
    const alloc = &general_purpose_allocator.allocator;
    const args = try std.process.argsAlloc(alloc);
    defer std.process.argsFree(alloc, args);

    var cuda = try Cuda.init(0);
    defer cuda.deinit();

    const img = try zigimg.Image.fromFilePath(alloc, "HW1/cinque_terre_small.png");
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
    var gray = try utils.grayscale(alloc, img.width, img.height);
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
    try utils.validate_output(alloc, resources_dir);
}
