const std = @import("std");
const log = std.log;
const assert = std.debug.assert;

const zigimg = @import("zigimg");
const cudaz = @import("cudaz");
const Cuda = cudaz.Cuda;
const cu = cudaz.cu;

const png = @import("png.zig");
const utils = @import("utils.zig");

const resources_dir = "CS344/hw1_resources/";

pub fn main() anyerror!void {
    log.info("***** HW1 ******", .{});

    var general_purpose_allocator = std.heap.GeneralPurposeAllocator(.{}){};
    const alloc = &general_purpose_allocator.allocator;
    const args = try std.process.argsAlloc(alloc);
    defer std.process.argsFree(alloc, args);

    var cuda = try Cuda.init(0);
    defer cuda.deinit();

    const img = try zigimg.Image.fromFilePath(alloc, resources_dir ++ "cinque_terre_small.png");
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

    var timer = cudaz.GpuTimer.init(&cuda);
    const kernel = try cudaz.Function("rgba_to_greyscale").init(&cuda);
    timer.start();
    try kernel.launch(
        cudaz.Grid.init1D(img.height * img.width, 64),
        .{
            @ptrCast([*c]const cu.uchar3, d_img.ptr),
            @ptrCast([*c]u8, d_gray.ptr),
            @intCast(c_int, img.height),
            @intCast(c_int, img.width),
        },
    );
    timer.stop();

    try cuda.memcpyDtoH(Gray8, gray.pixels.?.Grayscale8, d_gray);
    try png.writePngToFilePath(gray, resources_dir ++ "output.png");
    try utils.validate_output(alloc, resources_dir);
}
