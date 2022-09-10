const std = @import("std");
const log = std.log;
const assert = std.debug.assert;

const cudaz = @import("cudaz");
const cu = cudaz.cu;

const png = @import("png.zig");
const utils = @import("utils.zig");
// const hw1_kernel = @import("hw1_kernel.zig");

const resources_dir = "resources/hw1_resources/";

pub fn main() anyerror!void {
    log.info("***** HW1 ******", .{});

    var general_purpose_allocator = std.heap.GeneralPurposeAllocator(.{}){};
    var alloc = general_purpose_allocator.allocator();
    const args = try std.process.argsAlloc(alloc);
    defer std.process.argsFree(alloc, args);

    var stream = try cudaz.Stream.init(0);
    defer stream.deinit();

    const img = try png.Image.fromFilePath(alloc, resources_dir ++ "cinque_terre_small.png");
    defer img.deinit();
    log.info("Loaded {}", .{img});

    var d_img = try cudaz.allocAndCopy(png.Rgb24, img.px.rgb24);
    defer cudaz.free(d_img);

    var gray = try png.grayscale(alloc, img.width, img.height);
    defer gray.deinit();
    var d_gray = try cudaz.alloc(png.Gray8, img.width * img.height);
    defer cudaz.free(d_gray);
    try cudaz.memset(png.Gray8, d_gray, 0);

    const kernel = try cudaz.CudaKernel("rgba_to_greyscale").init();
    // const kernel = try cudaz.ZigStruct(hw1_kernel, "rgba_to_greyscale").init();
    var timer = cudaz.GpuTimer.start(&stream);
    try kernel.launch(
        &stream,
        cudaz.Grid.init1D(img.height * img.width, 64),
        .{
            @ptrCast([*c]const cu.uchar3, d_img.ptr),
            @ptrCast([*c]u8, d_gray.ptr),
            @intCast(c_int, img.height),
            @intCast(c_int, img.width),
            // std.mem.sliceAsBytes(d_img),
            // std.mem.sliceAsBytes(d_gray),
        },
    );
    timer.stop();
    stream.synchronize();
    try cudaz.memcpyDtoH(png.Gray8, gray.px.gray8, d_gray);
    log.info("Got grayscale {}", .{img});
    try gray.writeToFilePath(resources_dir ++ "output.png");
    try utils.validate_output(alloc, resources_dir, 1.0);
}
