const std = @import("std");
const log = std.log;
const assert = std.debug.assert;

const cudaz = @import("cudaz");
const cu = cudaz.cu;

const png = @import("png.zig");
const utils = @import("utils.zig");
const hw1_kernel = @import("hw1_kernel.zig");

const resources_dir = "resources/hw1_resources/";

pub fn main() anyerror!void {
    log.info("***** HW1 ******", .{});

    var general_purpose_allocator = std.heap.GeneralPurposeAllocator(.{}){};
    const alloc = general_purpose_allocator.allocator();
    const args = try std.process.argsAlloc(alloc);
    defer std.process.argsFree(alloc, args);

    var stream = try cudaz.Stream.init(0);
    defer stream.deinit();

    const img = try png.Image.fromFilePath(alloc, resources_dir ++ "cinque_terre_small.png");
    defer img.deinit();
    var max_show: usize = 12;
    log.info("Loaded img {}x{}: ({any}...)", .{ img.width, img.height, std.mem.sliceAsBytes(img.px.rgb24[200 .. 200 + max_show]) });

    const Rgb24 = png.Rgb24;
    var d_img = try cudaz.allocAndCopy(Rgb24, img.px.rgb24);
    defer cudaz.free(d_img);

    const Gray8 = png.Gray8;
    var gray = try png.grayscale(alloc, img.width, img.height);
    defer gray.deinit();
    var d_gray = try cudaz.alloc(Gray8, img.width * img.height);
    defer cudaz.free(d_gray);
    try cudaz.memset(Gray8, d_gray, 0);

    const kernel = try cudaz.ZigKernel(hw1_kernel, "rgbaToGreyscale").init();
    var timer = cudaz.GpuTimer.start(&stream);
    try kernel.launch(
        &stream,
        cudaz.Grid.init1D(img.height * img.width, 64),
        .{ std.mem.sliceAsBytes(d_img), std.mem.sliceAsBytes(d_gray) },
    );
    timer.stop();
    try cudaz.memcpyDtoH(Gray8, gray.px.gray8, d_gray);
    log.info("Got grayscale img {}x{}: ({any}...)", .{ img.width, img.height, std.mem.sliceAsBytes(gray.px.gray8[200 .. 200 + @divExact(max_show, 3)]) });
    try gray.writeToFilePath(resources_dir ++ "output.png");
    try utils.validate_output(alloc, resources_dir, 1.0);
}
