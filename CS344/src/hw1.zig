const std = @import("std");
const log = std.log;
const assert = std.debug.assert;

const zigimg = @import("zigimg");
const cudaz = @import("cudaz");
const cu = cudaz.cu;

const png = @import("png.zig");
const utils = @import("utils.zig");
// const hw1_kernel = @import("hw1_kernel.zig");

const resources_dir = "resources/hw1_resources/";

pub fn main() anyerror!void {
    log.info("***** HW1 ******", .{});

    var general_purpose_allocator = std.heap.GeneralPurposeAllocator(.{}){};
    const alloc = &general_purpose_allocator.allocator;
    const args = try std.process.argsAlloc(alloc);
    defer std.process.argsFree(alloc, args);

    var stream = try cudaz.Stream.init(0);
    defer stream.deinit();

    const img = try zigimg.Image.fromFilePath(alloc, resources_dir ++ "cinque_terre_small.png");
    defer img.deinit();
    assert(img.image_format == .Png);
    var max_show: usize = 12;
    log.info("Loaded img {}x{}: ({any}...)", .{ img.width, img.height, std.mem.sliceAsBytes(img.pixels.?.Rgb24[200 .. 200 + max_show]) });
    // try img.writeToFilePath("HW1/output.pbm", .Pbm, .{ .pbm = .{ .binary = false } });

    const Rgb24 = zigimg.color.Rgb24;
    var d_img = try cudaz.allocAndCopy(Rgb24, img.pixels.?.Rgb24);
    defer cudaz.free(d_img);

    const Gray8 = zigimg.color.Grayscale8;
    var gray = try utils.grayscale(alloc, img.width, img.height);
    defer gray.deinit();
    var d_gray = try cudaz.alloc(Gray8, img.width * img.height);
    defer cudaz.free(d_gray);
    try cudaz.memset(Gray8, d_gray, Gray8{ .value = 0 });

    const kernel = try cudaz.Function("rgba_to_greyscale").init();
    // const kernel = try cudaz.FnStruct("rgba_to_greyscale", hw1_kernel.rgba_to_greyscale).init();
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
    try cudaz.memcpyDtoH(Gray8, gray.pixels.?.Grayscale8, d_gray);
    log.info("Got grayscale img {}x{}: ({any}...)", .{ img.width, img.height, std.mem.sliceAsBytes(gray.pixels.?.Grayscale8[200 .. 200 + @divExact(max_show, 3)]) });
    try png.writePngToFilePath(gray, resources_dir ++ "output.png");
    try utils.validate_output(alloc, resources_dir, 1.0);
}
