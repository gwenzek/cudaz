const std = @import("std");
const log = std.log;
const assert = std.debug.assert;

const cuda = @import("cuda");
const cu = cuda.cu;

const hw1_kernel = @import("hw1_kernel.zig");
const png = @import("png.zig");
const utils = @import("utils.zig");

const hw1_ptx = @embedFile("hw1_ptx");

const resources_dir = "resources/hw1_resources/";

pub fn main() anyerror!void {
    log.info("***** HW1 ******", .{});

    var debug_allocator: std.heap.DebugAllocator(.{}) = .init;
    defer _ = debug_allocator.deinit();
    const allocator = debug_allocator.allocator();
    const args = try std.process.argsAlloc(allocator);
    defer std.process.argsFree(allocator, args);

    var stream = try cuda.Stream.init(0);
    defer stream.deinit();

    const module: *cuda.Module = .initFromData(hw1_ptx);
    defer module.deinit();

    var img = try png.fromFilePath(allocator, resources_dir ++ "cinque_terre_small.png");
    defer img.deinit(allocator);
    const max_show: usize = 12;
    log.info("Loaded img {}x{}: ({any}...)", .{ img.width, img.height, img.pixels.rgb24[200 .. 200 + max_show] });

    const d_img = try stream.allocAndCopy(u8, img.rawBytes());
    defer stream.free(d_img);

    const d_gray = try stream.alloc(u8, img.width * img.height);
    // stream.memset(u8, d_gray, 0xff);
    defer stream.free(d_gray);

    const rgba_to_gray: cuda.Kernel(hw1_kernel, "rgba_to_greyscale") = try .init(module);

    var timer = cuda.GpuTimer.start(stream);

    try rgba_to_gray.launch(
        stream,
        .init1D(img.height * img.width, 32),
        .{ d_img.ptr, d_gray.ptr, d_gray.len },
    );
    timer.stop();

    var gray = try png.grayscale(allocator, img.width, img.height);
    defer gray.deinit(allocator);
    stream.memcpyDtoH(u8, @ptrCast(gray.pixels.grayscale8), d_gray);
    stream.synchronize();

    log.info("Got grayscale img {}x{}: ({any}...)", .{ img.width, img.height, std.mem.sliceAsBytes(gray.pixels.grayscale8[200 .. 200 + @divExact(max_show, 3)]) });
    try png.writeToFilePath(gray, resources_dir ++ "output.png");
    try utils.validate_output(allocator, resources_dir, 1.0);
}
