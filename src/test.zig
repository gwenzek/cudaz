const std = @import("std");

const cuda = @import("cudaz");
const ptx = @import("nvptx");
pub const panic = ptx.panic;

const generated_ptx = @embedFile("generated_ptx");
const message = "Hello World !\x00";

const log = std.log.scoped(.@"test");

pub fn hello_world(out: [*]u8, len: u32) callconv(ptx.kernel) void {
    const i = ptx.getIdX();
    if (i > len) return;
    out[i] = if (i > message.len) 0 else message[i];
}

comptime {
    if (ptx.is_nvptx) {
        @export(&hello_world, .{ .name = "hello_world" });
    }
}

test hello_world {
    var stream = try cuda.Stream.init(1);
    defer stream.deinit();
    const d_buffer = try stream.alloc(u8, 20);
    defer stream.free(d_buffer);

    const module: *cuda.Module = .initFromData(generated_ptx);
    defer module.deinit();
    const gpu_hello_world: cuda.Kernel(@This(), "hello_world") = try .init(module);
    try gpu_hello_world.launch(stream, .init1D(32, 8), .{ d_buffer.ptr, @intCast(d_buffer.len) });
    var h_buffer = try stream.allocAndCopyResult(u8, std.testing.allocator, d_buffer);
    defer std.testing.allocator.free(h_buffer);

    const expected = "Hello World !";
    stream.synchronize();
    std.log.warn("GPU says: {s}", .{h_buffer});
    try std.testing.expectEqualSlices(u8, expected, h_buffer[0..expected.len]);
}

pub fn rgba_to_grayscale(rgbaImage: []const [4]u8, grayImage: []u8) callconv(ptx.kernel) void {
    const i = ptx.getIdX();
    if (i >= grayImage.len) return;
    const px = rgbaImage[i];
    const R: u16 = @intCast(px[0]);
    const G: u16 = @intCast(px[1]);
    const B: u16 = @intCast(px[2]);
    const gray: u16 = @divFloor(299 * R + 587 * G + 114 * B, 1000);
    grayImage[i] = @intCast(gray);
}

comptime {
    if (ptx.is_nvptx) {
        @export(&rgba_to_grayscale, .{ .name = "rgba_to_grayscale" });
    }
}

pub const rgba_to_grayscaleK = cuda.Kernel(@This(), "rgba_to_grayscale");

test rgba_to_grayscale {
    var stream = try cuda.Stream.init(0);
    defer stream.deinit();
    log.warn("cuda: {f}", .{stream});
    const module: *cuda.Module = .initFromData(generated_ptx);
    defer module.deinit();

    const rgba_to_grayscale_f: rgba_to_grayscaleK = try .init(module);
    const num_rows: u32 = 10;
    const num_cols: u32 = 20;
    const rgba_d = try stream.alloc([4]u8, num_rows * num_cols);
    stream.memset([4]u8, rgba_d, [4]u8{ 0xaa, 0, 0, 255 });
    const gray_d = try stream.alloc(u8, num_rows * num_cols);
    stream.memset(u8, gray_d, 0);

    var timer = cuda.GpuTimer.start(stream);
    try rgba_to_grayscale_f.launch(
        stream,
        .init1D(num_rows * num_cols, 16),
        .{ rgba_d, gray_d },
    );
    timer.stop();

    const gray_h = try stream.allocAndCopyResult(u8, std.testing.allocator, gray_d);
    defer std.testing.allocator.free(gray_h);
    stream.synchronize();

    const gray_expected: u8 = 50;
    for (0.., gray_h) |i, gray_px| {
        errdefer log.err("Error at pixel {}: got {}, expected: {}", .{ i, gray_px, gray_expected });
        errdefer log.err("Full image: {any}", .{gray_h});
        try std.testing.expectEqual(gray_expected, gray_px);
    }

    try std.testing.expect(timer.elapsed() > 0);
}

test "rgba_to_grayscale raw API" {
    var stream = try cuda.Stream.init(0);
    defer stream.deinit();

    const module: *cuda.Module = .initFromData(generated_ptx);
    defer module.deinit();
    const rgba_to_grayscale_f: rgba_to_grayscaleK = try .init(module);
    const num_rows: u32 = 10;
    const num_cols: u32 = 20;
    const rgba_d = try stream.alloc([4]u8, num_rows * num_cols);
    stream.memset([4]u8, rgba_d, [4]u8{ 0xaa, 0, 0, 255 });
    const gray_d = try stream.alloc(u8, num_rows * num_cols);
    stream.memset(u8, gray_d, 0);

    // This test uses the stream.launch api that takes argument by pointers.
    // It's a bit more error prone because of the `@ptrCast`.
    try stream.launch(
        rgba_to_grayscale_f.f,
        .init1D(num_rows * num_cols, num_rows * num_cols),
        &.{ @ptrCast(&rgba_d), @ptrCast(&gray_d) },
    );

    const gray_h = try stream.allocAndCopyResult(u8, std.testing.allocator, gray_d);
    defer std.testing.allocator.free(gray_h);
    stream.synchronize();

    const gray_expected: u8 = 50;
    for (0.., gray_h) |i, gray_px| {
        errdefer log.err("Error at pixel {}: got {}, expected: {}", .{ i, gray_px, gray_expected });
        errdefer log.err("Full image: {any}", .{gray_h});
        try std.testing.expectEqual(gray_expected, gray_px);
    }
}
