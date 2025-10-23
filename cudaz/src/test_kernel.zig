const std = @import("std");

const cuda = @import("cudaz");
const nvptx = @import("nvptx");

const generated_ptx = @embedFile("generated_ptx");
const message = "Hello World !\x00";

fn hello_world(out: [*]u8, len: u32) callconv(nvptx.kernel) void {
    const i = nvptx.getIdX();
    if (i > len) return;
    out[i] = if (i > message.len) 0 else message[i];
}

comptime {
    if (nvptx.is_nvptx) {
        @export(&hello_world, .{ .name = "hello_world" });
    }
}

test hello_world {
    var stream = try cuda.Stream.init(1);
    defer stream.deinit();
    const d_buffer = try cuda.alloc(u8, 20);
    defer cuda.free(d_buffer);

    const module = cuda.loadModule(.{ .embed = generated_ptx });
    const gpu_hello_world: cuda.FnStruct("hello_world", hello_world) = try .init(module);
    try gpu_hello_world.launch(&stream, .init1D(d_buffer.len, 0), .{ d_buffer.ptr, @intCast(d_buffer.len) });
    var h_buffer = try stream.allocAndCopyResult(u8, std.testing.allocator, d_buffer);
    defer std.testing.allocator.free(h_buffer);

    const expected = "Hello World !";
    stream.synchronize();
    std.log.warn("GPU says: {s}", .{h_buffer});
    try std.testing.expectEqualSlices(u8, expected, h_buffer[0..expected.len]);
}
