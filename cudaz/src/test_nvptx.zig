const std = @import("std");
const testing = std.testing;

const cuda = @import("cudaz");
const nvptx = @import("nvptx.zig");

test "hello_world" {
    var stream = try cuda.Stream.init(0);
    defer stream.deinit();
    var d_buffer = try cuda.alloc(u8, 20);
    defer cuda.free(d_buffer);

    const gpu_hello = try cuda.ZigKernel(nvptx, "test_hello_world").init();
    try gpu_hello.launch(&stream, cuda.Grid.init1D(d_buffer.len, 0), .{d_buffer});
    var h_buffer = try stream.allocAndCopyResult(u8, testing.allocator, d_buffer);
    defer testing.allocator.free(h_buffer);

    var expected = "Hello World !";
    std.log.warn("{s}", .{h_buffer});
    try testing.expectEqualSlices(u8, expected, h_buffer[0..expected.len]);
}

test "swap2_with_shared_memory" {
    var stream = try cuda.Stream.init(0);
    defer stream.deinit();
    var d_src = try stream.allocAndCopy(u8, "Hello Boobaz !");
    defer stream.free(d_src);
    var d_tgt = try stream.alloc(u8, d_src.len);
    defer stream.free(d_tgt);

    const gpu_swap = try cuda.ZigKernel(nvptx, "test_swap2_with_shared_buff").init();

    const grid = cuda.Grid.init1D(d_src.len, 2);
    try gpu_swap.launchWithSharedMem(&stream, grid, 2, .{ d_src, d_tgt });
    var h_buf = try stream.allocAndCopyResult(u8, testing.allocator, d_tgt);
    defer testing.allocator.free(h_buf);
    stream.synchronize();

    var expected = "eHll ooBboza! ";
    // std.log.warn("{s}", .{h_buf});
    try testing.expectEqualSlices(u8, h_buf, expected);
}
