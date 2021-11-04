const testing = @import("std").testing;
const cuda = @import("cudaz");
const nvptx = @import("nvptx.zig");

test "hello_world" {
    var stream = try cuda.Stream.init(0);
    var d_buffer = try cuda.alloc(u8, 20);

    const gpu_hello_world = try cuda.FnStruct("_test_hello_world", nvptx._test_hello_world).init();
    try gpu_hello_world.launch(&stream, cuda.Grid.init1D(d_buffer.len, 0), .{d_buffer});
    var h_buffer = try cuda.allocAndCopyResult(u8, testing.allocator, d_buffer);
    var expected = "Hello World!";
    try testing.expectEqualSlices(u8, expected, h_buffer[0..expected.len]);
}
