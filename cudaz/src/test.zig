const std = @import("std");
const testing = std.testing;

const cuda = @import("cudaz");
const kernel = @import("test_kernel.zig");

test "hello_world" {
    var stream = try cuda.Stream.init(0);
    defer stream.deinit();
    var d_buffer = try cuda.alloc(u8, 20);
    defer cuda.free(d_buffer);

    const gpu_hello = try cuda.ZigKernel(kernel, "testHelloWorld").init();
    try gpu_hello.launch(&stream, cuda.Grid.init1D(d_buffer.len, 0), .{d_buffer});
    var h_buffer = try stream.allocAndCopyResult(u8, testing.allocator, d_buffer);
    defer testing.allocator.free(h_buffer);

    var expected = "Hello World !";
    std.log.warn("{s}", .{h_buffer});
    try testing.expectEqualSlices(u8, expected, h_buffer[0..expected.len]);
}

test "log10" {
    var stream = try cuda.Stream.init(0);
    defer stream.deinit();

    const x = [_]f32{ 0.12345, 1.0, 10.0, std.math.inf(f32), 0 };

    var d_x = try stream.allocAndCopy(f32, &x);
    var d_out = try stream.alloc(f32, x.len);
    defer stream.free(d_out);

    const gpu_log10 = try cuda.ZigKernel(kernel, "testMathLog10").init();
    try gpu_log10.launch(&stream, cuda.Grid.init1D(x.len, 1), .{ d_x, d_out });
    var h_out = try stream.allocAndCopyResult(f32, testing.allocator, d_out);
    defer testing.allocator.free(h_out);

    for (x) |xx, i| {
        try testing.expectEqual(h_out[i], @log10(xx));
    }
}

test "swap2_with_shared_memory" {
    var stream = try cuda.Stream.init(0);
    defer stream.deinit();
    var d_src = try stream.allocAndCopy(u8, "Hello Boobaz !");
    defer stream.free(d_src);
    var d_tgt = try stream.alloc(u8, d_src.len);
    defer stream.free(d_tgt);

    const gpu_swap = try cuda.ZigKernel(kernel, "swap2").init();

    const grid = cuda.Grid.init1D(d_src.len, 2);
    try gpu_swap.launchWithSharedMem(&stream, grid, @sizeOf(@TypeOf(kernel._swap2_shared)), .{ d_src, d_tgt });
    var h_buf = try stream.allocAndCopyResult(u8, testing.allocator, d_tgt);
    defer testing.allocator.free(h_buf);
    stream.synchronize();

    var expected = "eHll ooBboza! ";
    // std.log.warn("{s}", .{h_buf});
    try testing.expectEqualSlices(u8, h_buf, expected);
}

test "reduce_with_shared_memory" {
    var stream = try cuda.Stream.init(0);
    defer stream.deinit();

    var h_src = [_]f32{ 1.0, 4.0, 2.0, 3.0 } ** 256;
    var d_src = try stream.allocAndCopy(f32, &h_src);
    defer stream.free(d_src);
    var d_tgt = try stream.alloc(f32, 1);
    defer stream.free(d_tgt);

    const reduce = try cuda.ZigKernel(kernel, "reduceSum").init();

    const grid = cuda.Grid.init1D(d_src.len, 1024);
    try reduce.launchWithSharedMem(&stream, grid, @sizeOf(@TypeOf(kernel._reduceSum_shared)), .{ d_src, &d_tgt[0] });
    var h_tgt = stream.copyResult(f32, &d_tgt[0]);
    stream.synchronize();

    try testing.expectEqual(@as(f32, 2560.0), h_tgt);
}
