const std = @import("std");
const builtin = @import("builtin");

const cuda = @import("cudaz");
const matmul = @import("matmul");

const matmul_ptx = @embedFile("matmul_ptx");

const log = std.log.scoped(.matmul);

pub const std_options: std.Options = .{
    .log_level = if (builtin.mode == .Debug) .debug else .info,
};

pub const matmulK = cuda.Kernel(matmul, "matmul");

pub fn main() !void {
    const allocator = std.heap.smp_allocator;

    var stream = try cuda.Stream.init(0);
    defer stream.deinit();
    log.warn("cuda: {f}", .{stream});
    const module: cuda.Module = .initFromData(matmul_ptx);
    defer module.deinit();

    const matmul_f: matmulK = try .init(module);
    const shape: matmul.Shape = .{ .m = 3, .n = 3, .k = 2 };

    const A_d = try cuda.alloc(f32, shape.m * shape.k);
    stream.memcpyHtoD(f32, A_d, &.{ 1.0, 2.0, -1.0, -2.0, 0.0, 0.0 });
    const B_d = try cuda.alloc(f32, shape.n * shape.k);
    stream.memcpyHtoD(f32, B_d, &.{ 1.0, 1.0, -1.0, -1.0, 0.0, 0.0 });

    const C_d = try cuda.alloc(f32, shape.m * shape.n);

    var timer = cuda.GpuTimer.start(&stream);
    try matmul_f.launch(
        &stream,
        .init2D(.{ shape.m, shape.n }, .{ 16, 16 }),
        .{ A_d, B_d, shape, C_d },
    );
    timer.stop();

    const C_h = try stream.allocAndCopyResult(f32, allocator, C_d);
    defer allocator.free(C_h);
    stream.synchronize();

    try std.testing.expectEqualSlices(
        f32,
        &.{ 3.0, -3.0, 0.0, -3.0, 3.0, 0.0, 0.0, 0.0, 0.0 },
        C_h,
    );

    std.debug.print("matmul {} took {e:.5}s\n", .{ shape, timer.elapsed() });
}
