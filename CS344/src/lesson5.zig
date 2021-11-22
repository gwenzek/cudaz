const std = @import("std");
const Random = std.rand.DefaultPrng;
const log = std.log.scoped(.lesson5);

const cuda = @import("cudaz");
const kernels = @import("lesson5_kernel.zig");

pub fn main() !void {
    var general_purpose_allocator = std.heap.GeneralPurposeAllocator(.{}){};
    const allocator = &general_purpose_allocator.allocator;
    var random = Random.init(10298374);

    const num_cols: usize = 1024;
    var data = try allocator.alloc(u32, num_cols * num_cols);
    defer allocator.free(data);
    random.fill(std.mem.sliceAsBytes(data));
    var trans_cpu = try allocator.alloc(u32, num_cols * num_cols);
    defer allocator.free(trans_cpu);

    var timer = std.time.Timer.start() catch unreachable;
    kernels.transposeCpu(data, trans_cpu, num_cols);
    const elapsed_cpu = @intToFloat(f32, timer.lap()) / std.time.ns_per_us;

    log.info("CPU transpose of {}x{} matrix took {:.3}ms", .{ num_cols, num_cols, elapsed_cpu });
    var stream = try cuda.Stream.init(0);

    const d_data = try cuda.allocAndCopy(u32, data);
    const d_trans = try cuda.alloc(u32, data.len);
    const elapsed_gpu = try gpuTransposeSerial(&stream, d_data, d_trans, num_cols);

    log.info("GPU serial transpose of {}x{} matrix took {:.3}ms", .{ num_cols, num_cols, elapsed_gpu });
}

fn gpuTransposeSerial(stream: *cuda.Stream, d_data: []const u32, d_out: []u32, num_cols: usize) !f64 {
    const transposeCpu = try cuda.FnStruct("transposeCpu", kernels.transposeCpu).init();
    var timer = cuda.GpuTimer.start(stream);
    try transposeCpu.launch(
        stream,
        cuda.Grid.init1D(1, 1),
        .{ d_data, d_out, num_cols },
    );
    timer.stop();
    return timer.elapsed();
}

fn gpu_transpose_one_row(stream: *cuda.Stream, d_data: []const u32, d_out: []u32, num_cols: usize) !f64 {
    const transpose = try cuda.FnStruct("transpose_one_row", kernels.transpose_one_row).init();
    var timer = cuda.GpuTimer.start(stream);
    try transpose.launch(
        stream,
        cuda.Grid.init1D(1, 1),
        .{ d_data, d_out, num_cols },
    );
    timer.stop();
    return timer.elapsed();
}
