//! Compare implementations of different transpose algorithms
//! Compile with Zig, then run: sudo /usr/local/cuda/bin/ncu ./CS344/zig-out/bin/lesson5 > ./CS344/resources/lesson5/ncu_report.txt
//! This will generate
const std = @import("std");
const Random = std.rand.DefaultPrng;
const log = std.log.scoped(.lesson5);

const cuda = @import("cudaz");
const cu = cuda.cu;
const RawKernels = @import("lesson5_kernel.zig");

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
    RawKernels.transposeCpu(data, trans_cpu, num_cols);
    const elapsed_cpu = @intToFloat(f32, timer.lap()) / std.time.ns_per_us;

    log.info("CPU transpose of {}x{} matrix took {:.3}ms", .{ num_cols, num_cols, elapsed_cpu });
    var stream = initStreamAndModule(0);
    const d_data = try cuda.allocAndCopy(u32, data);
    const d_trans = try cuda.alloc(u32, data.len);
    var elapsed_gpu = try transposeSerial(&stream, d_data, d_trans, num_cols);
    log.info("GPU serial transpose of {}x{} matrix took {:.3}ms", .{ num_cols, num_cols, elapsed_gpu });

    elapsed_gpu = try transposePerRow(&stream, d_data, d_trans, num_cols);
    log.info("GPU transpose per row of {}x{} matrix took {:.3}ms", .{ num_cols, num_cols, elapsed_gpu });

    elapsed_gpu = try transposePerCell(&stream, d_data, d_trans, num_cols);
    log.info("GPU transpose per cell of {}x{} matrix took {:.3}ms", .{ num_cols, num_cols, elapsed_gpu });
}

fn transposeSerial(stream: *cuda.Stream, d_data: []const u32, d_out: []u32, num_cols: usize) !f64 {
    var timer = cuda.GpuTimer.start(stream);
    try k.transposeCpu.launch(
        stream,
        cuda.Grid.init1D(1, 1),
        .{ d_data, d_out, num_cols },
    );
    timer.stop();
    return timer.elapsed();
}

fn transposePerRow(stream: *cuda.Stream, d_data: []const u32, d_out: []u32, num_cols: usize) !f64 {
    var timer = cuda.GpuTimer.start(stream);
    try k.transposePerRow.launch(
        stream,
        cuda.Grid.init1D(num_cols, 32),
        .{ d_data, d_out, num_cols },
    );
    timer.stop();
    return timer.elapsed();
}

fn transposePerCell(stream: *cuda.Stream, d_data: []const u32, d_out: []u32, num_cols: usize) !f64 {
    var timer = cuda.GpuTimer.start(stream);
    try k.transposePerCell.launch(
        stream,
        cuda.Grid.init2D(num_cols, num_cols, 32, 32),
        .{ d_data, d_out, num_cols },
    );
    timer.stop();
    return timer.elapsed();
}

const Kernels = struct {
    transposeCpu: cuda.FnStruct("transposeCpu", RawKernels.transposeCpu),
    transposePerRow: cuda.FnStruct("transposePerRow", RawKernels.transposePerRow),
    transposePerCell: cuda.FnStruct("transposePerCell", RawKernels.transposePerCell),
};
var k: Kernels = undefined;

fn initStreamAndModule(device: u8) cuda.Stream {
    const stream = cuda.Stream.init(device) catch unreachable;
    // Panic if we can't load the module.
    k = Kernels{
        .transposeCpu = @TypeOf(k.transposeCpu).init() catch unreachable,
        .transposePerRow = @TypeOf(k.transposePerRow).init() catch unreachable,
        .transposePerCell = @TypeOf(k.transposePerCell).init() catch unreachable,
    };
    return stream;
}

fn peekBandwith(device: u8) f32 {
    var d: cu.CUdevice = null;
    cu.check(cu.cuDeviceGetProperties(&d, device)) catch unreachable;
    var props: cu.CUdevprop = undefined;
    cu.check(cu.cuDeviceGetProperties(&props, d)) catch unreachable;
    const mem_clock_rate_hz = props.memoryClockRate * 1e3;
    const mem_bus_width_bytes = @divExact(props.memoryBusWidth, 8);
    return mem_clock_rate_hz * mem_bus_width_bytes;
}
