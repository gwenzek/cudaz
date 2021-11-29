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

    const num_cols: usize = 2048;
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

    log.info("GPU peek bandwith: {}MB/s", .{peekBandwith(0) * 1e-6});
    const d_data = try cuda.allocAndCopy(u32, data);
    const d_trans = try cuda.alloc(u32, data.len);
    // var elapsed = try transposeSerial(&stream, d_data, d_trans, num_cols);
    // log.info("GPU serial transpose of {}x{} matrix took {:.3}ms", .{ num_cols, num_cols, elapsed });
    // log.info("GPU serial transpose bandwith: {}MB/s", .{computeBandwith(elapsed, data) * 1e-6});

    var elapsed = try transposePerRow(&stream, d_data, d_trans, num_cols);
    log.info("GPU transpose per row of {}x{} matrix took {:.3}ms", .{ num_cols, num_cols, elapsed });
    log.info("GPU transpose per row bandwith: {}MB/s", .{computeBandwith(elapsed, data) * 1e-6});

    elapsed = try transposePerCell(&stream, d_data, d_trans, num_cols);
    log.info("GPU transpose per cell of {}x{} matrix took {:.3}ms", .{ num_cols, num_cols, elapsed });
    log.info("GPU transpose per cell bandwith: {}MB/s", .{computeBandwith(elapsed, data) * 1e-6});

    elapsed = try transposePerBlock(&stream, d_data, d_trans, num_cols);
    log.info("GPU transpose per block of {}x{} matrix took {:.3}ms", .{ num_cols, num_cols, elapsed });
    log.info("GPU transpose per block bandwith: {}MB/s", .{computeBandwith(elapsed, data) * 1e-6});

    elapsed = try transposePerBlockInlined(&stream, d_data, d_trans, num_cols);
    log.info("GPU transpose per block inlined of {}x{} matrix took {:.3}ms", .{ num_cols, num_cols, elapsed });
    log.info("GPU transpose per block inlined bandwith: {}MB/s", .{computeBandwith(elapsed, data) * 1e-6});
}

fn transposeSerial(stream: *cuda.Stream, d_data: []const u32, d_out: []u32, num_cols: usize) !f64 {
    var timer = cuda.GpuTimer.start(stream);
    try k.transposeCpu.launch(
        stream,
        cuda.Grid.init1D(1, 1),
        .{ d_data, d_out, num_cols },
    );
    timer.stop();
    try expectTransposed(d_data, d_out, num_cols);
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
    try expectTransposed(d_data, d_out, num_cols);
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
    try expectTransposed(d_data, d_out, num_cols);
    return timer.elapsed();
}

fn transposePerBlock(stream: *cuda.Stream, d_data: []const u32, d_out: []u32, num_cols: usize) !f64 {
    var timer = cuda.GpuTimer.start(stream);
    const block_size = RawKernels.block_size;
    const grid = cuda.Grid.init2D(num_cols, num_cols, 32, block_size);
    try k.transposePerBlock.launchWithSharedMem(
        stream,
        grid,
        @sizeOf(u32) * 16 * 16 * 16,
        .{ d_data, d_out, num_cols },
    );
    timer.stop();
    try expectTransposed(d_data, d_out, num_cols);
    return timer.elapsed();
}

fn transposePerBlockInlined(stream: *cuda.Stream, d_data: []const u32, d_out: []u32, num_cols: usize) !f64 {
    var timer = cuda.GpuTimer.start(stream);
    const block_size = RawKernels.block_size_inline;
    const grid = cuda.Grid.init2D(num_cols, num_cols, 256 / block_size, block_size);
    try k.transposePerBlockInlined.launch(
        stream,
        grid,
        .{ d_data, d_out, num_cols },
    );
    timer.stop();
    try expectTransposed(d_data, d_out, num_cols);
    return timer.elapsed();
}

const Kernels = struct {
    transposeCpu: cuda.FnStruct("transposeCpu", RawKernels.transposeCpu),
    transposePerRow: cuda.FnStruct("transposePerRow", RawKernels.transposePerRow),
    transposePerCell: cuda.FnStruct("transposePerCell", RawKernels.transposePerCell),
    transposePerBlock: cuda.FnStruct("transposePerBlock", RawKernels.transposePerBlock),
    transposePerBlockInlined: cuda.FnStruct("transposePerBlockInlined", RawKernels.transposePerBlockInlined),
};
var k: Kernels = undefined;

fn initStreamAndModule(device: u8) cuda.Stream {
    const stream = cuda.Stream.init(device) catch unreachable;
    // Panic if we can't load the module.
    k = Kernels{
        .transposeCpu = @TypeOf(k.transposeCpu).init() catch unreachable,
        .transposePerRow = @TypeOf(k.transposePerRow).init() catch unreachable,
        .transposePerCell = @TypeOf(k.transposePerCell).init() catch unreachable,
        .transposePerBlock = @TypeOf(k.transposePerBlock).init() catch unreachable,
        .transposePerBlockInlined = @TypeOf(k.transposePerBlockInlined).init() catch unreachable,
    };
    return stream;
}

fn peekBandwith(device: u8) f64 {
    var d: cu.CUdevice = undefined;
    cuda.check(cu.cuDeviceGet(&d, device)) catch unreachable;

    var mem_clock_rate_khz: i32 = undefined;
    _ = cu.cuDeviceGetAttribute(&mem_clock_rate_khz, cu.CU_DEVICE_ATTRIBUTE_MEMORY_CLOCK_RATE, d);
    var mem_bus_width_bits: i32 = undefined;
    _ = cu.cuDeviceGetAttribute(&mem_bus_width_bits, cu.CU_DEVICE_ATTRIBUTE_GLOBAL_MEMORY_BUS_WIDTH, d);
    const mem_bus_width_bytes = @intToFloat(f64, @divExact(mem_bus_width_bits, 8));
    return @intToFloat(f64, mem_clock_rate_khz) * 1e3 * mem_bus_width_bytes;
}

fn computeBandwith(elapsed_ms: f64, data: []const u32) f64 {
    const n = @intToFloat(f64, data.len);
    const bytes = @intToFloat(f64, @sizeOf(u32));
    return n * bytes / elapsed_ms * 1000;
}

fn expectTransposed(d_data: []const u32, d_trans: []u32, num_cols: usize) !void {
    var allocator = std.testing.allocator;
    const data = try cuda.allocAndCopyResult(u32, allocator, d_data);
    defer allocator.free(data);
    const trans = try cuda.allocAndCopyResult(u32, allocator, d_trans);
    defer allocator.free(trans);
    var i: usize = 0;
    while (i < num_cols) : (i += 1) {
        var j: usize = 0;
        while (j < num_cols) : (j += 1) {
            std.testing.expectEqual(data[num_cols * j + i], trans[num_cols * i + j]) catch {
                if (data.len < 100) {
                    log.err("original: {any}", .{data});
                    log.err("transposed: {any}", .{trans});
                }
                log.err("failed expectTransposed !", .{});
                return;
            };
        }
    }
    try cuda.memset(u32, d_trans, 0);
}
