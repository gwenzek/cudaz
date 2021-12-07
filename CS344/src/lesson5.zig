//! Compare implementations of different transpose algorithms
//! Compile with Zig, then run: sudo /usr/local/cuda/bin/ncu ./CS344/zig-out/bin/lesson5 > ./CS344/resources/lesson5/ncu_report.txt
//! This will generate
const std = @import("std");
const Allocator = std.mem.Allocator;
const Random = std.rand.DefaultPrng;
const base_log = std.log.scoped(.lesson5);

const cuda = @import("cudaz");
const cu = cuda.cu;
const RawKernels = @import("lesson5_kernel.zig");

pub fn main() !void {
    try nosuspend amain();
}

pub fn amain() !void {
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

    base_log.info("CPU transpose of {}x{} matrix took {:.3}ms", .{ num_cols, num_cols, elapsed_cpu });
    var stream = initStreamAndModule(0);
    defer stream.deinit();

    gpuInfo(0);
    // var elapsed = try transposeSerial(&stream, d_data, d_trans, num_cols);
    // log.info("GPU serial transpose of {}x{} matrix took {:.3}ms", .{ num_cols, num_cols, elapsed });
    // log.info("GPU serial transpose bandwith: {:.3}MB/s", .{computeBandwith(elapsed, data) * 1e-6});

    // This is not the best way of doing profiling cause we are
    // scheduling all kernels at the same time on the GPU
    // and they will interfere with each others.
    // But it shows how multi streams work.
    var per_row = async transposePerRow(allocator, data, num_cols);
    var per_cell = async transposePerCell(allocator, data, num_cols);
    var per_block = async transposePerBlock(allocator, data, num_cols);
    var per_block_inline = async transposePerBlockInlined(allocator, data, num_cols);

    resume per_row;
    try await per_row;
    resume per_cell;
    try await per_cell;
    resume per_block;
    try await per_block;
    resume per_block_inline;
    try await per_block_inline;
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

fn transposePerRow(allocator: *Allocator, data: []const u32, num_cols: usize) !void {
    var stream = cuda.Stream.init(0) catch unreachable;
    defer stream.deinit();
    const d_data = try stream.alloc(u32, data.len);
    try stream.memcpyHtoD(u32, d_data, data);
    var out = try allocator.alloc(u32, data.len);
    defer allocator.free(out);
    const d_out = try stream.alloc(u32, data.len);
    defer cuda.free(d_out);
    const log = std.log.scoped(.transposePerRow);
    log.info("Scheduling GPU", .{});
    var timer = cuda.GpuTimer.start(&stream);
    k.transposePerRow.launch(
        &stream,
        cuda.Grid.init1D(num_cols, 32),
        .{ d_data, d_out, num_cols },
    ) catch unreachable;
    timer.stop();
    try stream.memcpyDtoH(u32, out, d_out);
    // Yield control to main loop
    suspend {}
    stream.synchronize();
    const elapsed = timer.elapsed();
    log.info("{}x{} matrix took {:.3}ms", .{ num_cols, num_cols, elapsed });
    log.info("bandwith: {:.3}MB/s", .{computeBandwith(elapsed, data) * 1e-6});
    expectTransposed(data, out, num_cols) catch {
        log.err("didn't properly transpose !", .{});
    };
}

fn transposePerCell(allocator: *Allocator, data: []const u32, num_cols: usize) !void {
    var stream = cuda.Stream.init(0) catch unreachable;
    defer stream.deinit();
    const d_data = try stream.alloc(u32, data.len);
    try stream.memcpyHtoD(u32, d_data, data);
    var out = try allocator.alloc(u32, data.len);
    defer allocator.free(out);
    const d_out = try stream.alloc(u32, data.len);
    defer cuda.free(d_out);
    const log = std.log.scoped(.transposePerCell);
    log.info("Scheduling GPU", .{});
    var timer = cuda.GpuTimer.start(&stream);
    k.transposePerCell.launch(
        &stream,
        cuda.Grid.init2D(num_cols, num_cols, 32, 32),
        .{ d_data, d_out, num_cols },
    ) catch unreachable;
    timer.stop();
    try stream.memcpyDtoH(u32, out, d_out);
    // Yield control to main loop
    suspend {}
    stream.synchronize();
    const elapsed = timer.elapsed();
    log.info("{}x{} matrix took {:.3}ms", .{ num_cols, num_cols, elapsed });
    log.info("bandwith: {:.3}MB/s", .{computeBandwith(elapsed, data) * 1e-6});
    expectTransposed(data, out, num_cols) catch {
        log.err("didn't properly transpose !", .{});
    };
}

fn transposePerBlock(allocator: *Allocator, data: []const u32, num_cols: usize) !void {
    var out = try allocator.alloc(u32, data.len);
    defer allocator.free(out);
    var stream = cuda.Stream.init(0) catch unreachable;
    defer stream.deinit();
    const d_data = try stream.alloc(u32, data.len);
    try stream.memcpyHtoD(u32, d_data, data);
    const d_out = try stream.alloc(u32, data.len);
    defer cuda.free(d_out);
    const log = std.log.scoped(.transposePerBlock);
    log.info("Scheduling GPU", .{});
    var timer = cuda.GpuTimer.start(&stream);
    const block_size = RawKernels.block_size;
    const grid = cuda.Grid.init2D(num_cols, num_cols, 32, block_size);
    try k.transposePerBlock.launchWithSharedMem(
        &stream,
        grid,
        @sizeOf(u32) * 16 * 16 * 16,
        .{ d_data, d_out, num_cols },
    );
    timer.stop();
    try stream.memcpyDtoH(u32, out, d_out);
    // Yield control to main loop
    suspend {}
    stream.synchronize();
    const elapsed = timer.elapsed();
    log.info("{}x{} matrix took {:.3}ms", .{ num_cols, num_cols, elapsed });
    log.info("bandwith: {:.3}MB/s", .{computeBandwith(elapsed, data) * 1e-6});
    expectTransposed(data, out, num_cols) catch {
        log.err("didn't properly transpose !", .{});
    };
}

fn transposePerBlockInlined(allocator: *Allocator, data: []const u32, num_cols: usize) !void {
    var out = try allocator.alloc(u32, data.len);
    defer allocator.free(out);
    var stream = cuda.Stream.init(0) catch unreachable;
    defer stream.deinit();
    const d_data = try stream.alloc(u32, data.len);
    try stream.memcpyHtoD(u32, d_data, data);
    const d_out = try stream.alloc(u32, data.len);
    defer cuda.free(d_out);
    const log = std.log.scoped(.transposePerBlockInlined);
    log.info("Scheduling GPU", .{});
    var timer = cuda.GpuTimer.start(&stream);
    const block_size = RawKernels.block_size_inline;
    const grid = cuda.Grid.init2D(num_cols, num_cols, 256 / block_size, block_size);
    try k.transposePerBlockInlined.launch(
        &stream,
        grid,
        .{ d_data, d_out, num_cols },
    );
    timer.stop();
    try stream.memcpyDtoH(u32, out, d_out);
    // Yield control to main loop
    suspend {}
    stream.synchronize();
    const elapsed = timer.elapsed();
    log.info("{}x{} matrix took {:.3}ms", .{ num_cols, num_cols, elapsed });
    log.info("bandwith: {:.3}MB/s", .{computeBandwith(elapsed, data) * 1e-6});
    expectTransposed(data, out, num_cols) catch {
        log.err("didn't properly transpose !", .{});
    };
}

const Kernels = struct {
    transposeCpu: cuda.FnStruct("transposeCpu", RawKernels.transposeCpu),
    transposePerRow: cuda.FnStruct("transposePerRow", RawKernels.transposePerRow),
    transposePerCell: cuda.FnStruct("transposePerCell", RawKernels.transposePerCell),
    transposePerBlock: cuda.FnStruct("transposePerBlock", RawKernels.transposePerBlock),
    transposePerBlockInlined: cuda.FnStruct("transposePerBlockInlined", RawKernels.transposePerBlockInlined),
};
var k: Kernels = undefined;

fn initStreamAndModule(device: u3) cuda.Stream {
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

fn gpuInfo(device: u8) void {
    var d: cu.CUdevice = undefined;
    cuda.check(cu.cuDeviceGet(&d, device)) catch unreachable;

    var mem_clock_rate_khz: i32 = undefined;
    _ = cu.cuDeviceGetAttribute(&mem_clock_rate_khz, cu.CU_DEVICE_ATTRIBUTE_MEMORY_CLOCK_RATE, d);
    var mem_bus_width_bits: i32 = undefined;
    _ = cu.cuDeviceGetAttribute(&mem_bus_width_bits, cu.CU_DEVICE_ATTRIBUTE_GLOBAL_MEMORY_BUS_WIDTH, d);
    const mem_bus_width_bytes = @intToFloat(f64, @divExact(mem_bus_width_bits, 8));
    const peek_bandwith = @intToFloat(f64, mem_clock_rate_khz) * 1e3 * mem_bus_width_bytes;
    base_log.info("GPU peek bandwith: {:.3}MB/s", .{peek_bandwith * 1e-6});

    var l1_cache: i32 = undefined;
    _ = cu.cuDeviceGetAttribute(&l1_cache, cu.CU_DEVICE_ATTRIBUTE_GLOBAL_L1_CACHE_SUPPORTED, d);
    var l2_cache: i32 = undefined;
    _ = cu.cuDeviceGetAttribute(&l2_cache, cu.CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE, d);

    base_log.info("GPU L1 cache {}, L2 cache {}", .{ l1_cache, l2_cache });
}

fn computeBandwith(elapsed_ms: f64, data: []const u32) f64 {
    const n = @intToFloat(f64, data.len);
    const bytes = @intToFloat(f64, @sizeOf(u32));
    return n * bytes / elapsed_ms * 1000;
}

fn expectTransposed(data: []const u32, trans: []u32, num_cols: usize) !void {
    var i: usize = 0;
    while (i < num_cols) : (i += 1) {
        var j: usize = 0;
        while (j < num_cols) : (j += 1) {
            std.testing.expectEqual(data[num_cols * j + i], trans[num_cols * i + j]) catch |err| {
                if (data.len < 100) {
                    base_log.err("original: {any}", .{data});
                    base_log.err("transposed: {any}", .{trans});
                }
                base_log.err("failed expectTransposed !", .{});
                return err;
            };
        }
    }
}
