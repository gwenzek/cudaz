const std = @import("std");
const log = std.log;

const cudaz = @import("cudaz");
const Cuda = cudaz.Cuda;
const cu = cudaz.cu;

const ShmemReduce = cudaz.KernelSignature("shmem_reduce_kernel");
const GlobalReduce = cudaz.KernelSignature("global_reduce_kernel");

fn reduce(cuda: *Cuda, reduce_kernel: GlobalReduce, d_out: []f32, d_intermediate: []f32, d_in: []f32, uses_shared_memory: bool) !f32 {
    // assumes that size is not greater than maxThreadsPerBlock^2
    // and that size is a multiple of maxThreadsPerBlock
    const maxThreadsPerBlock: usize = 1024;
    var size = d_in.len;
    if (size % maxThreadsPerBlock != 0 or size > maxThreadsPerBlock * maxThreadsPerBlock) {
        log.err("Can't run reduce operator on an array of size {} with maxThreadsPerBlock={} ({})", .{ size, maxThreadsPerBlock, maxThreadsPerBlock * maxThreadsPerBlock });
        return cudaz.CudaError.InvalidValue;
    }
    const full_grid = cudaz.Grid{
        .blockDim = .{ .x = @intCast(c_uint, @divExact(size, maxThreadsPerBlock)) },
        .threadDim = .{ .x = maxThreadsPerBlock },
    };

    var shared_mem: usize = 0;
    if (uses_shared_memory) shared_mem = full_grid.threadDim.x * @sizeOf(f32);

    try reduce_kernel.launchWithSharedMem(
        full_grid,
        shared_mem,
        .{ .@"0" = d_intermediate.ptr, .@"1" = d_in.ptr },
    );

    // try cuda.synchronize();

    // now we're down to one block left, so reduce it
    const one_block = cudaz.Grid{ .blockDim = .{}, .threadDim = full_grid.blockDim };
    shared_mem = if (uses_shared_memory) one_block.threadDim.x * @sizeOf(f32) else 0;
    try reduce_kernel.launchWithSharedMem(
        one_block,
        shared_mem,
        .{ .@"0" = d_out.ptr, .@"1" = d_intermediate.ptr },
    );
    // try cuda.synchronize();
    var result: [1]f32 = undefined;
    try cuda.memcpyDtoH(f32, &result, d_out);
    return result[0];
}

fn benchmark_reduce(
    cuda: *Cuda,
    iter: u32,
    d_out: []f32,
    d_intermediate: []f32,
    d_in: []f32,
    uses_shared_memory: bool,
) !f32 {
    if (uses_shared_memory) {
        log.info("Running shared memory reduce {} times.", .{iter});
    } else {
        log.info("Running global reduce {} times.", .{iter});
    }
    log.info("using cuda: {}", .{cuda});
    var reduce_kernel: GlobalReduce = undefined;
    if (uses_shared_memory) {
        const shared_r = try ShmemReduce.init(cuda);
        reduce_kernel = GlobalReduce{ .f = shared_r.f, .cuda = cuda };
    } else {
        reduce_kernel = try GlobalReduce.init(cuda);
    }

    // Set the buffers to 0 to have correct computation on the first try
    try cuda.memset(f32, d_intermediate, 0.0);
    try cuda.memset(f32, d_out, 0.0);

    var timer = cudaz.GpuTimer.init(cuda);
    defer timer.deinit();
    timer.start();
    var result: f32 = undefined;
    var i: usize = 0;
    while (i < iter) : (i += 1) {
        var r_i = try reduce(
            cuda,
            reduce_kernel,
            d_out,
            d_intermediate,
            d_in,
            uses_shared_memory,
        );
        if (i == 0) {
            result = r_i;
        }
    }
    timer.stop();
    var avg_time = timer.elapsed() / @intToFloat(f32, iter) * 1000.0;
    log.info("average time elapsed: {d:.1}ms", .{avg_time});

    log.info("found sum = {}", .{result});
    return result;
}

pub fn main() !void {
    var general_purpose_allocator = std.heap.GeneralPurposeAllocator(.{}){};
    const gpa = &general_purpose_allocator.allocator;
    var cuda = try Cuda.init(0);

    try main_reduce(&cuda, gpa);
    try main_histo(&cuda, gpa);
}

fn main_reduce(cuda: *Cuda, gpa: *std.mem.Allocator) !void {
    const array_size: i32 = 1 << 20;
    // generate the input array on the host
    var h_in = try gpa.alloc(f32, array_size);
    defer gpa.free(h_in);
    log.debug("h_in = {*}", .{h_in.ptr});
    var prng = std.rand.DefaultPrng.init(0);
    var sum: f32 = 0.0;
    for (h_in) |*value| {
        // generate random float in [-1.0f, 1.0f]
        var v = -1.0 + prng.random.float(f32) * 2.0;
        value.* = v;
        sum += v;
    }
    log.info("original sum = {}", .{sum});

    // allocate GPU memory
    var d_in = try cuda.allocAndCopy(f32, h_in);
    defer cuda.free(d_in);
    var d_intermediate = try cuda.alloc(f32, array_size); // overallocated
    defer cuda.free(d_intermediate);
    var d_out = try cuda.alloc(f32, 1);
    defer cuda.free(d_out);

    // launch the kernel
    // Run shared_reduced first because it doesn't modify d_in
    _ = try benchmark_reduce(cuda, 100, d_out, d_intermediate, d_in, true);
    _ = try benchmark_reduce(cuda, 100, d_out, d_intermediate, d_in, false);
}

fn log2(i: i32) u5 {
    var r: u5 = 0;
    var j = i;
    while (j > 0) : (j >>= 1) {
        r += 1;
    }
    return r;
}

fn bit_reverse(w: i32, bits: u5) i32 {
    var r: i32 = 0;
    var i: u5 = 0;
    while (i < bits) : (i += 1) {
        const bit = (w & (@intCast(i32, 1) << i)) >> i;
        r |= bit << (bits - i - 1);
    }
    return r;
}

fn main_histo(cuda: *Cuda, gpa: *std.mem.Allocator) !void {
    const array_size = 65536;
    const bin_count = 16;

    // generate the input array on the host
    var cpu_bins = [_]i32{0} ** bin_count;
    var h_in = try gpa.alloc(i32, array_size);
    for (h_in) |*value, i| {
        var item = bit_reverse(@intCast(i32, i), log2(array_size));
        value.* = item;
        cpu_bins[@intCast(usize, @mod(item, bin_count))] += 1;
    }
    log.info("Cpu bins: {any}", .{cpu_bins});
    var naive_bins = [_]i32{0} ** bin_count;
    var simple_bins = [_]i32{0} ** bin_count;

    // allocate GPU memory
    var d_in = try cuda.allocAndCopy(i32, h_in);
    var d_bins = try cuda.allocAndCopy(i32, &naive_bins);

    log.info("Running naive histo", .{});
    const grid = cudaz.Grid{ .blockDim = .{ .x = @divExact(array_size, 64) }, .threadDim = .{ .x = 64 } };
    const naive_histo = try cudaz.KernelSignature("naive_histo").init(cuda);
    try naive_histo.launch(grid, .{ .@"0" = d_bins.ptr, .@"1" = d_in.ptr, .@"2" = bin_count });
    try cuda.memcpyDtoH(i32, &naive_bins, d_bins);
    log.info("naive bins: {any}", .{naive_bins});

    log.info("Running simple histo", .{});
    try cuda.memcpyHtoD(i32, d_bins, &simple_bins);
    const simple_histo = try cudaz.KernelSignature("simple_histo").init(cuda);
    try simple_histo.launch(grid, .{ .@"0" = d_bins.ptr, .@"1" = d_in.ptr, .@"2" = bin_count });
    try cuda.memcpyDtoH(i32, &simple_bins, d_bins);
    log.info("simple bins: {any}", .{simple_bins});
}
