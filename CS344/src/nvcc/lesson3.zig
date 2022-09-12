const std = @import("std");
const log = std.log;

const cuda = @import("cudaz");
const cu = cuda.cu;

const ShmemReduce = cuda.CudaKernel("shmem_reduce_kernel");
const GlobalReduce = cuda.CudaKernel("global_reduce_kernel");

pub fn main() !void {
    log.info("***** Lesson 3 ******", .{});
    var general_purpose_allocator = std.heap.GeneralPurposeAllocator(.{}){};
    const gpa = general_purpose_allocator.allocator();
    var stream = try cuda.Stream.init(0);

    try main_reduce(&stream, gpa);
    try main_histo(&stream, gpa);
}

fn main_reduce(stream: *const cuda.Stream, gpa: std.mem.Allocator) !void {
    const array_size: i32 = 1 << 20;
    // generate the input array on the host
    var h_in = try gpa.alloc(f32, array_size);
    defer gpa.free(h_in);
    log.debug("h_in = {*}", .{h_in.ptr});
    var prng = std.rand.DefaultPrng.init(0);
    const random = prng.random();
    var sum: f32 = 0.0;
    for (h_in) |*value| {
        // generate random float in [-1.0f, 1.0f]
        var v = -1.0 + random.float(f32) * 2.0;
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
    _ = try benchmark_reduce(stream, 100, d_out, d_intermediate, d_in, true);
    _ = try benchmark_reduce(stream, 100, d_out, d_intermediate, d_in, false);
}

fn benchmark_reduce(
    stream: *const cuda.Stream,
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
        reduce_kernel = GlobalReduce{ .f = (try ShmemReduce.init()).f };
    } else {
        reduce_kernel = try GlobalReduce.init();
    }

    // Set the buffers to 0 to have correct computation on the first try
    try cuda.memset(f32, d_intermediate, 0.0);
    try cuda.memset(f32, d_out, 0.0);

    var timer = cuda.GpuTimer.start(stream);
    errdefer timer.deinit();
    var result: f32 = undefined;
    var i: usize = 0;
    while (i < iter) : (i += 1) {
        var r_i = try reduce(
            stream,
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

fn reduce(stream: *const cuda.Stream, reduce_kernel: GlobalReduce, d_out: []f32, d_intermediate: []f32, d_in: []f32, uses_shared_memory: bool) !f32 {
    // assumes that size is not greater than blockSize^2
    // and that size is a multiple of blockSize
    const blockSize: usize = 1024;
    var size = d_in.len;
    if (size % blockSize != 0 or size > blockSize * blockSize) {
        log.err("Can't run reduce operator on an array of size {} with blockSize={} ({})", .{ size, blockSize, blockSize * blockSize });
        @panic("Invalid reduce on too big array");
    }
    const full_grid = cuda.Grid.init1D(size, blockSize);
    var shared_mem = if (uses_shared_memory) blockSize * @sizeOf(f32) else 0;
    try reduce_kernel.launchWithSharedMem(
        stream,
        full_grid,
        shared_mem,
        .{ d_intermediate.ptr, d_in.ptr },
    );

    // try cuda.synchronize();

    // now we're down to one block left, so reduce it
    const one_block = cuda.Grid.init1D(blockSize, 0);
    try reduce_kernel.launchWithSharedMem(
        stream,
        one_block,
        shared_mem,
        .{ d_out.ptr, d_intermediate.ptr },
    );
    // try cuda.synchronize();
    var result: [1]f32 = undefined;
    try cuda.memcpyDtoH(f32, &result, d_out);
    return result[0];
}

fn main_histo(stream: *const cuda.Stream, gpa: std.mem.Allocator) !void {
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
    const grid = cuda.Grid{ .blocks = .{ .x = @divExact(array_size, 64) }, .threads = .{ .x = 64 } };
    const naive_histo = try cuda.CudaKernel("naive_histo").init();
    try naive_histo.launch(stream, grid, .{ d_bins.ptr, d_in.ptr, bin_count });
    try cuda.memcpyDtoH(i32, &naive_bins, d_bins);
    log.info("naive bins: {any}", .{naive_bins});

    log.info("Running simple histo", .{});
    try cuda.memcpyHtoD(i32, d_bins, &simple_bins);
    const simple_histo = try cuda.CudaKernel("simple_histo").init();
    try simple_histo.launch(stream, grid, .{ d_bins.ptr, d_in.ptr, bin_count });
    try cuda.memcpyDtoH(i32, &simple_bins, d_bins);
    log.info("simple bins: {any}", .{simple_bins});
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
