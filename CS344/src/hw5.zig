const std = @import("std");
const math = std.math;
const testing = std.testing;
const Allocator = std.mem.Allocator;

const cuda = @import("cuda");
const cu = cuda.cu;

const hw5_kernel = @import("hw5_kernel.zig");
const utils = @import("utils.zig");

const hw5_ptx = @embedFile("hw5_ptx");

const log = std.log.scoped(.hw5);
pub fn main() !void {
    var general_purpose_allocator = std.heap.GeneralPurposeAllocator(.{}){};
    const allocator = &general_purpose_allocator.allocator();

    var stream = try cuda.Stream.init(0);
    defer stream.deinit();

    const module: *cuda.Module = .initFromData(hw5_ptx);
    defer module.deinit();
    try initModule(module);

    log.info("***** HW5 ******", .{});
    const num_bins: usize = 64;
    const num_elems: usize = 10_000 * num_bins;

    const data = try allocator.alloc(u32, num_elems);
    defer allocator.free(data);
    var prng = std.Random.DefaultPrng.init(387418298);
    const random = prng.random();

    // make the mean unpredictable, but close enough to the middle
    // so that timings are unaffected
    const mean: f32 = @floatFromInt(num_bins / 2);
    const std_dev: f32 = 5;
    // TODO: generate this on the GPU
    for (data) |*x| {
        const r: f32 = random.floatNorm(f32) * std_dev + mean;
        const clipped_r = @min(@max(r, 0), @as(f32, @floatFromInt(num_bins - 1)));
        x.* = @intFromFloat(clipped_r);
    }

    const ref_histo = try allocator.alloc(u32, num_bins);
    defer allocator.free(ref_histo);
    cpu_histogram(data, ref_histo);

    const atomic_histo = try allocator.alloc(u32, num_bins);
    defer allocator.free(atomic_histo);
    if (true) {
        const elapsed = try histogram(k.atomicHistogram, data, atomic_histo, ref_histo, .init1D(data.len, 1024));
        log.info("atomicHistogram of {} array took {:.3}ms", .{ num_elems, elapsed });
        log.info("atomicHistogram bandwith: {:.3}MB/s", .{computeBandwith(elapsed, data) * 1e-6});
    }

    {
        const elapsed = try histogram(k.bychunkHistogram, data, atomic_histo, ref_histo, .init1D(data.len / 32, 1024));
        log.info("bychunkHistogram of {} array took {:.3}ms", .{ num_elems, elapsed });
        log.info("bychunkHistogram bandwith: {:.3}MB/s", .{computeBandwith(elapsed, data) * 1e-6});
    }

    if (false) {
        const elapsed = try fastHistogramBroken(data, atomic_histo, ref_histo);
        log.info("fastHistogram of {} array took {:.3}ms", .{ num_elems, elapsed });
        log.info("fastHistogram bandwith: {:.3}MB/s", .{computeBandwith(elapsed, data) * 1e-6});
    }
}

pub fn cpu_histogram(data: []const u32, histo: []u32) void {
    @memset(histo, 0);
    for (data) |x| {
        histo[x] += 1;
    }
}

pub fn histogram(kernel: anytype, data: []const u32, histo: []u32, ref_histo: []const u32, grid: cuda.Grid) !f64 {
    var stream = try cuda.Stream.init(0);
    const d_data = try stream.allocAndCopy(u32, data);
    const d_histo = try stream.alloc(u32, histo.len);
    stream.memset(u32, d_histo, 0);

    var timer = cuda.GpuTimer.start(stream);
    try kernel.launchWithSharedMem(
        stream,
        grid,
        d_histo.len * @sizeOf(u32),
        .{ d_data, d_histo },
    );
    timer.stop();
    stream.memcpyDtoH(u32, histo, d_histo);
    stream.synchronize();
    const elapsed = timer.elapsed();
    std.testing.expectEqualSlices(u32, ref_histo, histo) catch {
        if (ref_histo.len < 100) {
            log.err("Histogram mismatch. Expected: {any}, got {any}", .{ ref_histo, histo });
        }
        // return err;
    };
    return elapsed;
}

const Kernels = struct {
    atomicHistogram: cuda.Kernel(hw5_kernel, "atomicHistogram"),
    bychunkHistogram: cuda.Kernel(hw5_kernel, "bychunkHistogram"),
    coarseBins: cuda.Kernel(hw5_kernel, "coarseBins"),
    shuffleCoarseBins: cuda.Kernel(hw5_kernel, "shuffleCoarseBins32"),
    cdfIncremental: cuda.Kernel(hw5_kernel, "cdfIncremental"),
    cdfIncrementalShift: cuda.Kernel(hw5_kernel, "cdfIncrementalShift"),
};

var k: Kernels = undefined;

fn initModule(module: *cuda.Module) !void {
    k = Kernels{
        .atomicHistogram = try .init(module),
        .bychunkHistogram = try .init(module),
        .coarseBins = try .init(module),
        .shuffleCoarseBins = try .init(module),
        .cdfIncremental = try .init(module),
        .cdfIncrementalShift = try .init(module),
    };
}

fn computeBandwith(elapsed_ms: f64, data: []const u32) f64 {
    const n: f64 = @floatFromInt(data.len);
    const bytes: f64 = @floatFromInt(@sizeOf(u32));
    return n * bytes / elapsed_ms * 1000;
}

fn fastHistogram(data: []const u32, histo: []u32, ref_histo: []const u32) !f32 {
    var stream = &(try cuda.Stream.init(0));
    _ = try stream.allocAndCopy(u32, data);
    const d_histo = try stream.alloc(u32, histo.len);
    stream.memset(u32, d_histo, 0);
    const d_radix = try stream.alloc(u32, data.len * 32);
    stream.memset(u32, d_radix, 0);

    stream.memset(u32, d_radix, 0);
    _ = ref_histo;
    return 0.0;
}

fn fastHistogramBroken(data: []const u32, histo: []u32, ref_histo: []const u32) !f32 {
    const stream = try cuda.Stream.init(0);
    const d_values = try stream.allocAndCopy(u32, data);
    const d_histo = try stream.alloc(u32, histo.len);
    stream.memset(u32, d_histo, 0);
    const d_radix = try stream.alloc(u32, data.len * 32);
    stream.memset(u32, d_radix, 0);

    var timer = cuda.GpuTimer.start(stream);

    const n = d_values.len;
    stream.memset(u32, d_radix, 0);
    // We split the bins into 32 coarse bins.
    const d_histo_boundaries = try stream.alloc(u32, 32);

    try k.coarseBins.launch(
        stream,
        .init1D(n, 1024),
        .{ d_values, d_radix },
    );
    // const radix_sum = try cuda.algorithms.reduce(stream, k.sumU32, d_radix);
    // log.debug("Radix sums to {}, expected {}", .{ radix_sum, d_values.len });
    // std.debug.assert(radix_sum == d_values.len);
    try inPlaceCdf(stream, d_radix, 1024);
    // debugDevice("d_radix + cdf", d_radix);
    try k.shuffleCoarseBins.launch(
        stream,
        .init1D(n, 1024),
        .{ d_histo, d_histo_boundaries, d_radix, d_values },
    );
    var histo_boundaries: [33]u32 = undefined;
    stream.memcpyDtoH(u32, &histo_boundaries, d_histo_boundaries);
    timer.stop();
    stream.synchronize();
    // Now we can partition d_values into coarse bins.
    var bin: u32 = 0;
    while (bin < 32) : (bin += 1) {
        const bin_start = histo_boundaries[bin];
        const bin_end = histo_boundaries[bin + 1];
        const d_bin_values = d_histo[bin_start .. bin_end + 1];
        // TODO histogram(d_bin_values)
        _ = d_bin_values;
    }

    const elapsed = timer.elapsed();
    std.testing.expectEqualSlices(u32, ref_histo, histo) catch {
        if (ref_histo.len < 100) {
            log.err("Histogram mismatch. Expected: {any}, got {any}", .{ ref_histo, histo });
        }
        // return err;
    };
    return elapsed;
}

// TODO: the cdf kernels should be part of cudaz
pub fn inPlaceCdf(stream: cuda.Stream, d_values: []u32, n_threads: u32) cuda.Error!void {
    const n = d_values.len;
    const grid_N: cuda.Grid = .init1D(n, n_threads);
    const n_blocks = grid_N.blocks.x;
    const d_grid_bins = try stream.alloc(u32, n_blocks);
    defer stream.free(d_grid_bins);
    var n_threads_pow_2 = n_threads;
    while (n_threads_pow_2 > 1) {
        std.debug.assert(n_threads_pow_2 % 2 == 0);
        n_threads_pow_2 /= 2;
    }
    log.warn("cdf(n={}, n_threads={}, n_blocks={})", .{ n, n_threads, n_blocks });
    try k.cdfIncremental.launchWithSharedMem(
        stream,
        grid_N,
        n_threads * @sizeOf(u32),
        .{ d_values, d_grid_bins },
    );
    if (n_blocks == 1) return;

    try inPlaceCdf(stream, d_grid_bins, n_threads);
    try k.cdfIncrementalShift.launch(
        stream,
        grid_N,
        .{ d_values, d_grid_bins },
    );
}

test "inPlaceCdf" {
    try initModule(0);
    var stream = try cuda.Stream.init(0);
    defer stream.deinit();
    const h_x = [_]u32{ 0, 2, 1, 1, 0, 1, 3, 0, 2 };
    var h_out = [_]u32{0} ** h_x.len;
    const h_cdf = [_]u32{ 0, 0, 2, 3, 4, 4, 5, 8, 8 };
    const d_x = try stream.alloc(u32, h_x.len);
    defer stream.free(d_x);

    try stream.memcpyHtoD(u32, d_x, &h_x);
    try inPlaceCdf(stream, d_x, 16);
    try utils.expectEqualDeviceSlices(u32, &h_cdf, d_x);

    try stream.memcpyHtoD(u32, d_x, &h_x);
    try inPlaceCdf(stream, d_x, 8);
    try stream.memcpyDtoH(u32, &h_out, d_x);
    try testing.expectEqual(h_cdf, h_out);

    // Try with smaller batch sizes, forcing several passes
    try stream.memcpyHtoD(u32, d_x, &h_x);
    try inPlaceCdf(stream, d_x, 4);
    try stream.memcpyDtoH(u32, &h_out, d_x);
    try testing.expectEqual(h_cdf, h_out);

    try stream.memcpyHtoD(u32, d_x, &h_x);
    try inPlaceCdf(stream, d_x, 2);
    try utils.expectEqualDeviceSlices(u32, &h_cdf, d_x);
}
