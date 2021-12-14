const std = @import("std");
const log = std.log.scoped(.hw5);
const math = std.math;
const Allocator = std.mem.Allocator;
const Random = std.rand.Random;

const cuda = @import("cudaz");
const cu = cuda.cu;
const RawKernels = @import("hw5_kernel.zig");

pub fn main() !void {
    var general_purpose_allocator = std.heap.GeneralPurposeAllocator(.{}){};
    const allocator = &general_purpose_allocator.allocator();
    try initModule(0);

    log.info("***** HW5 ******", .{});
    const num_bins: usize = 1024;
    const num_elems: usize = 10_000 * num_bins;

    var data = try allocator.alloc(u32, num_elems);
    defer allocator.free(data);
    var random = std.rand.DefaultPrng.init(387418298).random();

    // make the mean unpredictable, but close enough to the middle
    // so that timings are unaffected
    const mean = random.intRangeLessThan(u32, num_bins / 2 - num_bins / 8, num_bins / 2 + num_bins / 8);
    const std_dev: f32 = 100;
    // TODO: generate this on the GPU
    for (data) |*x| {
        var r = @floatToInt(i32, random.floatNorm(f32) * std_dev) + @intCast(i32, mean);
        x.* = math.min(math.absCast(r), @intCast(u32, num_bins - 1));
    }

    var ref_histo = try allocator.alloc(u32, num_bins);
    defer allocator.free(ref_histo);
    cpu_histogram(data, ref_histo);

    var atomic_histo = try allocator.alloc(u32, num_bins);
    defer allocator.free(atomic_histo);
    var elapsed = try histogram(k.atomicHistogram.f, data, atomic_histo, ref_histo, cuda.Grid.init1D(data.len, 1024));
    log.info("atomicHistogram of {} array took {:.3}ms", .{ num_elems, elapsed });
    log.info("atomicHistogram bandwith: {:.3}MB/s", .{computeBandwith(elapsed, data) * 1e-6});

    elapsed = try histogram(k.bychunkHistogram.f, data, atomic_histo, ref_histo, cuda.Grid.init1D(data.len / 32, 1024));
    log.info("bychunkHistogram of {} array took {:.3}ms", .{ num_elems, elapsed });
    log.info("bychunkHistogram bandwith: {:.3}MB/s", .{computeBandwith(elapsed, data) * 1e-6});
}

pub fn cpu_histogram(data: []const u32, histo: []u32) void {
    std.mem.set(u32, histo, 0);
    for (data) |x| {
        histo[x] += 1;
    }
}

pub fn histogram(kernel: cu.CUfunction, data: []const u32, histo: []u32, ref_histo: []const u32, grid: cuda.Grid) !f64 {
    var stream = try cuda.Stream.init(0);
    var d_data = try stream.allocAndCopy(u32, data);
    var d_histo = try stream.alloc(u32, histo.len);
    stream.memset(u32, d_histo, 0);

    var timer = cuda.GpuTimer.start(&stream);
    try stream.launchWithSharedMem(
        kernel,
        grid,
        d_histo.len * @sizeOf(u32),
        .{ d_data, d_histo },
    );
    timer.stop();
    stream.memcpyDtoH(u32, histo, d_histo);
    stream.synchronize();
    var elapsed = timer.elapsed();
    std.testing.expectEqualSlices(u32, ref_histo, histo) catch {
        if (ref_histo.len < 100) {
            log.err("Histogram mismatch. Expected: {d}, got {d}", .{ ref_histo, histo });
        }
        // return err;
    };
    return elapsed;
}

const Kernels = struct {
    atomicHistogram: cuda.FnStruct("atomicHistogram", RawKernels.atomicHistogram),
    bychunkHistogram: cuda.FnStruct("bychunkHistogram", RawKernels.bychunkHistogram),
    coarseBins: cuda.FnStruct("coarseBins", RawKernels.coarseBins),
    shuffleCoarseBins: cuda.FnStruct("shuffleCoarseBins32", RawKernels.shuffleCoarseBins32),
};
var k: Kernels = undefined;

fn initModule(device: u3) !void {
    _ = try cuda.Stream.init(device);
    // Panic if we can't load the module.
    k = Kernels{
        .atomicHistogram = try @TypeOf(k.atomicHistogram).init(),
        .bychunkHistogram = try @TypeOf(k.bychunkHistogram).init(),
        .coarseBins = try @TypeOf(k.coarseBins).init(),
        .shuffleCoarseBins = try @TypeOf(k.shuffleCoarseBins).init(),
    };
}

fn computeBandwith(elapsed_ms: f64, data: []const u32) f64 {
    const n = @intToFloat(f64, data.len);
    const bytes = @intToFloat(f64, @sizeOf(u32));
    return n * bytes / elapsed_ms * 1000;
}

fn fastHistogram(data: []const u32, histo: []u32, ref_histo: []const u32) !void {
    var stream = try cuda.Stream.init(0);
    var d_values = try stream.allocAndCopy(u32, data);
    var d_histo = try stream.alloc(u32, histo.len);
    stream.memset(u32, d_histo, 0);
    var d_radix = try stream.alloc(u32, data.len * 32);
    stream.memset(u32, d_radix, 0);

    var timer = cuda.GpuTimer.start(&stream);

    const n = d_values.len;
    try cuda.memset(u32, d_radix, 0);
    // We split the bins into 32 coarse bins.
    const d_histo_boundaries = try stream.alloc(u32, 32);
    const shift = 5;
    const mask = 0b11111;
    try k.coarseBins.launch(
        stream,
        cuda.Grid.init1D(n, 1024),
        .{ d_radix.ptr, d_values.ptr, shift, mask, @intCast(c_int, n) },
    );
    const radix_sum = try cuda.algorithms.reduce(stream, k.sumU32, d_radix);
    log.debug("Radix sums to {}, expected {}", .{ radix_sum, d_values.len });
    std.debug.assert(radix_sum == d_values.len);
    try inPlaceCdf(stream, d_radix, 1024);
    // debugDevice("d_radix + cdf", d_radix);
    try k.shuffleCoarseBins.launch(
        stream,
        cuda.Grid.init1D(n, 1024),
        .{ d_histo.ptr, d_histo_boundaries.ptr, d_radix.ptr, d_values.ptr, shift, mask, @intCast(c_int, n) },
    );
    var histo_boundaries: [33]u32 = undefined;
    stream.memcpyDtoH(u32, &histo_boundaries, d_histo_boundaries);
    timer.stop();
    stream.synchronize();
    // Now we can partition d_values into coarse bins.
    var bin: u32 = 0;
    while (bin < 32) : (bin += 1) {
        var bin_start = histo_boundaries[bin];
        var bin_end = histo_boundaries[bin + 1];
        var d_bin_values = d_histo[bin_start .. bin_end + 1];
        // TODO histogram(d_bin_values)
        _ = d_bin_values;
    }

    var elapsed = timer.elapsed();
    std.testing.expectEqualSlices(u32, ref_histo, histo) catch {
        if (ref_histo.len < 100) {
            log.err("Histogram mismatch. Expected: {d}, got {d}", .{ ref_histo, histo });
        }
        // return err;
    };
    return elapsed;
}

// TODO: the cdf kernels should be part of cudaz
pub fn inPlaceCdf(stream: *const cuda.Stream, d_values: []u32, n_threads: u32) cuda.Error!void {
    const n = d_values.len;
    const grid_N = cuda.Grid.init1D(n, n_threads);
    const n_blocks = grid_N.blocks.x;
    var d_grid_bins = try cuda.alloc(u32, n_blocks);
    defer cuda.free(d_grid_bins);
    var n_threads_pow_2 = n_threads;
    while (n_threads_pow_2 > 1) {
        std.debug.assert(n_threads_pow_2 % 2 == 0);
        n_threads_pow_2 /= 2;
    }
    try k.cdfIncremental.launchWithSharedMem(
        stream,
        grid_N,
        n_threads * @sizeOf(u32),
        .{ d_values.ptr, d_grid_bins.ptr, @intCast(c_int, n) },
    );
    if (n_blocks == 1) return;

    try inPlaceCdf(stream, d_grid_bins, n_threads);
    try k.cdfShift.launch(
        stream,
        grid_N,
        .{ d_values.ptr, d_grid_bins.ptr, @intCast(c_int, n) },
    );
}
