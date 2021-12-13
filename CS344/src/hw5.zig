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
};
var k: Kernels = undefined;

fn initModule(device: u3) !void {
    _ = try cuda.Stream.init(device);
    // Panic if we can't load the module.
    k = Kernels{
        .atomicHistogram = try @TypeOf(k.atomicHistogram).init(),
        .bychunkHistogram = try @TypeOf(k.bychunkHistogram).init(),
    };
}

fn computeBandwith(elapsed_ms: f64, data: []const u32) f64 {
    const n = @intToFloat(f64, data.len);
    const bytes = @intToFloat(f64, @sizeOf(u32));
    return n * bytes / elapsed_ms * 1000;
}
