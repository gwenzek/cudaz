const std = @import("std");
const builtin = @import("builtin");

const ptx = @import("nvptx");
pub const panic = ptx.panic;

pub fn atomicHistogram(d_data: []u32, d_bins: []u32) callconv(ptx.kernel) void {
    const gid = ptx.getIdX();
    if (gid >= d_data.len) return;

    const bin = d_data[gid];
    _ = @atomicRmw(u32, &d_bins[bin], .Add, 1, .seq_cst);
}

pub const bychunkHistogram_step: u32 = 32;

/// Fist accumulate into a shared histogram
/// then accumulate to the global histogram.
/// Should decreases contention when doing atomic adds.
pub fn bychunkHistogram(d_data: []u32, d_bins: []u32) callconv(ptx.kernel) void {
    const n = d_data.len;
    const num_bins = d_bins.len;
    const step = bychunkHistogram_step;
    const s_bins = ptx.sharedMemory(u32);
    const tid = ptx.threadIdX();
    if (tid < num_bins) s_bins[ptx.threadIdX()] = 0;
    ptx.syncThreads();

    var i: u32 = 0;
    while (i < step) : (i += 1) {
        const offset = ptx.ctaIdX() * ptx.numThreadsX() * step + i * ptx.numThreadsX() + tid;
        if (offset < n) {
            _ = @atomicRmw(u32, &s_bins[d_data[offset]], .Add, 1, .seq_cst);
        }
    }

    ptx.syncThreads();
    if (tid < num_bins) {
        _ = @atomicRmw(u32, &d_bins[tid], .Add, s_bins[tid], .seq_cst);
    }
}

pub fn coarseBins(d_data: []u32, d_coarse_bins: []u32) callconv(ptx.kernel) void {
    const n = d_data.len;
    const id = ptx.getIdX();
    if (id < n) {
        const rad = d_data[id] / 32;
        d_coarse_bins[rad * n + id] = 1;
    }
}

pub fn shuffleCoarseBins32(
    d_coarse_bins: []u32,
    d_coarse_bins_boundaries: []u32,
    d_cdf: []const u32,
    d_in: []const u32,
) callconv(ptx.kernel) void {
    const n = d_in.len;
    const id = ptx.getIdX();
    if (id >= n) return;
    const x = d_in[id];
    const rad = x >> 5 & 0b11111;
    const new_id = d_cdf[rad * n + id];
    d_coarse_bins[new_id] = x;

    if (id < 32) {
        d_coarse_bins_boundaries[id] = d_cdf[id * n];
    }
    if (id == 32) {
        d_coarse_bins_boundaries[id] = d_cdf[id * n - 1];
    }
}

pub fn cdfIncremental(d_glob_bins: []u32, d_block_bins: []u32) callconv(ptx.kernel) void {
    const n = d_glob_bins.len;
    const global_id = ptx.getIdX();
    if (global_id >= n) return;
    const tid = ptx.threadIdX();

    const d_bins = ptx.sharedMemory(u32);
    ptx.syncThreads();
    const last_tid = ptx.lastTid(@intCast(n));
    const total = ptx.exclusiveScan(.Add, d_bins[0..ptx.totalSharedMemory()], tid, last_tid);
    if (tid == last_tid) {
        d_block_bins[ptx.ctaIdX()] = total;
    }
    d_glob_bins[global_id] = d_bins[tid];
}

pub fn cdfIncrementalShift(d_glob_bins: []u32, d_block_bins: []const u32) callconv(ptx.kernel) void {
    const block_shift = d_block_bins[ptx.ctaIdX()];
    d_glob_bins[ptx.getIdX()] += block_shift;
}

comptime {
    if (ptx.is_nvptx) {
        @export(&atomicHistogram, .{ .name = "atomicHistogram" });
        @export(&bychunkHistogram, .{ .name = "bychunkHistogram" });
        @export(&coarseBins, .{ .name = "coarseBins" });
        @export(&shuffleCoarseBins32, .{ .name = "shuffleCoarseBins32" });
        @export(&cdfIncremental, .{ .name = "cdfIncremental" });
        @export(&cdfIncrementalShift, .{ .name = "cdfIncrementalShift" });
    }
}
