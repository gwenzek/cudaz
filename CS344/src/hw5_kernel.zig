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

// const step: u32 = 32;
const SharedMem = opaque {};
// extern var bychunkHistogram_shared: SharedMem align(8) addrspace(.shared); // stage2
var bychunkHistogram_shared: [1024]u32 = undefined; // stage1

const bychunkHistogram_step: u32 = 32;

/// Fist accumulate into a shared histogram
/// then accumulate to the global histogram.
/// Should decreases contention when doing atomic adds.
pub fn bychunkHistogram(d_data: []u32, d_bins: []u32) callconv(ptx.kernel) void {
    const n = d_data.len;
    const num_bins = d_bins.len;
    const step = bychunkHistogram_step;
    // var s_bins = @ptrCast([*]addrspace(.shared) u32, &bychunkHistogram_shared); // stage2
    var s_bins: [*]u32 = @ptrCast(&bychunkHistogram_shared); // stage1
    const tid = ptx.threadIdX();
    if (tid < num_bins) s_bins[ptx.threadIdX()] = 0;
    ptx.syncThreads();

    var i: u32 = 0;
    while (i < step) : (i += 1) {
        const offset = ptx.ctaIdX() * ptx.numThreadsX() * step + i * ptx.numThreadsX() + tid;
        if (offset < n) {
            // Passing a .shared pointer to atomicAdd crashes stage2 here
            // atomicAdd(&s_bins[d_data[offset]], 1);
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

// extern var cdfIncremental_shared: SharedMem align(8) addrspace(.shared); // stage2
var cdfIncremental_shared: [1024]u32 = undefined; // stage1

pub fn cdfIncremental(d_glob_bins: []u32, d_block_bins: []u32) callconv(ptx.kernel) void {
    const n = d_glob_bins.len;
    const global_id = ptx.getIdX();
    if (global_id >= n) return;
    const tid = ptx.threadIdX();

    // var d_bins = @ptrCast([*]addrspace(.shared) u32, &cdfIncremental_shared); // stage2
    const d_bins: [*]u32 = @ptrCast(&cdfIncremental_shared); // stage1
    ptx.syncThreads();
    const last_tid = ptx.lastTid(@intCast(n));
    const total = ptx.exclusiveScan(.Add, d_bins, tid, last_tid);
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
