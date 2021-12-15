const std = @import("std");
const builtin = @import("builtin");
const CallingConvention = @import("std").builtin.CallingConvention;
const is_nvptx = builtin.cpu.arch == .nvptx64;
const PtxKernel = if (is_nvptx) CallingConvention.PtxKernel else CallingConvention.Unspecified;
const ku = @import("kernel_utils.zig");

pub export fn atomicHistogram(d_data: []u32, d_bins: []u32) callconv(PtxKernel) void {
    const gid = ku.getIdX();
    if (gid >= d_data.len) return;

    const bin = d_data[gid];
    ku.atomicAdd(&d_bins[bin], 1);
}

// const step: u32 = 32;
const SharedMem = opaque {};
// extern var bychunkHistogram_shared: SharedMem align(8) addrspace(.shared); // stage2
var bychunkHistogram_shared: [1024]u32 = undefined; // stage1

const bychunkHistogram_step: u32 = 32;

/// Fist accumulate into a shared histogram
/// then accumulate to the global histogram.
/// Should decreases contention when doing atomic adds.
pub export fn bychunkHistogram(d_data: []u32, d_bins: []u32) callconv(PtxKernel) void {
    const n = d_data.len;
    const num_bins = d_bins.len;
    const step = bychunkHistogram_step;
    // var s_bins = @ptrCast([*]addrspace(.shared) u32, &bychunkHistogram_shared); // stage2
    var s_bins = @ptrCast([*]u32, &bychunkHistogram_shared); // stage1
    const tid = ku.threadIdX();
    if (tid < num_bins) s_bins[ku.threadIdX()] = 0;
    ku.syncThreads();

    var i: u32 = 0;
    while (i < step) : (i += 1) {
        const offset = ku.blockIdX() * ku.blockDimX() * step + i * ku.blockDimX() + tid;
        if (offset < n) {
            // Passing a .shared pointer to atomicAdd crashes stage2 here
            // atomicAdd(&s_bins[d_data[offset]], 1);
            _ = @atomicRmw(u32, &s_bins[d_data[offset]], .Add, 1, .SeqCst);
        }
    }

    ku.syncThreads();
    if (tid < num_bins) {
        ku.atomicAdd(&d_bins[tid], s_bins[tid]);
    }
}

pub export fn coarseBins(d_data: []u32, d_coarse_bins: []u32) callconv(PtxKernel) void {
    const n = d_data.len;
    const id = ku.getIdX();
    if (id < n) {
        const rad = d_data[id] / 32;
        d_coarse_bins[rad * n + id] = 1;
    }
}

pub export fn shuffleCoarseBins32(
    d_coarse_bins: []u32,
    d_coarse_bins_boundaries: []u32,
    d_cdf: []const u32,
    d_in: []const u32,
) callconv(PtxKernel) void {
    const n = d_in.len;
    const id = ku.getIdX();
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

pub export fn cdfIncremental(d_glob_bins: []u32, d_block_bins: []u32) callconv(PtxKernel) void {
    const n = d_glob_bins.len;
    const global_id = ku.getIdX();
    if (global_id >= n) return;
    const tid = ku.threadIdX();

    // var d_bins = @ptrCast([*]addrspace(.shared) u32, &cdfIncremental_shared); // stage2
    var d_bins = @ptrCast([*]u32, &cdfIncremental_shared); // stage1
    ku.syncThreads();
    const last_tid = ku.lastTid(n);
    const total = ku.exclusiveScan(.add, d_bins, tid, last_tid);
    if (tid == last_tid) {
        d_block_bins[ku.blockIdX()] = total;
    }
    d_glob_bins[global_id] = d_bins[tid];
}

pub export fn cdfIncrementalShift(d_glob_bins: []u32, d_block_bins: []const u32) callconv(PtxKernel) void {
    const block_shift = d_block_bins[ku.blockIdX()];
    d_glob_bins[ku.getIdX()] += block_shift;
}
