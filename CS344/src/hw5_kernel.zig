const std = @import("std");
const builtin = @import("builtin");
const CallingConvention = @import("std").builtin.CallingConvention;
const is_nvptx = builtin.cpu.arch == .nvptx64;
const PtxKernel = if (is_nvptx) CallingConvention.PtxKernel else CallingConvention.Unspecified;

pub export fn atomicHistogram(d_data: []u32, d_bins: []u32) callconv(PtxKernel) void {
    const gid = threadIdX() + blockDimX() * blockIdX();
    if (gid >= d_data.len) return;

    const bin = d_data[gid];
    atomicAdd(&d_bins[bin], 1);
}

// const step: u32 = 32;
const SharedMem = opaque {};
// extern var bychunkHistogram_shared: SharedMem align(8) addrspace(.shared); // stage2
var bychunkHistogram_shared: [1024]u32 = undefined; // stage1

const step: u32 = 32;

/// Fist accumulate into a shared histogram
/// then accumulate to the global histogram.
/// Should decreases contention when doing atomic adds.
pub export fn bychunkHistogram(d_data: []u32, d_bins: []u32) callconv(PtxKernel) void {
    const n = d_data.len;
    const num_bins = d_bins.len;
    // var s_bins = @ptrCast([*]addrspace(.shared) u32, &bychunkHistogram_shared); // stage2
    var s_bins = @ptrCast([*]u32, &bychunkHistogram_shared); // stage1
    const tid = threadIdX();
    if (tid < num_bins) s_bins[threadIdX()] = 0;
    syncThreads();

    var i: u32 = 0;
    while (i < step) : (i += 1) {
        const offset = blockIdX() * blockDimX() * step + i * blockDimX() + tid;
        if (offset < n) {
            // Passing a .shared pointer to atomicAdd crashes stage2 here
            // atomicAdd(&s_bins[d_data[offset]], 1);
            _ = @atomicRmw(u32, &s_bins[d_data[offset]], .Add, 1, .SeqCst);
        }
    }

    syncThreads();
    if (tid < num_bins) {
        atomicAdd(&d_bins[tid], s_bins[tid]);
    }
}

pub inline fn threadIdX() usize {
    if (!is_nvptx) return 0;
    var tid = asm volatile ("mov.u32 \t$0, %tid.x;"
        : [ret] "=r" (-> u32),
    );
    return @intCast(usize, tid);
}

pub inline fn blockDimX() usize {
    if (!is_nvptx) return 0;
    var ntid = asm volatile ("mov.u32 \t$0, %ntid.x;"
        : [ret] "=r" (-> u32),
    );
    return @intCast(usize, ntid);
}

pub inline fn blockIdX() usize {
    if (!is_nvptx) return 0;
    var ctaid = asm volatile ("mov.u32 \t$0, %ctaid.x;"
        : [ret] "=r" (-> u32),
    );
    return @intCast(usize, ctaid);
}

pub inline fn syncThreads() void {
    if (!is_nvptx) return;
    asm volatile ("bar.sync \t0;");
}

pub inline fn atomicAdd(x: *u32, a: u32) void {
    _ = @atomicRmw(u32, x, .Add, a, .SeqCst);
}
