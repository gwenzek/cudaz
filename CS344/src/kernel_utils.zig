// TODO: should be moved to Cudaz
const std = @import("std");
const builtin = @import("builtin");
const CallingConvention = @import("std").builtin.CallingConvention;
pub const is_nvptx = builtin.cpu.arch == .nvptx64;
pub const Kernel = if (is_nvptx) CallingConvention.PtxKernel else CallingConvention.Win64;

pub fn panic(msg: []const u8, error_return_trace: ?*std.builtin.StackTrace) noreturn {
    _ = error_return_trace;
    _ = msg;
    asm volatile ("trap;");
    unreachable;
}

pub fn threadIdX() usize {
    if (!is_nvptx) return 0;
    var tid = asm volatile ("mov.u32 \t%[r], %tid.x;"
        : [r] "=r" (-> u32),
    );
    return @as(usize, tid);
}

pub fn blockDimX() usize {
    if (!is_nvptx) return 0;
    var ntid = asm volatile ("mov.u32 \t%[r], %ntid.x;"
        : [r] "=r" (-> u32),
    );
    return @as(usize, ntid);
}

pub fn blockIdX() usize {
    if (!is_nvptx) return 0;
    var ctaid = asm volatile ("mov.u32 \t%[r], %ctaid.x;"
        : [r] "=r" (-> u32),
    );
    return @as(usize, ctaid);
}

pub fn gridDimX() usize {
    if (!is_nvptx) return 0;
    var nctaid = asm volatile ("mov.u32 \t%[r], %nctaid.x;"
        : [r] "=r" (-> u32),
    );
    return @as(usize, nctaid);
}

pub fn getIdX() usize {
    return threadIdX() + blockDimX() * blockIdX();
}

pub fn threadIdY() usize {
    if (!is_nvptx) return 0;
    var tid = asm volatile ("mov.u32 \t$0, %tid.y;"
        : [ret] "=r" (-> u32),
    );
    return @as(usize, tid);
}

pub fn blockDimY() usize {
    if (!is_nvptx) return 0;
    var ntid = asm volatile ("mov.u32 \t$0, %ntid.y;"
        : [ret] "=r" (-> u32),
    );
    return @as(usize, ntid);
}

pub fn blockIdY() usize {
    if (!is_nvptx) return 0;
    var ctaid = asm volatile ("mov.u32 \t$0, %ctaid.y;"
        : [ret] "=r" (-> u32),
    );
    return @as(usize, ctaid);
}

pub fn syncThreads() void {
    if (!is_nvptx) return;
    asm volatile ("bar.sync \t0;");
}

pub fn atomicAdd(x: *u32, a: u32) void {
    _ = @atomicRmw(u32, x, .Add, a, .SeqCst);
}

pub fn lastTid(n: usize) usize {
    var block_dim = blockDimX();
    return if (blockIdX() == gridDimX() - 1) (n - 1) % block_dim else block_dim - 1;
}

pub const Operator = enum { add, mul, min, max };

/// Exclusive scan using Blelloch algorithm
/// Returns the total value which won't be part of the array
// TODO: use generics once it works in Stage2
pub fn exclusiveScan(
    comptime op: Operator,
    data: anytype,
    tid: usize,
    last_tid: usize,
) u32 {
    var step: u32 = 1;
    while (step <= last_tid) : (step *= 2) {
        if (tid >= step and (last_tid - tid) % (step * 2) == 0) {
            var right = data[tid];
            var left = data[tid - step];
            data[tid] = switch (op) {
                .add => right + left,
                .mul => right * left,
                .min => if (left < right) left else right,
                .max => if (left > right) left else right,
            };
        }
        syncThreads();
    }

    var total: u32 = 0;
    if (tid == last_tid) {
        total = data[tid];
        data[tid] = 0;
    }
    syncThreads();

    step /= 2;
    while (step > 0) : (step /= 2) {
        if (tid >= step and (last_tid - tid) % (step * 2) == 0) {
            var right = data[tid];
            var left = data[tid - step];
            data[tid] = switch (op) {
                .add => right + left,
                .mul => right * left,
                .min => if (left < right) left else right,
                .max => if (left > right) left else right,
            };
            data[tid - step] = right;
        }
        syncThreads();
    }
    return total;
}
