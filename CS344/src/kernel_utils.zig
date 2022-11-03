// TODO: should be moved to Cudaz
const std = @import("std");
const builtin = @import("builtin");
const TypeInfo = std.builtin.TypeInfo;
const CallingConvention = @import("std").builtin.CallingConvention;

pub const is_device = builtin.cpu.arch == .nvptx64;
pub const Kernel = if (is_device) CallingConvention.PtxKernel else CallingConvention.Inline;

// Size for storing a thread id
pub const utid = u32;

pub fn threadIdX() utid {
    if (!is_device) return 0;
    return asm ("mov.u32 \t%[r], %tid.x;"
        : [r] "=r" (-> utid),
    );
}

pub fn blockDimX() utid {
    if (!is_device) return 0;
    var ntid = asm ("mov.u32 \t%[r], %ntid.x;"
        : [r] "=r" (-> utid),
    );
    return ntid;
}

pub fn blockIdX() utid {
    if (!is_device) return 0;
    var ctaid = asm ("mov.u32 \t%[r], %ctaid.x;"
        : [r] "=r" (-> utid),
    );
    return ctaid;
}

pub fn gridDimX() utid {
    if (!is_device) return 0;
    var nctaid = asm ("mov.u32 \t%[r], %nctaid.x;"
        : [r] "=r" (-> utid),
    );
    return nctaid;
}

pub fn getId_1D() utid {
    return threadIdX() + blockDimX() * blockIdX();
}

pub fn threadIdY() utid {
    if (!is_device) return 0;
    var tid = asm ("mov.u32 \t%[r], %tid.y;"
        : [r] "=r" (-> utid),
    );
    return tid;
}

pub fn blockDimY() utid {
    if (!is_device) return 0;
    var ntid = asm ("mov.u32 \t%[r], %ntid.y;"
        : [r] "=r" (-> utid),
    );
    return ntid;
}

pub fn blockIdY() utid {
    if (!is_device) return 0;
    var ctaid = asm ("mov.u32 \t%[r], %ctaid.y;"
        : [r] "=r" (-> utid),
    );
    return ctaid;
}

pub fn threadIdZ() utid {
    if (!is_device) return 0;
    var tid = asm ("mov.u32 \t%[r], %tid.z;"
        : [r] "=r" (-> utid),
    );
    return tid;
}

pub fn blockDimZ() utid {
    if (!is_device) return 0;
    var ntid = asm ("mov.u32 \t%[r], %ntid.z;"
        : [r] "=r" (-> utid),
    );
    return ntid;
}

pub fn blockIdZ() utid {
    if (!is_device) return 0;
    var ctaid = asm ("mov.u32 \t%[r], %ctaid.z;"
        : [r] "=r" (-> utid),
    );
    return ctaid;
}
pub fn syncThreads() void {
    if (!is_device) return;
    asm volatile ("bar.sync \t0;");
}

pub fn atomicAdd(x: *u32, a: u32) void {
    _ = @atomicRmw(u32, x, .Add, a, .SeqCst);
}

pub fn lastTid(n: usize) utid {
    var block_dim = blockDimX();
    if (blockIdX() == gridDimX() - 1) {
        return @intCast(utid, (n - 1) % block_dim);
    } else {
        return block_dim - 1;
    }
}

const Dim2 = struct { x: usize, y: usize };
pub fn getId_2D() Dim2 {
    return Dim2{
        .x = threadIdX() + blockDimX() * blockIdX(),
        .y = threadIdY() + blockDimY() * blockIdY(),
    };
}

const Dim3 = struct { x: u32, y: u32, z: u32 };
pub fn getId_3D() Dim3 {
    return Dim3{
        .x = threadIdX() + blockDimX() * blockIdX(),
        .y = threadIdY() + blockDimY() * blockIdY(),
        .z = threadIdZ() + blockDimZ() * blockIdZ(),
    };
}

pub fn divApprox(x: f32, y: f32) f32 {
    return asm ("div.approx.f32 \t%[r], %[x], %[y];"
        : [r] "=f" (-> f32),
        : [x] "f" (x),
          [y] "f" (y),
    );
}

pub fn log10(x: f32) f32 {
    // TODO: investigate why @log10 resolve to `log10f` and not some asm.
    const log2x = asm ("lg2.approx.f32 \t%[r], %[x];"
        : [r] "=f" (-> f32),
        : [x] "f" (x),
    );
    const log10_quotient: f32 = 1.0 / @log2(10.0);
    return log2x * log10_quotient;
}

// pub fn exportModule(comptime Module: anytype, comptime Exports: anytype) void {
//     if (!is_device) return;
//     // TODO assert call conv
//     const fields: []const TypeInfo.StructField = std.meta.fields(Exports);
//     // var args_ptrs: [fields.len:0]usize = undefined;
//     // https://github.com/ziglang/zig/issues/12532
//     inline for (fields) |field, i| {
//         @export(@field(Module, field.name), .{ .name = field.name, .linkage = .Strong });
//     }
// }

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
