// TODO: should be moved to Cudaz
const std = @import("std");
const TypeInfo = std.builtin.TypeInfo;
const CallingConvention = std.builtin.CallingConvention;
const builtin = @import("builtin");

pub const is_nvptx = builtin.cpu.arch == .nvptx64;
pub const kernel: CallingConvention = if (is_nvptx) CallingConvention.kernel else CallingConvention.Unspecified;

// Size for storing a thread id
pub const utid = u32;

pub fn panic(msg: []const u8, error_return_trace: ?*std.builtin.StackTrace) noreturn {
    _ = error_return_trace;
    _ = msg;
    asm volatile ("trap;");
    unreachable;
}

pub fn threadIdX() utid {
    if (!is_nvptx) return 0;
    const tid = asm volatile ("mov.u32 \t%[r], %tid.x;"
        : [r] "=r" (-> utid),
    );
    return tid;
}

pub fn blockDimX() utid {
    if (!is_nvptx) return 0;
    return @workGroupSize(0);
}

pub fn blockIdX() utid {
    if (!is_nvptx) return 0;
    const ctaid = asm volatile ("mov.u32 \t%[r], %ctaid.x;"
        : [r] "=r" (-> utid),
    );
    return ctaid;
}

pub fn gridDimX() utid {
    if (!is_nvptx) return 0;
    const nctaid = asm volatile ("mov.u32 \t%[r], %nctaid.x;"
        : [r] "=r" (-> utid),
    );
    return nctaid;
}

pub fn getIdX() utid {
    return threadIdX() + blockDimX() * blockIdX();
}

pub fn threadIdY() utid {
    if (!is_nvptx) return 0;
    const tid = asm volatile ("mov.u32 \t%[r], %tid.y;"
        : [r] "=r" (-> utid),
    );
    return tid;
}

pub fn blockDimY() utid {
    if (!is_nvptx) return 0;
    const ntid = asm volatile ("mov.u32 \t%[r], %ntid.y;"
        : [r] "=r" (-> utid),
    );
    return ntid;
}

pub fn blockIdY() utid {
    if (!is_nvptx) return 0;
    const ctaid = asm volatile ("mov.u32 \t%[r], %ctaid.y;"
        : [r] "=r" (-> utid),
    );
    return ctaid;
}

pub fn threadIdZ() utid {
    if (!is_nvptx) return 0;
    const tid = asm volatile ("mov.u32 \t%[r], %tid.z;"
        : [r] "=r" (-> utid),
    );
    return tid;
}

pub fn blockDimZ() utid {
    if (!is_nvptx) return 0;
    const ntid = asm volatile ("mov.u32 \t%[r], %ntid.z;"
        : [r] "=r" (-> utid),
    );
    return ntid;
}

pub fn blockIdZ() utid {
    if (!is_nvptx) return 0;
    const ctaid = asm volatile ("mov.u32 \t%[r], %ctaid.z;"
        : [r] "=r" (-> utid),
    );
    return ctaid;
}
pub fn syncThreads() void {
    if (!is_nvptx) return;
    asm volatile ("bar.sync \t0;");
}

pub fn atomicAdd(x: *u32, a: u32) void {
    _ = @atomicRmw(u32, x, .Add, a, .SeqCst);
}

pub fn lastTid(n: usize) utid {
    const block_dim = blockDimX();
    if (blockIdX() == gridDimX() - 1) {
        return @intCast((n - 1) % block_dim);
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

// pub fn exportModule(comptime Module: anytype, comptime Exports: anytype) void {
//     if (!is_nvptx) return;
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
            const right = data[tid];
            const left = data[tid - step];
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
            const right = data[tid];
            const left = data[tid - step];
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
