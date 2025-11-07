//! Bindings to ptx intrinsics.
//!
//! The Parallel Thread Execution (PTX) programming model is explicitly parallel: a PTX program specifies the execution of a given thread of a parallel thread array. A cooperative thread array, or CTA, is an array of threads that execute a kernel concurrently or in parallel.
//!
//! Threads within a CTA can communicate with each other. To coordinate the communication of the threads within the CTA, one can specify synchronization points where threads wait until all threads in the CTA have arrived.
//!
//! Each thread has a unique thread identifier within the CTA. Programs use a data parallel decomposition to partition inputs, work, and results across the threads of the CTA.
//! Each CTA thread uses its thread identifier to determine its assigned role, assign specific input and output positions, compute addresses, and select work to perform.
//! Each thread identifier component ranges from zero up to the number of thread ids in that CTA dimension.
//!
//! Each CTA has a 1D, 2D, or 3D shape specified by a three-element vector ntid (with elements ntid.x, ntid.y, and ntid.z). The vector ntid specifies the number of threads in each CTA dimension.
//!
//! Threads within a CTA execute in SIMT (single-instruction, multiple-thread) fashion in groups called warps. A warp is a maximal subset of threads from a single CTA, such that the threads execute the same instructions at the same time. Threads within a warp are sequentially numbered. The warp size is a machine-dependent constant. Typically, a warp has 32 threads.
const std = @import("std");
const CallingConvention = std.builtin.CallingConvention;
const builtin = @import("builtin");
pub const root = @import("root");

pub const is_nvptx = builtin.cpu.arch == .nvptx64;
pub const kernel: CallingConvention = if (builtin.cpu.arch == .nvptx64) .nvptx_kernel else .auto;

comptime {
    if (is_nvptx and !@hasDecl(root, "panic")) {
        @compileError("You must add a `pub const panic = ptx.panic;` at the top of zig kernel files");
    }
}

pub const panic = if (is_nvptx) std.debug.no_panic else std.debug.simple_panic;

// Equivalent of Cuda's __syncthreads()
/// Wait to all the threads in this block to reach this barrier
/// before going on.
pub inline fn syncThreads() void {
    if (!is_nvptx) return;
    asm volatile ("bar.sync \t0;");
}

/// Id of the thread in current CTA.
pub fn threadIdX() u32 {
    return @workItemId(0);
}

/// Number of threads in current CTA.
pub fn numThreadsX() u32 {
    return @workGroupSize(0);
}

/// Id of current CTA
pub fn ctaIdX() u32 {
    return @workGroupId(0);
}

/// Number of CTA
pub fn numCtasX() u32 {
    if (comptime !is_nvptx) return 0;
    return asm ("mov.u32 \t%[r], %nctaid.x;"
        : [r] "=r" (-> u32),
    );
}

pub fn getIdX() u32 {
    return threadIdX() + numThreadsX() * ctaIdX();
}

pub fn threadIdY() u32 {
    return @workItemId(1);
}

pub fn numThreadsY() u32 {
    return @workGroupSize(1);
}

pub fn ctaIdY() u32 {
    return @workGroupId(1);
}

pub fn numCtasY() u32 {
    if (comptime !is_nvptx) return 0;
    return asm ("mov.u32 \t%[r], %nctaid.y;"
        : [r] "=r" (-> u32),
    );
}

pub fn getIdY() u32 {
    return threadIdY() + numThreadsY() * ctaIdY();
}

pub fn threadIdZ() u32 {
    return @workItemId(2);
}

pub fn numThreadsZ() u32 {
    return @workGroupSize(2);
}

pub fn ctaIdZ() u32 {
    return @workGroupId(2);
}

pub fn numCtasZ() u32 {
    if (comptime !is_nvptx) return 0;
    return asm ("mov.u32 \t%[r], %nctaid.z;"
        : [r] "=r" (-> u32),
    );
}

pub fn getIdZ() u32 {
    return threadIdZ() + numThreadsZ() * ctaIdZ();
}

pub const Dim2 = struct { x: u32, y: u32 };
pub fn getId_2D() Dim2 {
    return Dim2{
        .x = threadIdX() + numThreadsX() * ctaIdX(),
        .y = threadIdY() + numThreadsY() * ctaIdY(),
    };
}

pub const Dim3 = struct { x: u32, y: u32, z: u32 };
pub fn getId_3D() Dim3 {
    return Dim3{
        .x = threadIdX() + numThreadsX() * ctaIdX(),
        .y = threadIdY() + numThreadsY() * ctaIdY(),
        .z = threadIdZ() + numThreadsZ() * ctaIdZ(),
    };
}

/// Exclusive scan using Blelloch algorithm
/// Returns the total value which won't be part of the array
pub fn exclusiveScan(
    comptime op: std.builtin.ReduceOp,
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
                .Add => right + left,
                .Mul => right * left,
                .Min => @min(left, right),
                .Max => @max(left, right),
                .And => right & left,
                .Or => right | left,
                .Xor => right ^ left,
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
                .Add => right + left,
                .Mul => right * left,
                .Min => @min(left, right),
                .Max => @max(left, right),
                .And => right & left,
                .Or => right | left,
                .Xor => right ^ left,
            };
            data[tid - step] = right;
        }
        syncThreads();
    }
    return total;
}

pub const SharedMem = opaque {};
pub extern var shared_memory: SharedMem align(64) addrspace(.shared);

/// Expose all shared memory available to the kernel as a single slice.
pub fn sharedMemory(T: type) []align(64) addrspace(.shared) T {
    const mem_u8: [*]align(64) addrspace(.shared) u8 = @ptrCast(&shared_memory);
    return @ptrCast(mem_u8[0..totalSharedMemory()]);
}

pub fn totalSharedMemory() u32 {
    if (comptime !is_nvptx) return 0;
    return asm ("mov.u32 \t%[r], %total_smem_size;"
        : [r] "=r" (-> u32),
    );
}

pub fn chunkByCta(slice: anytype) @TypeOf(slice) {
    const chunk_size = std.math.divCeil(usize, slice.len, numCtasX()) catch unreachable;
    const trailing = slice[chunk_size * ctaIdX() ..];

    return trailing[0..@min(chunk_size, trailing.len)];
}

pub fn lastTid(n: usize) u32 {
    if (ctaIdX() == numCtasX() - 1) {
        return @intCast((n - 1) % numThreadsX());
    } else {
        return numThreadsX() - 1;
    }
}
