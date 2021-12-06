const std = @import("std");
const builtin = @import("builtin");
const CallingConvention = @import("std").builtin.CallingConvention;
const is_nvptx = builtin.cpu.arch == .nvptx64;
const PtxKernel = if (is_nvptx) CallingConvention.PtxKernel else CallingConvention.Unspecified;

pub export fn transposeCpu(data: []const u32, trans: []u32, num_cols: usize) callconv(PtxKernel) void {
    var i: usize = 0;
    while (i < num_cols) : (i += 1) {
        var j: usize = 0;
        while (j < num_cols) : (j += 1) {
            trans[num_cols * i + j] = data[num_cols * j + i];
        }
    }
}

pub export fn transposePerRow(data: []const u32, trans: []u32, num_cols: usize) callconv(PtxKernel) void {
    const i = getIdX();
    var j: usize = 0;
    while (j < num_cols) : (j += 1) {
        trans[num_cols * i + j] = data[num_cols * j + i];
    }
}

pub export fn transposePerCell(data: []const u32, trans: []u32, num_cols: usize) callconv(PtxKernel) void {
    const i = getIdX();
    const j = getIdY();
    if (i >= num_cols or j >= num_cols) return;
    trans[num_cols * i + j] = data[num_cols * j + i];
}

pub const block_size = 16;
// Stage1 can't parse addrspace, so we use pre-processing tricks to only
// set the addrspace in Stage2.
// In Cuda, the kernel will have access to shared memory. This memory
// can have a compile-known size or a dynamic size.
// In the case of dynamic size the corresponding Cuda code is:
// extern __shared__ int buffer[];
// In Zig, the only type with unknown size is "opaque".
// Also note that the extern keyword is technically not correct, because the variable
// isn't defined in another compilation unit.
// This seems to only work for Ptx target, not sure why.
// The generated .ll code will be:
// `@transpose_per_block_buffer = external dso_local addrspace(3) global %lesson5_kernel.SharedMem, align 8`
// Corresponding .ptx:
// `.extern .shared .align 8 .b8 transpose_per_block_buffer[]`
const SharedMem = opaque {};
// extern var transpose_per_block_buffer: SharedMem align(8) addrspace(.shared); // stage2
var transpose_per_block_buffer: [block_size][block_size]u32 = undefined; // stage1

/// Each threads copy one element to the shared buffer and then back to the output
/// The speed up comes from the fact that all threads in the block will read contiguous
/// data and then write contiguous data.
pub export fn transposePerBlock(data: []const u32, trans: []u32, num_cols: usize) callconv(PtxKernel) void {
    if (!is_nvptx) return;
    // var buffer = @ptrCast([*]addrspace(.shared) [block_size]u32, &transpose_per_block_buffer); // stage2
    var buffer = @ptrCast([*][block_size]u32, &transpose_per_block_buffer); // stage1
    const block_i = gridIdX() * block_size;
    const block_j = gridIdY() * block_size;
    const block_out_i = block_j;
    const block_out_j = block_i;
    const i = threadIdX();
    const j = threadIdY();

    // coalesced read
    if (i + block_i < num_cols and j + block_j < num_cols) {
        buffer[j][i] = data[num_cols * (block_j + j) + (block_i + i)];
    }
    syncThreads();

    // coalesced write
    if (i + block_out_i < num_cols and j + block_out_j < num_cols) {
        trans[num_cols * (block_out_j + j) + (block_out_i + i)] = buffer[i][j];
    }
}

pub const block_size_inline = block_size;
// pub var transpose_per_block_inlined_buffer: [16][block_size][block_size]u32 addrspace(.shared) = undefined; // stage2
pub var transpose_per_block_inlined_buffer: [16][block_size][block_size]u32 = undefined; // stage1

/// Each threads copy a `block_size` contiguous elements to the shared buffer
/// and copy non-contiguous element from the buffer to a contiguous slice of the output
pub export fn transposePerBlockInlined(data: []const u32, trans: []u32, num_cols: usize) callconv(PtxKernel) void {
    var buffer = &transpose_per_block_inlined_buffer[threadIdX()];
    const block_i = getIdX() * block_size;
    const block_j = gridIdY() * block_size;
    const block_out_i = block_j;
    const block_out_j = block_i;
    const i = threadIdY();
    if (i + block_i >= num_cols) return;
    var j: usize = 0;
    // coalesced read
    while (j < block_size and j + block_j < num_cols) : (j += 1) {
        buffer[j][i] = data[num_cols * (block_j + j) + (block_i + i)];
    }

    syncThreads();

    if (block_out_i + i >= num_cols) return;
    // coalesced write
    j = 0;
    while (j < block_size and block_out_j + j < num_cols) : (j += 1) {
        trans[num_cols * (block_out_j + j) + (block_out_i + i)] = buffer[i][j];
    }
}

pub inline fn threadIdX() usize {
    if (!is_nvptx) return 0;
    var tid = asm volatile ("mov.u32 \t$0, %tid.x;"
        : [ret] "=r" (-> u32),
    );
    return @intCast(usize, tid);
}

pub inline fn threadDimX() usize {
    if (!is_nvptx) return 0;
    var ntid = asm volatile ("mov.u32 \t$0, %ntid.x;"
        : [ret] "=r" (-> u32),
    );
    return @intCast(usize, ntid);
}

pub inline fn gridIdX() usize {
    if (!is_nvptx) return 0;
    var ctaid = asm volatile ("mov.u32 \t$0, %ctaid.x;"
        : [ret] "=r" (-> u32),
    );
    return @intCast(usize, ctaid);
}

pub inline fn threadIdY() usize {
    if (!is_nvptx) return 0;
    var tid = asm volatile ("mov.u32 \t$0, %tid.y;"
        : [ret] "=r" (-> u32),
    );
    return @intCast(usize, tid);
}

pub inline fn threadDimY() usize {
    if (!is_nvptx) return 0;
    var ntid = asm volatile ("mov.u32 \t$0, %ntid.y;"
        : [ret] "=r" (-> u32),
    );
    return @intCast(usize, ntid);
}

pub inline fn gridIdY() usize {
    if (!is_nvptx) return 0;
    var ctaid = asm volatile ("mov.u32 \t$0, %ctaid.y;"
        : [ret] "=r" (-> u32),
    );
    return @intCast(usize, ctaid);
}

pub inline fn getIdX() usize {
    return threadIdX() + threadDimX() * gridIdX();
}
pub inline fn getIdY() usize {
    return threadIdY() + threadDimY() * gridIdY();
}

pub inline fn syncThreads() void {
    // @"llvm.nvvm.barrier0"();
    if (!is_nvptx) return;
    asm volatile ("bar.sync \t0;");
}
