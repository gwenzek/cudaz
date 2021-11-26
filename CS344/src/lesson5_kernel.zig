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

const block_size = 16;
// Stage1 can't parse addrspace, so we use pre-processing tricks to only
// set the addrspace in Stage2.
// pub var transpose_per_block_buffer: [block_size][block_size]u32 addrspace(.fs) = undefined; // stage2
pub var transpose_per_block_buffer: [block_size][block_size]u32 = undefined; // stage1

// **** **** **** ****
// **** **** **** ****
// **** **** **** ****
// **** **** **** ****

// **** **** **** ****
// **** **** **** ****
// **** **** **** ****
// **** **** **** ****

pub export fn transposePerBlock(data: []const u32, trans: []u32, num_cols: usize) callconv(PtxKernel) void {
    var buffer = &transpose_per_block_buffer;
    const block_i = gridIdX() * block_size;
    const block_j = gridIdY() * block_size;
    const block_out_i = block_j;
    const block_out_j = block_i;
    const i = threadIdX();
    const j = threadIdY();
    if (i + block_i >= num_cols or j + block_j >= num_cols) return;

    // coalesced read
    buffer[j][i] = data[num_cols * (block_j + j) + (block_i + i)];
    syncThreads();

    // coalesced write
    trans[num_cols * (block_out_j + j) + (block_out_i + i)] = buffer[i][j];
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
