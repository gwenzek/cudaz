const std = @import("std");
const builtin = @import("builtin");
const ptx = @import("kernel_utils.zig");
pub const panic = ptx.panic;

pub fn transposeCpu(data: []const u32, trans: []u32, num_cols: usize) callconv(ptx.Kernel) void {
    var i: usize = 0;
    while (i < num_cols) : (i += 1) {
        var j: usize = 0;
        while (j < num_cols) : (j += 1) {
            trans[num_cols * i + j] = data[num_cols * j + i];
        }
    }
}

pub fn transposePerRow(data: []const u32, trans: []u32, num_cols: usize) callconv(ptx.Kernel) void {
    const i = ptx.getId_1D();
    var j: usize = 0;
    while (j < num_cols) : (j += 1) {
        trans[num_cols * i + j] = data[num_cols * j + i];
    }
}

pub fn transposePerCell(data: []const u32, trans: []u32, num_cols: usize) callconv(ptx.Kernel) void {
    const coord = ptx.getId_2D();
    const i = coord.x;
    const j = coord.y;
    if (i >= num_cols or j >= num_cols) return;
    trans[num_cols * i + j] = data[num_cols * j + i];
}

pub const block_size = 16;
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
pub fn transposePerBlock(data: []const u32, trans: []u32, num_cols: usize) callconv(ptx.Kernel) void {
    // var buffer = @ptrCast([*]addrspace(.shared) [block_size]u32, &transpose_per_block_buffer); // stage2
    var buffer = @ptrCast([*][block_size]u32, &transpose_per_block_buffer); // stage1
    // var buffer = &transpose_per_block_buffer;
    const block_i = ptx.blockIdX() * block_size;
    const block_j = ptx.blockIdY() * block_size;
    const block_out_i = block_j;
    const block_out_j = block_i;
    const i = ptx.threadIdX();
    const j = ptx.threadIdY();

    // coalesced read
    if (i + block_i < num_cols and j + block_j < num_cols) {
        buffer[j][i] = data[num_cols * (block_j + j) + (block_i + i)];
    }
    ptx.syncThreads();

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
pub fn transposePerBlockInlined(data: []const u32, trans: []u32, num_cols: usize) callconv(ptx.Kernel) void {
    var buffer = &transpose_per_block_inlined_buffer[ptx.threadIdX()];
    const block_i = ptx.getId_1D() * block_size;
    const block_j = ptx.blockIdY() * block_size;
    const block_out_i = block_j;
    const block_out_j = block_i;
    const i = ptx.threadIdY();
    if (i + block_i >= num_cols) return;
    var j: usize = 0;
    // coalesced read
    while (j < block_size and j + block_j < num_cols) : (j += 1) {
        buffer[j][i] = data[num_cols * (block_j + j) + (block_i + i)];
    }

    ptx.syncThreads();

    if (block_out_i + i >= num_cols) return;
    // coalesced write
    j = 0;
    while (j < block_size and block_out_j + j < num_cols) : (j += 1) {
        trans[num_cols * (block_out_j + j) + (block_out_i + i)] = buffer[i][j];
    }
}

comptime {
    if (ptx.is_device) {
        @export(transposeCpu, .{ .name = "transposeCpu" });
        @export(transposePerRow, .{ .name = "transposePerRow" });
        @export(transposePerCell, .{ .name = "transposePerCell" });
        @export(transposePerBlock, .{ .name = "transposePerBlock" });
        @export(transposePerBlockInlined, .{ .name = "transposePerBlockInlined" });
    }
}
