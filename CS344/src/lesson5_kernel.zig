const std = @import("std");

pub export fn transposeCpu(data: []const u32, trans: []u32, num_cols: usize) callconv(.PtxKernel) void {
    var i: usize = 0;
    while (i < num_cols) : (i += 1) {
        var j: usize = 0;
        while (j < num_cols) : (j += 1) {
            trans[num_cols * i + j] = data[num_cols * j + i];
        }
    }
}

pub export fn transposePerRow(data: []const u32, trans: []u32, num_cols: usize) callconv(.PtxKernel) void {
    const i = getIdX();
    var j: usize = 0;
    while (j < num_cols) : (j += 1) {
        trans[num_cols * i + j] = data[num_cols * j + i];
    }
}

const builtin = @import("builtin");
const is_nvptx = builtin.cpu.arch == .nvptx64;

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

pub inline fn getIdX() usize {
    return threadIdX() + threadDimX() * gridIdX();
}
