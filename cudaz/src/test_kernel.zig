const ptx = @import("nvptx.zig");
const std = @import("std");
const builtin = @import("builtin");
const message = "Hello World !\x00";
pub const panic = ptx.panic;

pub fn testMathLog10(x: []const f32, out: []f32) callconv(ptx.Kernel) void {
    const i = ptx.getIdX();
    if (i >= x.len) return;
    // `call float @llvm.log10.f32(float %0)`
    // doesn't seem to be handled by LLVM14-ptx backend
    // TODO: Try with LLVM15
    // out[i] = @log10(x[i]);
    out[i] = _log10(x[i]);
}

fn _log10(x: f32) f32 {
    const log2_x = asm ("lg2.approx.f32 \t%[r], %[x];"
        : [r] "=r" (-> f32),
        : [x] "r" (x),
    );
    return log2_x / @log2(10.0);
}

pub fn testHelloWorld(out: []u8) callconv(ptx.Kernel) void {
    const i = ptx.getIdX();
    if (i > message.len or i > out.len) return;
    ptx.syncThreads();
    out[i] = message[i];
}

var shared_buffer: [2]u8 align(8) addrspace(.shared) = undefined;

pub fn testSwap2WithSharedBuff(src: []const u8, tgt: []u8) callconv(ptx.Kernel) void {
    if (!ptx.is_device) return;
    var buffer = &shared_buffer;
    const i = ptx.threadIdX();
    const x = ptx.getIdX();
    buffer[i] = src[x];
    ptx.syncThreads();
    tgt[x] = buffer[1 - i % 2];
}

var _sdata: [1024]f32 addrspace(.shared) = undefined;

pub fn testReduceSum(d_x: []const f32, out: *f32) callconv(ptx.Kernel) void {
    var sdata = @addrSpaceCast(std.builtin.AddressSpace.generic, &_sdata);
    const tid = ptx.threadIdX();
    var sum = d_x[tid];
    sdata[tid] = sum;
    asm volatile ("bar.sync \t0;");
    var s: u32 = 512;
    while (s > 0) : (s = s >> 1) {
        if (tid < s) {
            sum += sdata[tid + s];
            sdata[tid] = sum;
        }
        asm volatile ("bar.sync \t0;");
    }

    if (tid == 0) {
        out.* = sdata[tid];
    }
}

comptime {
    if (ptx.is_device) {
        @export(testHelloWorld, .{ .name = "testHelloWorld" });
        @export(testMathLog10, .{ .name = "testMathLog10" });
        @export(testSwap2WithSharedBuff, .{ .name = "testSwap2WithSharedBuff" });
        @export(testReduceSum, .{ .name = "testReduceSum" });
    }
}
