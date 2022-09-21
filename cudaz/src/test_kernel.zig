const ptx = @import("nvptx.zig");
const message = "Hello World !\x00";
pub const panic = ptx.panic;

pub fn testMathLog10(x: []const f32, out: []f32) callconv(ptx.Kernel) void {
    const i = ptx.getIdX();
    if (i >= x.len) return;
    // `call float @llvm.log10.f32(float %0)`
    // doesn't seem to be handled by LLVM14-ptx backend
    // TODO: Try with LLVM15
    // out[i].* = @log10(x[i]);
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

comptime {
    if (ptx.is_device) {
        @export(testHelloWorld, .{ .name = "testHelloWorld" });
        @export(testMathLog10, .{ .name = "testMathLog10" });
        @export(testSwap2WithSharedBuff, .{ .name = "testSwap2WithSharedBuff" });
    }
}
