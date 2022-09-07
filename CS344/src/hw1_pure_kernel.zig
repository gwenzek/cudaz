const builtin = @import("builtin");
const CallingConvention = @import("std").builtin.CallingConvention;
const is_nvptx = builtin.cpu.arch == .nvptx64;
const PtxKernel = if (is_nvptx) CallingConvention.PtxKernel else CallingConvention.Unspecified;

const ptx = @import("kernel_utils.zig");
pub const panic = ptx.panic;

pub export fn rgba_to_greyscale(rgbaImage: []u8, greyImage: []u8) callconv(PtxKernel) void {
    const i = getId_1D();
    if (i >= greyImage.len) return;
    const px = rgbaImage[i * 3 .. i * 3 + 3];
    const R = @intCast(u32, px[0]);
    const G = @intCast(u32, px[1]);
    const B = @intCast(u32, px[2]);
    var grey = @divFloor(299 * R + 587 * G + 114 * B, 1000);
    greyImage[i] = @intCast(u8, grey);
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

pub fn getId_1D() usize {
    return threadIdX() + threadDimX() * gridIdX();
}
