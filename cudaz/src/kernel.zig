// const ptx = @import("nvptx.zig");
const message = []u8{ 72, 101, 108, 108, 111, 32, 119, 111, 114, 108, 100, 33, 13, 10 };

export fn hello(out: []u8) void {
    const i = getId_1D();
    if (i > message.len or i > out.len) return;
    // ptx.syncThreads();
    out[i] = message[i];
}

pub inline fn threadIdX() usize {
    var tid = asm volatile ("mov.u32 \t$0, %tid.x;"
        : [ret] "=r" (-> u32),
    );
    return @intCast(usize, tid);
}

pub inline fn threadDimX() usize {
    var ntid = asm volatile ("mov.u32 \t$0, %ntid.x;"
        : [ret] "=r" (-> u32),
    );
    return @intCast(usize, ntid);
}

pub inline fn gridIdX() usize {
    var ctaid = asm volatile ("mov.u32 \t$0, %ctaid.x;"
        : [ret] "=r" (-> u32),
    );
    return @intCast(usize, ctaid);
}

pub inline fn getId_1D() usize {
    return threadIdX() + threadDimX() * gridIdX();
}
