const builtin = @import("builtin");
const is_nvptx = builtin.cpu.arch == .nvptx64;

// Equivalent of Cuda's __syncthreads()
/// Wait to all the threads in this block to reach this barrier
/// before going on.
pub inline fn syncThreads() void {
    // @"llvm.nvvm.barrier0"();
    if (!is_nvptx) return;
    asm volatile ("bar.sync \t0;");
}

// extern fn @"llvm.nvvm.barrier0"() void;

// This doesn't seem to work. LLVM (called from Zig) crashes with a "Cannot select error"
// pub inline fn threadDimX() usize {
//     return @intCast(usize, @"llvm.nvvm.read.ptx.sreg.ntid.x"());
// }
// extern fn @"llvm.nvvm.read.ptx.sreg.ntid.x"() i32;

/// threadId.x
pub inline fn threadIdX() usize {
    if (!is_nvptx) return 0;
    var tid = asm volatile ("mov.u32 \t%[ret], %%tid.x;"
        : [ret] "=r" (-> u32)
    );
    return @intCast(usize, tid);
}
/// threadId.y
pub inline fn threadIdY() usize {
    var tid = asm volatile ("mov.u32 \t$0, %tid.y;"
        : [ret] "=r" (-> u32)
    );
    return @intCast(usize, tid);
}
/// threadId.z
pub inline fn threadIdZ() usize {
    var tid = asm volatile ("mov.u32 \t$0, %tid.z;"
        : [ret] "=r" (-> u32)
    );
    return @intCast(usize, tid);
}

/// threadDim.x
pub inline fn threadDimX() usize {
    if (!is_nvptx) return 0;
    var ntid = asm volatile ("mov.u32 \t%[ret], %%ntid.x;"
        : [ret] "=r" (-> u32)
    );
    return @intCast(usize, ntid);
}
/// threadDim.y
pub inline fn threadDimY() usize {
    var ntid = asm volatile ("mov.u32 \t$0, %ntid.y;"
        : [ret] "=r" (-> u32)
    );
    return @intCast(usize, ntid);
}
/// threadDim.z
pub inline fn threadDimZ() usize {
    var ntid = asm volatile ("mov.u32 \t$0, %ntid.z;"
        : [ret] "=r" (-> u32)
    );
    return @intCast(usize, ntid);
}

/// gridId.x
pub inline fn gridIdX() usize {
    if (!is_nvptx) return 0;
    var ctaid = asm volatile ("mov.u32 \t%[ret], %%ctaid.x;"
        : [ret] "=r" (-> u32)
    );
    return @intCast(usize, ctaid);
}
/// gridId.y
pub inline fn gridIdY() usize {
    var ctaid = asm volatile ("mov.u32 \t$0, %ctaid.y;"
        : [ret] "=r" (-> u32)
    );
    return @intCast(usize, ctaid);
}
/// gridId.z
pub inline fn gridIdZ() usize {
    var ctaid = asm volatile ("mov.u32 \t$0, %ctaid.z;"
        : [ret] "=r" (-> u32)
    );
    return @intCast(usize, ctaid);
}

/// gridDim.x
pub inline fn gridDimX() usize {
    var nctaid = asm volatile ("mov.u32 \t$0, %nctaid.x;"
        : [ret] "=r" (-> u32)
    );
    return @intCast(usize, nctaid);
}
/// gridDim.y
pub inline fn gridDimY() usize {
    var nctaid = asm volatile ("mov.u32 \t$0, %nctaid.y;"
        : [ret] "=r" (-> u32)
    );
    return @intCast(usize, nctaid);
}
/// gridDim.z
pub inline fn gridDimZ() usize {
    var nctaid = asm volatile ("mov.u32 \t$0, %nctaid.z;"
        : [ret] "=r" (-> u32)
    );
    return @intCast(usize, nctaid);
}

pub inline fn getId_1D() usize {
    return threadIdX() + threadDimX() * gridIdX();
}

const Dim2 = struct { x: usize, y: usize };
pub fn getId_2D() Dim2 {
    return Dim2{
        .x = threadIdX() + threadDimX() * gridIdX(),
        .y = threadIdY() + threadDimY() * gridIdY(),
    };
}

const Dim3 = struct { x: usize, y: usize, z: usize };
pub fn getId_3D() Dim3 {
    return Dim3{
        .x = threadIdX() + threadDimX() * gridIdX(),
        .y = threadIdY() + threadDimY() * gridIdY(),
        .z = threadIdZ() + threadDimZ() * gridIdZ(),
    };
}

const message = "Hello World !\x00";

pub export fn _test_hello_world(out: []u8) void {
    const i = getId_1D();
    if (i > message.len or i > out.len) return;
    syncThreads();
    out[i] = message[i];
}
