const std = @import("std");
const builtin = @import("builtin");
const CallingConvention = std.builtin.CallingConvention;
pub const is_nvptx = builtin.cpu.arch == .nvptx64;
pub const Kernel: CallingConvention = if (is_nvptx) .Kernel else .C;

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
//     return @intCast(@"llvm.nvvm.read.ptx.sreg.ntid.x"());
// }
// extern fn @"llvm.nvvm.read.ptx.sreg.ntid.x"() i32;

pub fn threadIdX() usize {
    if (!is_nvptx) return 0;
    const tid = asm volatile ("mov.u32 \t%[r], %tid.x;"
        : [r] "=r" (-> u32),
    );
    return @as(usize, tid);
}

pub fn blockDimX() usize {
    if (!is_nvptx) return 0;
    const ntid = asm volatile ("mov.u32 \t%[r], %ntid.x;"
        : [r] "=r" (-> u32),
    );
    return @as(usize, ntid);
}

pub fn blockIdX() usize {
    if (!is_nvptx) return 0;
    const ctaid = asm volatile ("mov.u32 \t%[r], %ctaid.x;"
        : [r] "=r" (-> u32),
    );
    return @as(usize, ctaid);
}

pub fn gridDimX() usize {
    if (!is_nvptx) return 0;
    const nctaid = asm volatile ("mov.u32 \t%[r], %nctaid.x;"
        : [r] "=r" (-> u32),
    );
    return @as(usize, nctaid);
}

pub fn getIdX() usize {
    return threadIdX() + blockDimX() * blockIdX();
}

/// threadId.y
pub inline fn threadIdY() usize {
    const tid = asm volatile ("mov.u32 \t%[r], %tid.y;"
        : [r] "=r" (-> u32),
    );
    return @intCast(tid);
}
/// threadId.z
pub inline fn threadIdZ() usize {
    const tid = asm volatile ("mov.u32 \t%[r], %tid.z;"
        : [r] "=r" (-> u32),
    );
    return @intCast(tid);
}

/// threadDim.y
pub inline fn threadDimY() usize {
    const ntid = asm volatile ("mov.u32 \t%[r], %ntid.y;"
        : [r] "=r" (-> u32),
    );
    return @intCast(ntid);
}
/// threadDim.z
pub inline fn threadDimZ() usize {
    const ntid = asm volatile ("mov.u32 \t%[r], %ntid.z;"
        : [r] "=r" (-> u32),
    );
    return @intCast(ntid);
}

/// gridId.y
pub inline fn gridIdY() usize {
    const ctaid = asm volatile ("mov.u32 \t%[r], %ctaid.y;"
        : [r] "=r" (-> u32),
    );
    return @intCast(ctaid);
}
/// gridId.z
pub inline fn gridIdZ() usize {
    const ctaid = asm volatile ("mov.u32 \t%[r], %ctaid.z;"
        : [r] "=r" (-> u32),
    );
    return @intCast(ctaid);
}

/// gridDim.y
pub inline fn gridDimY() usize {
    const nctaid = asm volatile ("mov.u32 \t%[r], %nctaid.y;"
        : [r] "=r" (-> u32),
    );
    return @intCast(nctaid);
}
/// gridDim.z
pub inline fn gridDimZ() usize {
    const nctaid = asm volatile ("mov.u32 \t%[r], %nctaid.z;"
        : [r] "=r" (-> u32),
    );
    return @intCast(nctaid);
}

const Dim2 = struct { x: usize, y: usize };
pub fn getId_2D() Dim2 {
    return Dim2{
        .x = threadIdX() + blockDimX() * blockIdX(),
        .y = threadIdY() + threadDimY() * gridIdY(),
    };
}

const Dim3 = struct { x: usize, y: usize, z: usize };
pub fn getId_3D() Dim3 {
    return Dim3{
        .x = threadIdX() + blockDimX() * blockIdX(),
        .y = threadIdY() + threadDimY() * gridIdY(),
        .z = threadIdZ() + threadDimZ() * gridIdZ(),
    };
}

// var panic_message_buffer: ?[]u8 = null;

// pub export fn init_panic_message_buffer(buffer: []u8) callconv(Kernel) void {
//     panic_message_buffer = buffer;
// }
// if (!is_nvptx) @compileError("This panic handler is made for GPU");

pub fn panic(msg: []const u8, error_return_trace: ?*std.builtin.StackTrace, ret_addr: ?usize) noreturn {
    _ = ret_addr;
    _ = error_return_trace;
    _ = msg;
    // asm volatile ("trap;");
    // `unreachable` implictly calls panic recursively and confuses ptxas.
    unreachable;
    // `noreturn` crashes LLVM because "Basic Block in function 'nvptx.panic' does not have terminator!"
    // This seems to be a bad .ll generation
    // return asm volatile ("trap;"
    //     : [r] "=r" (-> noreturn),
    // );
    // while(true) fails to compile because of "LLVM ERROR: Symbol name with unsupported characters"
    // while(true){}
}
// if (panic_message_buffer) |*buffer| {
// const len = std.math.min(msg.len, buffer.len);
// std.mem.copy(u8, buffer.*.ptr[0..len], msg[0..len]);
// TODO: this assumes nobody will try to write afterward, which I'm not sure
// TODO: prevent all threads wirting in the same place
// buffer.*.len = len;
// }

const message = "Hello World !\x00";

pub export fn _test_hello_world(out: [*]u8, len: usize) callconv(Kernel) void {
    const i = getIdX();
    if (i > message.len or i > len) return;
    syncThreads();
    out[i] = message[i];
}
