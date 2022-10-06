const std = @import("std");
const builtin = @import("builtin");
const CallingConvention = @import("std").builtin.CallingConvention;
const AddressSpace = @import("std").builtin.AddressSpace;
pub const is_device = builtin.cpu.arch == .nvptx64;

pub const Kernel: std.builtin.CallingConvention = if (is_device) .PtxKernel else .Inline;
pub const Shared: std.builtin.AddressSpace = if (is_device) .shared else .generic;

// Equivalent of Cuda's __syncthreads()
/// Wait to all the threads in this block to reach this barrier
/// before going on.
pub inline fn syncThreads() void {
    // @"llvm.nvvm.barrier0"();
    if (!is_device) return;
    asm volatile ("bar.sync \t0;");
}

// extern fn @"llvm.nvvm.barrier0"() void;

// This doesn't seem to work. LLVM (called from Zig) crashes with a "Cannot select error"
// pub inline fn threadDimX() usize {
//     return @intCast(usize, @"llvm.nvvm.read.ptx.sreg.ntid.x"());
// }
// extern fn @"llvm.nvvm.read.ptx.sreg.ntid.x"() i32;

pub fn threadIdX() usize {
    if (!is_device) return 0;
    var tid = asm volatile ("mov.u32 \t%[r], %tid.x;"
        : [r] "=r" (-> u32),
    );
    return @as(usize, tid);
}

pub fn blockDimX() usize {
    if (!is_device) return 0;
    var ntid = asm volatile ("mov.u32 \t%[r], %ntid.x;"
        : [r] "=r" (-> u32),
    );
    return @as(usize, ntid);
}

pub fn blockIdX() usize {
    if (!is_device) return 0;
    var ctaid = asm volatile ("mov.u32 \t%[r], %ctaid.x;"
        : [r] "=r" (-> u32),
    );
    return @as(usize, ctaid);
}

pub fn gridDimX() usize {
    if (!is_device) return 0;
    var nctaid = asm volatile ("mov.u32 \t%[r], %nctaid.x;"
        : [r] "=r" (-> u32),
    );
    return @as(usize, nctaid);
}

pub fn getIdX() usize {
    return threadIdX() + blockDimX() * blockIdX();
}

/// threadId.y
pub inline fn threadIdY() usize {
    var tid = asm volatile ("mov.u32 \t%[r], %tid.y;"
        : [r] "=r" (-> u32),
    );
    return @intCast(usize, tid);
}
/// threadId.z
pub inline fn threadIdZ() usize {
    var tid = asm volatile ("mov.u32 \t%[r], %tid.z;"
        : [r] "=r" (-> u32),
    );
    return @intCast(usize, tid);
}

/// threadDim.y
pub inline fn threadDimY() usize {
    var ntid = asm volatile ("mov.u32 \t%[r], %ntid.y;"
        : [r] "=r" (-> u32),
    );
    return @intCast(usize, ntid);
}
/// threadDim.z
pub inline fn threadDimZ() usize {
    var ntid = asm volatile ("mov.u32 \t%[r], %ntid.z;"
        : [r] "=r" (-> u32),
    );
    return @intCast(usize, ntid);
}

/// gridId.y
pub inline fn gridIdY() usize {
    var ctaid = asm volatile ("mov.u32 \t%[r], %ctaid.y;"
        : [r] "=r" (-> u32),
    );
    return @intCast(usize, ctaid);
}
/// gridId.z
pub inline fn gridIdZ() usize {
    var ctaid = asm volatile ("mov.u32 \t%[r], %ctaid.z;"
        : [r] "=r" (-> u32),
    );
    return @intCast(usize, ctaid);
}

/// gridDim.y
pub inline fn gridDimY() usize {
    var nctaid = asm volatile ("mov.u32 \t%[r], %nctaid.y;"
        : [r] "=r" (-> u32),
    );
    return @intCast(usize, nctaid);
}
/// gridDim.z
pub inline fn gridDimZ() usize {
    var nctaid = asm volatile ("mov.u32 \t%[r], %nctaid.z;"
        : [r] "=r" (-> u32),
    );
    return @intCast(usize, nctaid);
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
// if (!is_device) @compileError("This panic handler is made for GPU");

pub fn panic(msg: []const u8, error_return_trace: ?*std.builtin.StackTrace, _: ?usize) noreturn {
    _ = error_return_trace;
    _ = msg;
    @setCold(true);
    // Disable runtime safety to not introduce a recursive call to panic on "unreachable".
    @setRuntimeSafety(false);
    asm volatile ("trap;");
    unreachable;
}
// if (panic_message_buffer) |*buffer| {
// const len = std.math.min(msg.len, buffer.len);
// std.mem.copy(u8, buffer.*.ptr[0..len], msg[0..len]);
// TODO: this assumes nobody will try to write afterward, which I'm not sure
// TODO: prevent all threads wirting in the same place
// buffer.*.len = len;
// }
