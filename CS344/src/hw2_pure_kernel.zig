const builtin = @import("builtin");
const std = @import("std");
const is_nvptx = builtin.cpu.arch == .nvptx64;
const cu = @import("cudaz").cu;
const math = std.math;
const CallingConvention = std.builtin.CallingConvention;
const PtxKernel = if (is_nvptx) CallingConvention.PtxKernel else CallingConvention.Unspecified;

inline fn clampedOffset(x: i32, step: i32, n: i32) i32 {
    if (step < 0 and -step > x) return 0;
    const x_step = x + step;
    if (x_step >= n) return n - 1;
    return x_step;
}

pub export fn gaussianBlur(
    input: []const u8,
    num_cols: i32,
    num_rows: i32,
    filter: []f32,
    filter_width: i32,
    output: []u8,
) callconv(PtxKernel) void {
    const id = getId_3D();
    if (id.x >= num_cols or id.y >= num_rows)
        return;
    const pixel_id: i32 = id.y * num_cols + id.x;
    const channel_id = @intCast(usize, pixel_id * 3 + id.z);
    output[channel_id] = input[channel_id];

    // NOTE: If a thread's absolute position 2D position is within the image, but
    // some of its neighbors are outside the image, then you will need to be extra
    // careful. Instead of trying to read such a neighbor value from GPU memory
    // (which won't work because the value is out of bounds), you should
    // explicitly clamp the neighbor values you read to be within the bounds of
    // the image. If this is not clear to you, then please refer to sequential
    // reference solution for the exact clamping semantics you should follow.
    // _ = input;
    // _ = filter;
    // _ = filter_width;
    const half_width: i32 = filter_width >> 1;
    var pixel: f32 = 0.0;
    var r = -half_width;
    while (r <= half_width) : (r += 1) {
        const n_y = clampedOffset(id.y, r, num_rows);
        var c = -half_width;
        while (c <= half_width) : (c += 1) {
            const n_x = clampedOffset(id.x, c, num_cols);
            const weight = filter[@intCast(usize, (r + half_width) * filter_width + c + half_width)];
            // const weight: f32 = 0.1;
            pixel += weight * @intToFloat(f32, input[@intCast(usize, (n_y * num_cols + n_x) * 3 + id.z)]);
        }
    }
    output[channel_id] = @floatToInt(u8, pixel);
}

/// threadId.x
inline fn threadIdX() i32 {
    if (!is_nvptx) return 0;
    var tid = asm volatile ("mov.s32 \t$0, %tid.x;"
        : [ret] "=r" (-> i32),
    );
    return @intCast(i32, tid);
}
/// threadId.y
inline fn threadIdY() i32 {
    if (!is_nvptx) return 0;
    var tid = asm volatile ("mov.s32 \t$0, %tid.y;"
        : [ret] "=r" (-> i32),
    );
    return @intCast(i32, tid);
}
/// threadId.z
inline fn threadIdZ() i32 {
    if (!is_nvptx) return 0;
    var tid = asm volatile ("mov.s32 \t$0, %tid.z;"
        : [ret] "=r" (-> i32),
    );
    return @intCast(i32, tid);
}

/// threadDim.x
inline fn threadDimX() i32 {
    if (!is_nvptx) return 0;
    var ntid = asm volatile ("mov.s32 \t$0, %ntid.x;"
        : [ret] "=r" (-> i32),
    );
    return @intCast(i32, ntid);
}
/// threadDim.y
inline fn threadDimY() i32 {
    if (!is_nvptx) return 0;
    var ntid = asm volatile ("mov.s32 \t$0, %ntid.y;"
        : [ret] "=r" (-> i32),
    );
    return @intCast(i32, ntid);
}
/// threadDim.z
inline fn threadDimZ() i32 {
    if (!is_nvptx) return 0;
    var ntid = asm volatile ("mov.s32 \t$0, %ntid.z;"
        : [ret] "=r" (-> i32),
    );
    return @intCast(i32, ntid);
}

/// gridId.x
inline fn gridIdX() i32 {
    if (!is_nvptx) return 0;
    var ctaid = asm volatile ("mov.s32 \t$0, %ctaid.x;"
        : [ret] "=r" (-> i32),
    );
    return @intCast(i32, ctaid);
}
/// gridId.y
inline fn gridIdY() i32 {
    if (!is_nvptx) return 0;
    var ctaid = asm volatile ("mov.s32 \t$0, %ctaid.y;"
        : [ret] "=r" (-> i32),
    );
    return @intCast(i32, ctaid);
}
/// gridId.z
inline fn gridIdZ() i32 {
    if (!is_nvptx) return 0;
    var ctaid = asm volatile ("mov.s32 \t$0, %ctaid.z;"
        : [ret] "=r" (-> i32),
    );
    return @intCast(i32, ctaid);
}

/// gridDim.x
inline fn gridDimX() i32 {
    if (!is_nvptx) return 0;
    var nctaid = asm volatile ("mov.s32 \t$0, %nctaid.x;"
        : [ret] "=r" (-> i32),
    );
    return @intCast(i32, nctaid);
}
/// gridDim.y
inline fn gridDimY() i32 {
    if (!is_nvptx) return 0;
    var nctaid = asm volatile ("mov.s32 \t$0, %nctaid.y;"
        : [ret] "=r" (-> i32),
    );
    return @intCast(i32, nctaid);
}
/// gridDim.z
inline fn gridDimZ() i32 {
    if (!is_nvptx) return 0;
    var nctaid = asm volatile ("mov.s32 \t$0, %nctaid.z;"
        : [ret] "=r" (-> i32),
    );
    return @intCast(i32, nctaid);
}

inline fn getId_1D() i32 {
    return threadIdX() + threadDimX() * gridIdX();
}

const Dim2 = struct { x: i32, y: i32 };
pub fn getId_2D() Dim2 {
    return Dim2{
        .x = threadIdX() + threadDimX() * gridIdX(),
        .y = threadIdY() + threadDimY() * gridIdY(),
    };
}

const Dim3 = struct { x: i32, y: i32, z: i32 };
pub fn getId_3D() Dim3 {
    return Dim3{
        .x = threadIdX() + threadDimX() * gridIdX(),
        .y = threadIdY() + threadDimY() * gridIdY(),
        .z = threadIdZ() + threadDimZ() * gridIdZ(),
    };
}
