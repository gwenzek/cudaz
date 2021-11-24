const builtin = @import("builtin");
const std = @import("std");
const is_nvptx = builtin.cpu.arch == .nvptx64;
const CallingConvention = @import("std").builtin.CallingConvention;
const PtxKernel = if (is_nvptx) CallingConvention.PtxKernel else CallingConvention.Unspecified;

const cu = @import("cudaz").cu;

// stage2 seems unable to compile math.min, so we roll out our own clamp
fn clamp(x: i32, min: i32, max: i32) i32 {
    if (x < min) return min;
    if (x >= max) return max - 1;
    return x;
}

pub const Mat3 = struct {
    data: []const u8,
    shape: [3]i32,
    pub fn getClamped(self: Mat3, x: i32, y: i32, z: i32) u8 {
        return self.data[self.idxClamped(x, y, z)];
    }
    pub fn idx(self: Mat3, x: i32, y: i32, z: i32) usize {
        const i = (x * self.shape[1] + y) * self.shape[2] + z;
        return @intCast(usize, i);
    }

    pub fn idxClamped(self: Mat3, x: i32, y: i32, z: i32) usize {
        return self.idx(
            clamp(x, 0, self.shape[0]),
            clamp(y, 0, self.shape[1]),
            clamp(z, 0, self.shape[2]),
        );
    }
};

pub const Mat2Float = struct {
    data: []f32,
    shape: [2]i32,
    pub fn getClamped(self: Mat2Float, x: i32, y: i32) f32 {
        return self.data[self.idxClamped(x, y)];
    }
    pub fn idx(self: Mat2Float, x: i32, y: i32) usize {
        const i = x * self.shape[1] + y;
        return @intCast(usize, i);
    }

    pub fn idxClamped(self: Mat2Float, x: i32, y: i32) usize {
        return self.idx(
            clamp(x, 0, self.shape[0]),
            clamp(y, 0, self.shape[1]),
        );
    }
};
inline fn clampedOffset(x: i32, step: i32, n: i32) i32 {
    if (step < 0 and -step > x) return 0;
    const x_step = x + step;
    if (x_step >= n) return n - 1;
    return x_step;
}

pub const GaussianBlurArgs = struct {
    raw_input: []const u8,
    num_cols: i32,
    num_rows: i32,
    filter: []const f32,
    filter_width: i32,
    output: []u8,
};

pub export fn gaussianBlurStruct(args: GaussianBlurArgs) callconv(PtxKernel) void {
    return gaussianBlurVerbose(
        args.raw_input,
        args.num_cols,
        args.num_rows,
        args.filter,
        args.filter_width,
        args.output,
    );
}

pub export fn gaussianBlurVerbose(
    raw_input: []const u8,
    num_cols: i32,
    num_rows: i32,
    filter: []const f32,
    filter_width: i32,
    output: []u8,
) callconv(PtxKernel) void {
    const id = getId_3D();
    const input = Mat3{ .data = raw_input, .shape = [_]i32{ num_cols, num_rows, 3 } };
    if (id.x >= num_cols or id.y >= num_rows)
        return;

    const channel_id = @intCast(usize, input.idx(id.x, id.y, id.z));
    // output[channel_id] = input.data[channel_id];

    const half_width: i32 = filter_width >> 1;
    var pixel: f32 = 0.0;
    var r = -half_width;
    while (r <= half_width) : (r += 1) {
        var c = -half_width;
        while (c <= half_width) : (c += 1) {
            const weight = filter[@intCast(usize, (r + half_width) * filter_width + c + half_width)];
            pixel += weight * @intToFloat(f32, input.getClamped(id.x, id.y, id.z));
        }
    }
    output[channel_id] = @floatToInt(u8, pixel);
}

pub export fn gaussianBlur(
    input: Mat3,
    filter: Mat2Float,
    output: []u8,
) callconv(PtxKernel) void {
    const id = getId_3D();
    if (id.x >= input.shape[0] or id.y >= input.shape[1])
        return;
    const channel_id = @intCast(usize, input.idx(id.x, id.y, id.z));
    output[channel_id] = input.data[channel_id];

    const half_width: i32 = filter.shape[0] >> 1;
    var pixel: f32 = 0.0;
    var r = -half_width;
    while (r <= half_width) : (r += 1) {
        var c = -half_width;
        while (c <= half_width) : (c += 1) {
            const weight = filter.getClamped(r + half_width, c + half_width);
            pixel += weight * @intToFloat(f32, input.getClamped(id.x, id.y, id.z));
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
