const std = @import("std");

const ptx = @import("nvptx");

fn clamp(x: i32, min: i32, max: i32) i32 {
    if (x < min) return min;
    if (x >= max) return max - 1;
    return x;
}

pub const Mat3 = struct {
    data: [*]const u8,
    shape: [3]u32,

    pub fn getClamped(self: Mat3, x: i32, y: i32, z: i32) u8 {
        return self.data[self.idxClamped(x, y, z)];
    }

    pub fn idx(self: Mat3, x: usize, y: usize, z: usize) usize {
        const i = (x * self.shape[1] + y) * self.shape[2] + z;
        return @intCast(i);
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
    data: [*]f32,
    shape: [2]i32,
    pub fn getClamped(self: Mat2Float, x: i32, y: i32) f32 {
        return self.data[self.idxClamped(x, y)];
    }
    pub fn idx(self: Mat2Float, x: i32, y: i32) usize {
        const i = x * self.shape[1] + y;
        return @intCast(i);
    }

    pub fn idxClamped(self: Mat2Float, x: i32, y: i32) usize {
        return self.idx(
            clamp(x, 0, self.shape[0]),
            clamp(y, 0, self.shape[1]),
        );
    }
};
inline fn clampedOffset(x: usize, step: i32, n: usize) usize {
    var x_step = x;
    if (step < 0) {
        const abs_step = @abs(step);
        if (abs_step > x) return 0;
        x_step -= abs_step;
    }

    if (x_step >= n) return n - 1;
    return x_step;
}

pub const GaussianBlurArgs = struct {
    img: Mat3,
    filter: []const f32,
    filter_width: u32,
    output: [*]u8,
};

pub fn gaussianBlurStruct(args: GaussianBlurArgs) callconv(ptx.kernel) void {
    return gaussianBlurImpl(
        args.img.data,
        args.img.shape[0],
        args.img.shape[1],
        args.filter.ptr,
        args.filter_width,
        args.output,
    );
}

pub fn gaussianBlurVerbose(
    raw_input: [*]const u8,
    num_cols: u32,
    num_rows: u32,
    filter: [*]const f32,
    filter_width: u32,
    output: [*]u8,
) callconv(ptx.kernel) void {
    return gaussianBlurImpl(
        raw_input,
        num_cols,
        num_rows,
        filter,
        filter_width,
        output,
    );
}

fn gaussianBlurImpl(
    raw_input: [*]const u8,
    num_cols: u32,
    num_rows: u32,
    filter: [*]const f32,
    filter_width: u32,
    output: [*]u8,
) void {
    const id = ptx.getId_3D();
    const input = Mat3{ .data = raw_input, .shape = [_]u32{ num_cols, num_rows, 3 } };
    if (id.x >= num_cols or id.y >= num_rows)
        return;

    const channel_id: usize = @intCast(input.idx(id.x, id.y, id.z));

    var pixel: f32 = 0.0;
    const half_width: i32 = @intCast(filter_width >> 1);
    var r = -half_width;
    while (r <= half_width) : (r += 1) {
        var c = -half_width;
        while (c <= half_width) : (c += 1) {
            const weight = filter[@intCast((r + half_width) * @as(i32, @intCast(filter_width)) + c + half_width)];
            const neighbor: f32 = @floatFromInt(input.idx(
                clampedOffset(id.x, c, input.shape[0]),
                clampedOffset(id.y, r, input.shape[1]),
                @as(usize, id.z),
            ));
            pixel += neighbor * weight;
        }
    }
    output[channel_id] = @intFromFloat(pixel);
}

pub fn gaussianBlur(
    input: Mat3,
    filter: Mat2Float,
    output: []u8,
) callconv(ptx.kernel) void {
    const id = ptx.getId_3D();
    if (id.x >= input.shape[0] or id.y >= input.shape[1])
        return;
    const channel_id: usize = input.idx(id.x, id.y, id.z);

    const half_width: i32 = filter.shape[0] >> 1;
    var pixel: f32 = 0.0;
    var r = -half_width;
    while (r <= half_width) : (r += 1) {
        var c = -half_width;
        while (c <= half_width) : (c += 1) {
            const weight = filter.getClamped(r + half_width, c + half_width);
            const neighbor: f32 = @floatFromInt(input.idx(
                clampedOffset(id.x, c, input.shape[0]),
                clampedOffset(id.y, r, input.shape[1]),
                @as(usize, id.z),
            ));
            pixel += weight * neighbor;
        }
    }
    output[channel_id] = @intFromFloat(pixel);
}

comptime {
    if (ptx.is_nvptx) {
        @export(&gaussianBlur, .{ .name = "gaussianBlur" });
        @export(&gaussianBlurStruct, .{ .name = "gaussianBlurStruct" });
        @export(&gaussianBlurVerbose, .{ .name = "gaussianBlurVerbose" });
    }
}
