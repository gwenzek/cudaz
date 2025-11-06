const std = @import("std");

const ptx = @import("nvptx");
pub const panic = ptx.panic;

fn clamp(x: i32, min: u32, max: u32) u32 {
    if (x < 0) return min;
    const x_pos = @abs(x);
    if (x_pos < min) return min;
    if (x_pos >= max) return max - 1;
    return x_pos;
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
            clamp(x, 0, @intCast(self.shape[0])),
            clamp(y, 0, @intCast(self.shape[1])),
            clamp(z, 0, @intCast(self.shape[2])),
        );
    }
};

pub const Mat2Float = struct {
    data: [*]f32,
    shape: [2]u32,
    pub fn getClamped(self: Mat2Float, x: i32, y: i32) f32 {
        return self.data[self.idxClamped(x, y)];
    }
    pub fn idx(self: Mat2Float, x: u32, y: u32) usize {
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

fn gaussianBlurImpl(
    input: Mat3,
    filter: Mat2Float,
    output: []u8,
) void {
    const id = ptx.getId_3D();
    if (id.x >= input.shape[0] or id.y >= input.shape[1])
        return;
    const channel_id: usize = input.idx(id.x, id.y, id.z);

    const id_x: i32 = @intCast(id.x);
    const id_y: i32 = @intCast(id.y);
    const id_z: i32 = @intCast(id.z);

    const half_width: i32 = @intCast(filter.shape[0] >> 1);
    var pixel: f32 = 0.0;
    var r = -half_width;
    while (r <= half_width) : (r += 1) {
        var c = -half_width;
        while (c <= half_width) : (c += 1) {
            const weight = filter.getClamped(r + half_width, c + half_width);
            const neighbor: f32 = @floatFromInt(input.getClamped(id_x + c, id_y + r, id_z));
            pixel += weight * neighbor;
        }
    }
    output[channel_id] = @intFromFloat(pixel);
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
        Mat3{ .data = raw_input, .shape = .{ num_cols, num_rows, 3 } },

        Mat2Float{ .data = @constCast(filter), .shape = .{ filter_width, filter_width } },
        output[0 .. num_cols * num_rows * 3],
    );
}

pub fn gaussianBlur(
    input: Mat3,
    filter: Mat2Float,
    output: []u8,
) callconv(ptx.kernel) void {
    gaussianBlurImpl(input, filter, output);
}

comptime {
    if (ptx.is_nvptx) {
        // Export two versions of the same kernel.
        // Showcasing that you can pass complex Zig struct to kernels.
        @export(&gaussianBlur, .{ .name = "gaussianBlur" });
        @export(&gaussianBlurVerbose, .{ .name = "gaussianBlurVerbose" });
    }
}
