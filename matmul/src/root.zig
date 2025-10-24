const std = @import("std");

const nvptx = @import("nvptx");

pub const Shape = extern struct {
    m: u32,
    n: u32,
    k: u32,
};

/// Matmul of A[m, k] and B[n, k], results are written to C[m, n]
pub fn matmul(
    A: []const f32,
    B: []const f32,
    shape: Shape,
    C: []f32,
) callconv(nvptx.kernel) void {
    const i = nvptx.getIdX();
    const j = nvptx.getIdY();

    if (i >= shape.m or j >= shape.m) return;

    var c_i_j: f32 = 0.0;
    const a_i = A[i * shape.k ..][0..shape.k];
    const b_j = B[j * shape.k ..][0..shape.k];
    for (a_i, b_j) |a, b| {
        c_i_j += a * b;
    }

    C[i * shape.n + j] = c_i_j;
}

comptime {
    if (nvptx.is_nvptx) {
        @export(&matmul, .{ .name = "matmul" });
    }
}
