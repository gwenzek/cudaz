const std = @import("std");
const math = std.math;

const cuda = @import("cuda.zig");

/// Wrapper algorithm for a reduce kernel like min, max, sum, ...
/// Those kernels compute the reduce operator only for a block of data.
/// We need several passes for the global minimun/maximum/sum.
/// The block size is hardcoded to 1024 which is the max number of threads per block
/// on modern NVidia GPUs.
// TODO: copy tests from HW4
pub fn reduce(
    stream: *const cuda.Stream,
    operator: anytype,
    d_data: anytype,
) !std.meta.Elem(@TypeOf(d_data)) {
    const n_threads = 1024;
    const n1 = math.divCeil(usize, d_data.len, n_threads) catch unreachable;
    var n2 = math.divCeil(usize, n1, n_threads) catch unreachable;
    const DType = std.meta.Elem(@TypeOf(d_data));
    const buffer = try stream.alloc(DType, n1 + n2);
    defer stream.free(buffer);

    var d_in = d_data;
    var d_out = buffer[0..n1];
    var d_next = buffer[n1 .. n1 + n2];

    while (d_in.len > 1) {
        try operator.launchWithSharedMem(
            stream,
            cuda.Grid.init1D(d_in.len, n_threads),
            n_threads * @sizeOf(DType),
            .{ d_in.ptr, d_out.ptr, @intCast(c_int, d_in.len) },
        );
        d_in = d_out;
        d_out = d_next;
        n2 = math.divCeil(usize, d_next.len, n_threads) catch unreachable;
        d_next = d_out[0..n2];
    }
    return try cuda.readResult(DType, &d_in[0]);
}
