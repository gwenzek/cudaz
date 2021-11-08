const std = @import("std");
const log = std.log;
const math = std.math;
const testing = std.testing;
const assert = std.debug.assert;

const zigimg = @import("zigimg");

const cuda = @import("cudaz");
const cu = cuda.cu;

const png = @import("png.zig");
const utils = @import("utils.zig");

const resources_dir = "CS344/hw4_resources/";

pub fn main() !void {
    var general_purpose_allocator = std.heap.GeneralPurposeAllocator(.{}){};
    const allocator = &general_purpose_allocator.allocator;
    log.info("***** HW4 ******", .{});

    const img = try zigimg.Image.fromFilePath(allocator, resources_dir ++ "/red_eye_effect.png");
    const num_rows = img.height;
    const num_cols = img.width;
    const template = try zigimg.Image.fromFilePath(allocator, resources_dir ++ "/red_eye_effect_template.png");
    const num_rowsTemplate = template.height;
    const num_colsTemplate = template.width;
    img.deinit();

    log.info("loaded image", .{});
    var stream = try cuda.Stream.init(0);
    defer stream.deinit();
    const d_img = try cuda.allocAndCopy(cu.uchar3, utils.asUchar3(img));
    const d_template = try cuda.allocAndCopy(cu.uchar3, utils.asUchar3(template));
    const d_scores = try cuda.alloc(f32, num_rows * num_cols);

    // Create a 2D grid for the image and use the last dimension for the channel (R, G, B)
    const gridWithChannel = cuda.Grid.init3D(num_cols, num_rows, 3, 32, 8, 3);
    const crossCorrelation = try cuda.Function("naive_normalized_cross_correlation").init();
    try crossCorrelation.launch(&stream, gridWithChannel, .{
        d_scores.ptr,
        d_img.ptr,
        d_template.ptr,
        @intCast(c_int, num_rows),
        @intCast(c_int, num_cols),
        @intCast(c_int, num_rowsTemplate),
        @intCast(c_int, @divFloor(num_rowsTemplate, 2)),
        @intCast(c_int, num_colsTemplate),
        @intCast(c_int, @divFloor(num_colsTemplate, 2)),
        @intCast(c_int, num_rowsTemplate * num_colsTemplate),
    });

    // TODO: d_scores += min(d_scores)

    var d_sorted_scores = try cuda.alloc(f32, d_scores.len);
    defer cuda.free(d_sorted_scores);
    var timer = cuda.GpuTimer.init(&stream);
    timer.start();

    var d_permutation = try mySort(&stream, d_sorted_scores, d_scores);
    defer cuda.free(d_permutation);
    timer.stop();

    try stream.synchronize();
    std.log.info("Your code ran in: {d:.1} msecs.", .{timer.elapsed() * 1000});

    const remove_redness = try cuda.Function("remove_redness").init();
    try remove_redness.launch(&stream, cuda.Grid.init1D(d_img.len, 64), .{
        d_permutation.ptr,
        d_img.ptr,
        40,
        @intCast(c_int, num_rows),
        @intCast(c_int, num_cols),
        @intCast(c_int, num_rowsTemplate),
        @intCast(c_int, num_colsTemplate),
    });

    try cuda.memcpyDtoH(cu.uchar3, utils.asUchar3(img), d_img);
    try png.writePngToFilePath(img, resources_dir ++ "output.png");
    try utils.validate_output(allocator, resources_dir, 2.0);
}

/// Red Eye Removal
/// ===============
///
/// For this assignment we are implementing red eye removal.  This is
/// accomplished by first creating a score for every pixel that tells us how
/// likely it is to be a red eye pixel.  We have already done this for you - you
/// are receiving the scores and need to sort them in ascending order so that we
/// know which pixels to alter to remove the red eye.
///
/// Note: ascending order == smallest to largest
///
/// Each score is associated with a position, when you sort the scores, you must
/// also move the positions accordingly.
///
/// Implementing Parallel Radix Sort with CUDA
/// ==========================================
///
/// The basic idea is to construct a histogram on each pass of how many of each
/// "digit" there are.   Then we scan this histogram so that we know where to put
/// the output of each digit.  For example, the first 1 must come after all the
/// 0s so we have to know how many 0s there are to be able to start moving 1s
/// into the correct position.
///
/// 1) Histogram of the number of occurrences of each digit
/// 2) Exclusive Prefix Sum of Histogram
/// 3) Determine relative offset of each digit
///      For example [0 0 1 1 0 0 1]
///              ->  [0 1 0 1 2 3 2]
/// 4) Combine the results of steps 2 & 3 to determine the final
///    output location for each element and move it there
///
/// LSB Radix sort is an out-of-place sort and you will need to ping-pong values
/// between the input and output buffers we have provided.  Make sure the final
/// sorted results end up in the output buffer!  Hint: You may need to do a copy
/// at the end.
pub fn mySort(stream: *const cuda.Stream, d_out_vals: []f32, d_in_vals: []const f32) ![]c_uint {
    var coords = try range(stream, d_in_vals.len);
    defer cuda.free(coords);
    var d_permutation = try cuda.alloc(c_uint, coords.len);
    try cuda.memset(c_uint, d_permutation, 0);
    // const radix_cdf = cuda.Function("radix_cdf").init();
    // TODO
    _ = d_out_vals;
    // Sorting is done in several stages
    // Sorting of small arrays is done with Radix/sort_network
    // Then we can merge using parallel merging
    return d_permutation;
}

pub fn range(stream: *const cuda.Stream, len: usize) ![]c_uint {
    var coords = try cuda.alloc(c_uint, len);
    const rangeFn = try cuda.Function("range").init();
    try rangeFn.launch(stream, cuda.Grid.init1D(len, 64), .{ coords.ptr, @intCast(c_uint, len) });
    return coords;
}

/// Finds the minimum value of the given input slice.
/// Do several passes until the minimum is found.
/// Each block computes the minimum over 1024 elements.
/// Each pass divides the size of the input array per 1024.
pub fn reduce_min(stream: *const cuda.Stream, d_data: []const f32) !f32 {
    const n_threads = 1024;
    const n1 = math.divCeil(usize, d_data.len, n_threads) catch unreachable;
    var n2 = math.divCeil(usize, n1, n_threads) catch unreachable;
    const buffer = try cuda.alloc(f32, n1 + n2);
    defer cuda.free(buffer);

    var d_in = d_data;
    var d_out = buffer[0..n1];
    var d_next = buffer[n1 .. n1 + n2];
    const reduce = try cuda.Function("reduce_min").init();

    while (d_in.len > 1) {
        try reduce.launchWithSharedMem(
            stream,
            cuda.Grid.init1D(d_in.len, n_threads),
            n_threads * @sizeOf(f32),
            .{ d_in.ptr, d_out.ptr, @intCast(c_int, d_in.len) },
        );
        d_in = d_out;
        d_out = d_next;
        n2 = math.divCeil(usize, d_next.len, n_threads) catch unreachable;
        d_next = d_out[0..n2];
    }

    return try cuda.readResult(f32, &d_in[0]);
}

test "reduce min" {
    var stream = try cuda.Stream.init(0);
    defer stream.deinit();
    const h_x = try testing.allocator.alloc(f32, 2100);
    defer testing.allocator.free(h_x);
    std.mem.set(f32, h_x, 0.0);
    h_x[987] = -5.0;
    h_x[1024] = -6.0;
    h_x[1479] = -7.0;

    const d_x = try cuda.allocAndCopy(f32, h_x);
    try testing.expectEqual(try reduce_min(&stream, d_x), -7.0);
}

pub fn sort_network(stream: *const cuda.Stream, d_data: []f32, n_threads: usize) !void {
    const sort_net = try cuda.Function("sort_network").init();
    const grid = cuda.Grid.init1D(d_data.len, n_threads);
    try sort_net.launchWithSharedMem(
        stream,
        grid,
        grid.threads.x * @sizeOf(f32),
        .{ d_data.ptr, @intCast(c_uint, d_data.len) },
    );
}

test "sorting network" {
    var stream = try cuda.Stream.init(0);
    defer stream.deinit();
    const h_x = [_]f32{ 2, 3, 1, 0, 7, 9, 6, 5 };
    var h_out = [_]f32{0} ** h_x.len;
    const d_x = try cuda.alloc(f32, h_x.len);
    defer cuda.free(d_x);

    try cuda.memcpyHtoD(f32, d_x, &h_x);
    try sort_network(&stream, d_x, 2);
    try cuda.memcpyDtoH(f32, &h_out, d_x);
    try testing.expectEqual([_]f32{ 2, 3, 0, 1, 7, 9, 5, 6 }, h_out);

    try cuda.memcpyHtoD(f32, d_x, &h_x);
    try sort_network(&stream, d_x, 4);
    try cuda.memcpyDtoH(f32, &h_out, d_x);
    try testing.expectEqual([_]f32{ 0, 1, 2, 3, 5, 6, 7, 9 }, h_out);

    try cuda.memcpyHtoD(f32, d_x, &h_x);
    try sort_network(&stream, d_x, 8);
    try cuda.memcpyDtoH(f32, &h_out, d_x);
    try testing.expectEqual([_]f32{ 0, 1, 2, 3, 5, 6, 7, 9 }, h_out);

    try cuda.memcpyHtoD(f32, d_x, &h_x);
    try sort_network(&stream, d_x, 16);
    try cuda.memcpyDtoH(f32, &h_out, d_x);
    try testing.expectEqual([_]f32{ 0, 1, 2, 3, 5, 6, 7, 9 }, h_out);
}
