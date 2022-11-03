//! Red Eye Removal
//! ===============
//!
//! For this assignment we are implementing red eye removal.  This i
//! accomplished by first creating a score for every pixel that tells us how
//! likely it is to be a red eye pixel.  We have already done this for you - you
//! are receiving the scores and need to sort them in ascending order so that we
//! know which pixels to alter to remove the red eye.
//!
//! Note: ascending order == smallest to largest
//!
//! Each score is associated with a position, when you sort the scores, you must
//! also move the positions accordingly.
//!
//! Implementing Parallel Radix Sort with CUDA
//! ==========================================
//!
//! The basic idea is to construct a histogram on each pass of how many of each
//! "digit" there are.   Then we scan this histogram so that we know where to put
//! the output of each digit.  For example, the first 1 must come after all the
//! 0s so we have to know how many 0s there are to be able to start moving 1s
//! into the correct position.
//!
//! 1) Histogram of the number of occurrences of each digit
//! 2) Exclusive Prefix Sum of Histogram
//! 3) Determine relative offset of each digit
//!      For example [0 0 1 1 0 0 1]
//!              ->  [0 1 0 1 2 3 2]
//! 4) Combine the results of steps 2 & 3 to determine the final
//!    output location for each element and move it there
//!
//! LSB Radix sort is an out-of-place sort and you will need to ping-pong values
//! between the input and output buffers we have provided.  Make sure the final
//! sorted results end up in the output buffer!  Hint: You may need to do a copy
//! at the end.

const std = @import("std");
const math = std.math;
const testing = std.testing;
const assert = std.debug.assert;

const cuda = @import("cudaz");
const cu = cuda.cu;

const png = @import("png.zig");
const utils = @import("utils.zig");

const resources_dir = "resources/hw4/";

const log = std.log.scoped(.HW4);
const log_level = std.log.Level.warn;

pub fn main() !void {
    try hw4();
}

pub fn hw4() !void {
    var general_purpose_allocator = std.heap.GeneralPurposeAllocator(.{}){};
    const allocator = general_purpose_allocator.allocator();
    log.info("***** HW4 ******", .{});

    const img = try png.Image.fromFilePath(allocator, resources_dir ++ "/red_eye_effect.png");
    const num_rows = img.height;
    const num_cols = img.width;
    const template = try png.Image.fromFilePath(allocator, resources_dir ++ "/red_eye_effect_template.png");
    const num_rows_template = template.height;
    const num_cols_template = template.width;
    defer img.deinit();

    var stream = initStreamWithModule(0);
    defer stream.deinit();
    const d_img = try cuda.allocAndCopy(cu.uchar3, utils.asUchar3(img));
    const d_template = try cuda.allocAndCopy(cu.uchar3, utils.asUchar3(template));
    log.info("Loaded image: {}x{}", .{ num_rows, num_cols });
    log.info("Loaded template: {}x{}", .{ num_rows_template, num_cols_template });
    const d_scores = try cuda.alloc(f32, num_rows * num_cols);

    try test_naive_normalized_cross_correlation(&stream, d_template, num_rows_template, num_cols_template);
    // Create a 2D grid for the image and use the last dimension for the channel (R, G, B)
    const gridWithChannel = cuda.Grid.init3D(num_cols, num_rows, 3, 32, 1, 3);
    log.info("crossCorrelation({}, {},)", .{ gridWithChannel, gridWithChannel.sharedMem(f32, 1) });
    try k.crossCorrelation.launchWithSharedMem(
        &stream,
        gridWithChannel,
        gridWithChannel.sharedMem(f32, 1),
        .{
            d_scores.ptr,
            d_img.ptr,
            d_template.ptr,
            @intCast(c_int, num_rows),
            @intCast(c_int, num_cols),
            @intCast(c_int, num_rows_template),
            @intCast(c_int, @divFloor(num_rows_template, 2)),
            @intCast(c_int, num_cols_template),
            @intCast(c_int, @divFloor(num_cols_template, 2)),
            @intCast(c_int, num_rows_template * num_cols_template),
        },
    );

    const min_corr = try reduceMin(&stream, d_scores);
    try k.addConstant.launch(
        &stream,
        cuda.Grid.init1D(d_scores.len, 32),
        .{ d_scores.ptr, -min_corr, @intCast(c_uint, d_scores.len) },
    );
    log.info("min_corr = {}", .{min_corr});
    debugDevice(allocator, "crossCorrelation", d_scores[0..100]);

    var timer = cuda.GpuTimer.start(&stream);
    var d_permutation = try radixSortAlloc(&stream, bitCastU32(d_scores));
    defer cuda.free(d_permutation);
    timer.stop();

    const d_perm_min = try reduce(&stream, k.minU32, d_permutation);
    const d_perm_max = try reduce(&stream, k.maxU32, d_permutation);
    log.info("Permutation ranges from {} to {} (expected 0 to {})", .{ d_perm_min, d_perm_max, d_permutation.len - 1 });

    stream.synchronize();
    std.log.info("Your code ran in: {d:.1} msecs.", .{timer.elapsed() * 1000});

    var d_out = try cuda.alloc(cu.uchar3, d_img.len);
    debugDevice(allocator, "d_perm", d_permutation[20000..21000]);
    try k.removeRedness.launch(&stream, cuda.Grid.init1D(d_img.len, 64), .{
        d_permutation.ptr,
        d_img.ptr,
        d_out.ptr,
        30,
        @intCast(c_int, num_rows),
        @intCast(c_int, num_cols),
        9, // We use something smaller than the full template
        9, // to only edit the pupil and not the rest of the eye
    });

    try cuda.memcpyDtoH(cu.uchar3, utils.asUchar3(img), d_out);
    try img.writeToFilePath(resources_dir ++ "output.png");
    try utils.validate_output(allocator, resources_dir, 0.1);
}

pub fn range(stream: *const cuda.Stream, len: usize) ![]u32 {
    var coords = try cuda.alloc(c_uint, len);
    try k.rangeFn.launch(stream, cuda.Grid.init1D(len, 64), .{ coords.ptr, @intCast(c_uint, len) });
    return coords;
}

test "range" {
    var stream = initStreamWithModule(0);
    defer stream.deinit();
    var numbers = try range(&stream, 5);
    try expectEqualDeviceSlices(u32, &[_]u32{ 0, 1, 2, 3, 4 }, numbers);
}

pub fn reduceMin(stream: *const cuda.Stream, d_data: []const f32) !f32 {
    return reduce(stream, k.min, d_data);
}

/// Finds the minimum value of the given input slice.
/// Do several passes until the minimum is found.
/// Each block computes the minimum over 1024 elements.
/// Each pass divides the size of the input array per 1024.
pub fn reduce(stream: *const cuda.Stream, operator: anytype, d_data: anytype) !std.meta.Elem(@TypeOf(d_data)) {
    const n_threads = 1024;
    const n1 = math.divCeil(usize, d_data.len, n_threads) catch unreachable;
    var n2 = math.divCeil(usize, n1, n_threads) catch unreachable;
    const DType = std.meta.Elem(@TypeOf(d_data));
    const buffer = try cuda.alloc(DType, n1 + n2);
    defer cuda.free(buffer);

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
    return stream.copyResult(DType, &d_in[0]);
}

test "reduce min" {
    var stream = initStreamWithModule(0);
    defer stream.deinit();
    const h_x = try testing.allocator.alloc(f32, 2100);
    defer testing.allocator.free(h_x);
    std.mem.set(f32, h_x, 0.0);
    h_x[987] = -5.0;
    h_x[1024] = -6.0;
    h_x[1479] = -7.0;

    const d_x = try cuda.allocAndCopy(f32, h_x);
    try testing.expectEqual(try reduceMin(&stream, d_x), -7.0);
}

test "reduce sum" {
    var stream = initStreamWithModule(0);
    defer stream.deinit();

    const d1 = try cuda.allocAndCopy(u32, &[_]u32{ 1, 4, 8, 0, 1 });
    try testing.expectEqual(@intCast(u32, 14), try reduce(&stream, k.sumU32, d1));

    const h_x = try testing.allocator.alloc(u32, 2100);
    defer testing.allocator.free(h_x);
    std.mem.set(u32, h_x, 0.0);
    h_x[987] = 5;
    h_x[1024] = 6;
    h_x[1479] = 7;
    h_x[14] = 42;

    const d_x = try cuda.allocAndCopy(u32, h_x);
    try testing.expectEqual(@intCast(u32, 60), try reduce(&stream, k.sumU32, d_x));
}

pub fn sortNetwork(stream: *const cuda.Stream, d_data: []f32, n_threads: usize) !void {
    const grid = cuda.Grid.init1D(d_data.len, n_threads);
    try k.sortNet.launchWithSharedMem(
        stream,
        grid,
        grid.threads.x * @sizeOf(f32),
        .{ d_data.ptr, @intCast(c_uint, d_data.len) },
    );
}

test "sorting network" {
    var stream = initStreamWithModule(0);
    defer stream.deinit();
    const h_x = [_]f32{ 2, 3, 1, 0, 7, 9, 6, 5 };
    var h_out = [_]f32{0} ** h_x.len;
    const d_x = try cuda.alloc(f32, h_x.len);
    defer cuda.free(d_x);

    try cuda.memcpyHtoD(f32, d_x, &h_x);
    try sortNetwork(&stream, d_x, 2);
    try cuda.memcpyDtoH(f32, &h_out, d_x);
    try testing.expectEqual([_]f32{ 2, 3, 0, 1, 7, 9, 5, 6 }, h_out);

    try cuda.memcpyHtoD(f32, d_x, &h_x);
    try sortNetwork(&stream, d_x, 4);
    try cuda.memcpyDtoH(f32, &h_out, d_x);
    try testing.expectEqual([_]f32{ 0, 1, 2, 3, 5, 6, 7, 9 }, h_out);

    try cuda.memcpyHtoD(f32, d_x, &h_x);
    try sortNetwork(&stream, d_x, 8);
    try cuda.memcpyDtoH(f32, &h_out, d_x);
    try testing.expectEqual([_]f32{ 0, 1, 2, 3, 5, 6, 7, 9 }, h_out);

    try cuda.memcpyHtoD(f32, d_x, &h_x);
    try sortNetwork(&stream, d_x, 16);
    try cuda.memcpyDtoH(f32, &h_out, d_x);
    try testing.expectEqual([_]f32{ 0, 1, 2, 3, 5, 6, 7, 9 }, h_out);
}

pub fn inPlaceCdf(stream: *const cuda.Stream, d_values: []u32, n_threads: u32) cuda.Error!void {
    const n = d_values.len;
    const grid_N = cuda.Grid.init1D(n, n_threads);
    const n_blocks = grid_N.blocks.x;
    var d_grid_bins = try cuda.alloc(u32, n_blocks);
    defer cuda.free(d_grid_bins);
    log.debug("cdf(n={}, n_threads={}, n_blocks={})", .{ n, n_threads, n_blocks });
    var n_threads_pow_2 = n_threads;
    while (n_threads_pow_2 > 1) {
        std.debug.assert(n_threads_pow_2 % 2 == 0);
        n_threads_pow_2 /= 2;
    }
    try k.cdfIncremental.launchWithSharedMem(
        stream,
        grid_N,
        n_threads * @sizeOf(u32),
        .{ d_values.ptr, d_grid_bins.ptr, @intCast(c_int, n) },
    );
    var d_cdf_min = try reduce(stream, k.minU32, d_values);
    var d_cdf_max = try reduce(stream, k.maxU32, d_values);
    log.info("Cdf ranges from {} to {}", .{ d_cdf_min, d_cdf_max });
    if (n_blocks == 1) return;

    // log.debug("cdf_shift({}, {})", .{ n, N });
    try inPlaceCdf(stream, d_grid_bins, n_threads);
    try k.cdfShift.launch(
        stream,
        grid_N,
        .{ d_values.ptr, d_grid_bins.ptr, @intCast(c_int, n) },
    );
    d_cdf_min = try reduce(stream, k.minU32, d_values);
    d_cdf_max = try reduce(stream, k.maxU32, d_values);
    log.info("After shift cdf ranges from {} to {}", .{ d_cdf_min, d_cdf_max });
}

test "inPlaceCdf" {
    var stream = initStreamWithModule(0);
    defer stream.deinit();
    const h_x = [_]u32{ 0, 2, 1, 1, 0, 1, 3, 0, 2 };
    var h_out = [_]u32{0} ** h_x.len;
    const h_cdf = [_]u32{ 0, 0, 2, 3, 4, 4, 5, 8, 8 };
    const d_x = try cuda.alloc(u32, h_x.len);
    defer cuda.free(d_x);

    try cuda.memcpyHtoD(u32, d_x, &h_x);
    try inPlaceCdf(&stream, d_x, 16);
    try cuda.memcpyDtoH(u32, &h_out, d_x);
    try testing.expectEqual(h_cdf, h_out);

    try cuda.memcpyHtoD(u32, d_x, &h_x);
    try inPlaceCdf(&stream, d_x, 8);
    try cuda.memcpyDtoH(u32, &h_out, d_x);
    try testing.expectEqual(h_cdf, h_out);

    // Try with smaller batch sizes, forcing several passes
    try cuda.memcpyHtoD(u32, d_x, &h_x);
    try inPlaceCdf(&stream, d_x, 4);
    try cuda.memcpyDtoH(u32, &h_out, d_x);
    try testing.expectEqual(h_cdf, h_out);

    try cuda.memcpyHtoD(u32, d_x, &h_x);
    try inPlaceCdf(&stream, d_x, 2);
    try expectEqualDeviceSlices(u32, &h_cdf, d_x);
}

pub fn radixSortAlloc(stream: *const cuda.Stream, d_values: []const u32) ![]u32 {
    const n = d_values.len;
    const mask: u8 = 0b1111;
    const mask_bits: u8 = 8 - @clz(mask);

    const d_radix = try cuda.alloc(u32, n * (mask + 1));
    defer cuda.free(d_radix);
    var d_perm0 = try range(stream, n);
    errdefer cuda.free(d_perm0);
    var d_perm1 = try cuda.alloc(u32, n);
    errdefer cuda.free(d_perm1);

    // Unroll the loop at compile time.
    comptime var shift: u8 = 0;
    inline while (shift < 32) {
        try _radixSort(stream, d_values, d_perm0, d_perm1, d_radix, shift, mask);
        shift += mask_bits;
        if (shift >= 32) {
            cuda.free(d_perm0);
            return d_perm1;
        }
        try _radixSort(stream, d_values, d_perm1, d_perm0, d_radix, shift, mask);
        shift += mask_bits;
    }
    cuda.free(d_perm1);
    return d_perm0;
}

fn _radixSort(
    stream: *const cuda.Stream,
    d_values: []const u32,
    d_perm0: []const u32,
    d_perm1: []u32,
    d_radix: []u32,
    shift: u8,
    mask: u8,
) !void {
    const n = d_values.len;
    try cuda.memset(u32, d_radix, 0);
    // debugDevice(allocator, "d_values", d_values);
    // debugDevice(allocator, "d_perm0", d_perm0);
    log.debug("radixSort(n={}, shift={}, mask={})", .{ n, shift, mask });
    try k.findRadixSplitted.launch(
        stream,
        cuda.Grid.init1D(n, 1024),
        .{ d_radix.ptr, d_values.ptr, d_perm0.ptr, shift, mask, @intCast(c_int, n) },
    );
    const radix_sum = try reduce(stream, k.sumU32, d_radix);
    log.debug("Radix sums to {}, expected {}", .{ radix_sum, d_values.len });
    std.debug.assert(radix_sum == d_values.len);
    try inPlaceCdf(stream, d_radix, 1024);
    // debugDevice(allocator, "d_radix + cdf", d_radix);
    try k.updatePermutation.launch(
        stream,
        cuda.Grid.init1D(n, 1024),
        .{ d_perm1.ptr, d_radix.ptr, d_values.ptr, d_perm0.ptr, shift, mask, @intCast(c_int, n) },
    );
    // debugDevice(allocator, "d_perm1", d_perm1);
}

test "findRadixSplitted" {
    var stream = initStreamWithModule(0);
    defer stream.deinit();
    const h_x0 = [_]u32{ 0b10, 0b01, 0b00 };
    const n = h_x0.len;
    const d_values = try cuda.allocAndCopy(u32, &h_x0);
    defer cuda.free(d_values);
    const d_radix = try cuda.alloc(u32, 2 * n);
    defer cuda.free(d_radix);
    try cuda.memset(u32, d_radix, 0);
    const d_perm0 = try range(&stream, n);
    defer cuda.free(d_perm0);

    try k.findRadixSplitted.launch(
        &stream,
        cuda.Grid.init1D(n, 1024),
        .{ d_radix.ptr, d_values.ptr, d_perm0.ptr, 0, @intCast(c_uint, 0b1), @intCast(c_int, n) },
    );
    try expectEqualDeviceSlices(u32, &[_]u32{ 1, 0, 1, 0, 1, 0 }, d_radix);

    try cuda.memset(u32, d_radix, 0);
    try k.findRadixSplitted.launch(
        &stream,
        cuda.Grid.init1D(n, 1024),
        .{ d_radix.ptr, d_values.ptr, d_perm0.ptr, 1, @intCast(c_uint, 0b1), @intCast(c_int, n) },
    );
    try expectEqualDeviceSlices(u32, &[_]u32{ 0, 1, 1, 1, 0, 0 }, d_radix);

    try cuda.memcpyHtoD(u32, d_perm0, &[_]u32{ 0, 2, 1 });
    // values: { 0b10, 0b01, 0b00 }; perm: {0, 2, 1}
    try cuda.memset(u32, d_radix, 0);
    try k.findRadixSplitted.launch(
        &stream,
        cuda.Grid.init1D(n, 1024),
        .{ d_radix.ptr, d_values.ptr, d_perm0.ptr, 0, @intCast(c_uint, 0b1), @intCast(c_int, n) },
    );
    try expectEqualDeviceSlices(u32, &[_]u32{ 1, 1, 0, 0, 0, 1 }, d_radix);

    try cuda.memset(u32, d_radix, 0);
    try k.findRadixSplitted.launch(
        &stream,
        cuda.Grid.init1D(n, 1024),
        .{ d_radix.ptr, d_values.ptr, d_perm0.ptr, 1, @intCast(c_uint, 0b1), @intCast(c_int, n) },
    );
    try expectEqualDeviceSlices(u32, &[_]u32{ 0, 1, 1, 1, 0, 0 }, d_radix);
}

test "_radixSort" {
    var stream = initStreamWithModule(0);
    defer stream.deinit();
    testing.log_level = std.log.Level.debug;
    defer testing.log_level = std.log.Level.warn;
    const h_x0 = [_]u32{ 0b10, 0b01, 0b00 };
    const n = h_x0.len;
    const d_values = try cuda.allocAndCopy(u32, &h_x0);
    defer cuda.free(d_values);
    const d_radix = try cuda.alloc(u32, 2 * n);
    defer cuda.free(d_radix);
    const d_perm0 = try range(&stream, n);
    defer cuda.free(d_perm0);
    const d_perm1 = try range(&stream, n);
    defer cuda.free(d_perm1);

    try _radixSort(&stream, d_values, d_perm0, d_perm1, d_radix, 0, 0b1);
    // d_radix before cdf = { 1, 0, 1, 0, 1, 0 }
    try expectEqualDeviceSlices(u32, &[_]u32{ 0, 1, 1, 2, 2, 3 }, d_radix);
    try expectEqualDeviceSlices(u32, &[_]u32{ 0, 2, 1 }, d_perm1);

    try _radixSort(&stream, d_values, d_perm0, d_perm1, d_radix, 1, 0b1);
    // d_radix before cdf = { 0, 1, 1, 1, 0, 0 }
    try expectEqualDeviceSlices(u32, &[_]u32{ 0, 0, 1, 2, 3, 3 }, d_radix);
    try expectEqualDeviceSlices(u32, &[_]u32{ 2, 0, 1 }, d_perm1);

    try cuda.memcpyHtoD(u32, d_perm0, &[_]u32{ 0, 2, 1 });
    try _radixSort(&stream, d_values, d_perm0, d_perm1, d_radix, 0, 0b1);
    // d_radix before cdf = { 1, 1, 0, 0, 0, 1 }
    try expectEqualDeviceSlices(u32, &[_]u32{ 0, 1, 2, 2, 2, 2 }, d_radix);
    try expectEqualDeviceSlices(u32, &[_]u32{ 0, 2, 1 }, d_perm1);

    try _radixSort(&stream, d_values, d_perm0, d_perm1, d_radix, 1, 0b1);
    // d_radix before cdf = { 0, 1, 1, 1, 0, 0 }
    try expectEqualDeviceSlices(u32, &[6]u32{ 0, 0, 1, 2, 3, 3 }, d_radix);
    try expectEqualDeviceSlices(u32, &[_]u32{ 2, 1, 0 }, d_perm1);
}

test "updatePermutation" {
    var stream = initStreamWithModule(0);
    defer stream.deinit();

    const h_x0 = [_]u32{ 0b1000, 0b0001 };
    const n = h_x0.len;
    const d_values = try cuda.allocAndCopy(u32, &h_x0);
    defer cuda.free(d_values);

    var d_radix = try cuda.allocAndCopy(u32, &[_]u32{ 0, 1, 1, 1, 2, 2, 2, 2 });
    defer cuda.free(d_radix);
    var d_perm0 = try range(&stream, n);
    defer cuda.free(d_perm0);
    const d_perm1 = try range(&stream, n);
    defer cuda.free(d_perm1);

    try k.updatePermutation.launch(
        &stream,
        cuda.Grid.init1D(n, 1024),
        .{
            d_perm1.ptr,
            d_radix.ptr,
            d_values.ptr,
            d_perm0.ptr,
            0,
            @intCast(c_uint, 0b11),
            @intCast(c_int, n),
        },
    );
}

test "radixSort" {
    var stream = initStreamWithModule(0);
    defer stream.deinit();
    const h_x0 = [_]u32{ 2, 3, 1, 0, 6, 7, 5, 4 };
    // h_x should be it's own permutation, since there is only consecutive integers
    const expected = [_]u32{ 2, 3, 1, 0, 6, 7, 5, 4 };
    const d_x = try cuda.allocAndCopy(u32, &h_x0);
    defer cuda.free(d_x);
    var d_perm0 = try radixSortAlloc(&stream, d_x);
    defer cuda.free(d_perm0);
    try expectEqualDeviceSlices(u32, &expected, d_perm0);

    const h_x1 = [_]u32{ 1073741824, 1077936128, 1065353216, 0, 1088421888, 1091567616, 1086324736, 1084227584 };
    try cuda.memcpyHtoD(u32, d_x, &h_x1);
    var d_perm1 = try radixSortAlloc(&stream, d_x);
    try expectEqualDeviceSlices(u32, &expected, d_perm1);

    // With floats !
    const h_x2 = [_]f32{ 2.0, 3.0, 1.0, 0.0, 6.0, 7.0, 5.0, 4.0 };
    try cuda.memcpyHtoD(u32, d_x, bitCastU32(&h_x2));
    var d_perm2 = try radixSortAlloc(&stream, d_x);
    try expectEqualDeviceSlices(u32, &expected, d_perm2);
}

fn test_naive_normalized_cross_correlation(
    stream: *const cuda.Stream,
    d_template: []const cu.uchar3,
    num_rows: usize,
    num_cols: usize,
) !void {
    try testing.expectEqual(num_rows, num_cols);
    const half_height = @divFloor(num_rows, 2);
    log.info("Loaded template: {}x{}", .{ num_rows, num_rows });
    const d_scores = try cuda.alloc(f32, num_rows * num_rows);
    defer cuda.free(d_scores);

    const gridWithChannel = cuda.Grid.init3D(num_rows, num_rows, 3, 32, 1, 3);
    // Auto cross-correlation
    log.info("crossCorrelation({}, {},)", .{ gridWithChannel, gridWithChannel.sharedMem(f32, 1) });
    try k.crossCorrelation.launchWithSharedMem(
        stream,
        gridWithChannel,
        gridWithChannel.sharedMem(f32, 1),
        .{
            d_scores.ptr,
            d_template.ptr,
            d_template.ptr,
            @intCast(c_int, num_rows),
            @intCast(c_int, num_rows),
            @intCast(c_int, num_rows),
            @intCast(c_int, half_height),
            @intCast(c_int, num_rows),
            @intCast(c_int, half_height),
            @intCast(c_int, num_rows * num_rows),
        },
    );
    // debugDevice(allocator, "auto_corr", d_scores);
    // Should be maximal at the center
    const center_corr = stream.copyResult(f32, &d_scores[num_rows * half_height + half_height]);
    const max_corr = try reduce(stream, k.max, d_scores);
    try testing.expectEqual(max_corr, center_corr);
}

fn bitCastU32(data: anytype) []const u32 {
    return @ptrCast([*]const u32, data)[0..data.len];
}

fn expectEqualDeviceSlices(
    comptime DType: type,
    h_expected: []const DType,
    d_values: []const DType,
) !void {
    const allocator = std.testing.allocator;
    const h_values = try cuda.allocAndCopyResult(DType, allocator, d_values);
    defer allocator.free(h_values);
    testing.expectEqualSlices(DType, h_expected, h_values) catch |err| {
        log.err("Expected: {any}, got: {any}", .{ h_expected, h_values });
        return err;
    };
}

fn debugDevice(
    allocator: std.mem.Allocator,
    name: []const u8,
    d_values: anytype,
) void {
    const DType = std.meta.Elem(@TypeOf(d_values));
    var h = cuda.allocAndCopyResult(DType, allocator, d_values) catch unreachable;
    defer allocator.free(h);
    log.debug("{s} -> {any}", .{ name, h });
}

// TODO: generate this when the kernel is written in Zig.
const Kernels = struct {
    addConstant: cuda.CudaKernel("add_constant"),
    cdfIncremental: cuda.CudaKernel("cdf_incremental"),
    cdfShift: cuda.CudaKernel("cdf_incremental_shift"),
    crossCorrelation: cuda.CudaKernel("naive_normalized_cross_correlation"),
    findRadixSplitted: cuda.CudaKernel("find_radix_splitted"),
    rangeFn: cuda.CudaKernel("range"),
    min: cuda.CudaKernel("reduce_min"),
    max: cuda.CudaKernel("reduce_max"),
    minU32: cuda.CudaKernel("reduce_min_u32"),
    maxU32: cuda.CudaKernel("reduce_max_u32"),
    removeRedness: cuda.CudaKernel("remove_redness"),
    sortNet: cuda.CudaKernel("sort_network"),
    sumU32: cuda.CudaKernel("reduce_sum_u32"),
    updatePermutation: cuda.CudaKernel("update_permutation"),
};
var k: Kernels = undefined;

fn initStreamWithModule(device: u3) cuda.Stream {
    const stream = cuda.Stream.init(device) catch unreachable;
    // Panic if we can't load the module.
    k = Kernels{
        .addConstant = @TypeOf(k.addConstant).init() catch unreachable,
        .cdfIncremental = @TypeOf(k.cdfIncremental).init() catch unreachable,
        .cdfShift = @TypeOf(k.cdfShift).init() catch unreachable,
        .crossCorrelation = @TypeOf(k.crossCorrelation).init() catch unreachable,
        .findRadixSplitted = @TypeOf(k.findRadixSplitted).init() catch unreachable,
        .rangeFn = @TypeOf(k.rangeFn).init() catch unreachable,
        .min = @TypeOf(k.min).init() catch unreachable,
        .max = @TypeOf(k.max).init() catch unreachable,
        .minU32 = @TypeOf(k.minU32).init() catch unreachable,
        .maxU32 = @TypeOf(k.maxU32).init() catch unreachable,
        .removeRedness = @TypeOf(k.removeRedness).init() catch unreachable,
        .sortNet = @TypeOf(k.sortNet).init() catch unreachable,
        .sumU32 = @TypeOf(k.sumU32).init() catch unreachable,
        .updatePermutation = @TypeOf(k.updatePermutation).init() catch unreachable,
    };
    return stream;
}
