const std = @import("std");
// const log = std.log;
const math = std.math;
const testing = std.testing;
const assert = std.debug.assert;

const zigimg = @import("zigimg");

const cuda = @import("cudaz");
const cu = cuda.cu;

const png = @import("png.zig");
const utils = @import("utils.zig");

const resources_dir = "CS344/hw4_resources/";

const log = std.log.scoped(.HW4);

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

    // TODO
    // d_scores += try reduce_min(&stream, d_scores);

    var timer = cuda.GpuTimer.init(&stream);
    timer.start();

    var d_permutation = try mySort(&stream, d_scores);
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
pub fn mySort(stream: *const cuda.Stream, d_values: []const f32) ![]c_uint {
    var coords = try range(stream, d_values.len);
    defer cuda.free(coords);
    var d_permutation = try cuda.alloc(c_uint, coords.len);
    try cuda.memset(c_uint, d_permutation, 0);
    // const radix_cdf = try cuda.Function("radix_cdf").init();
    // TODO
    // Sorting is done in several stages
    // Sorting of small arrays is done with Radix/sortNetwork
    // Then we can merge using parallel merging
    return d_permutation;
}

pub fn range(stream: *const cuda.Stream, len: usize) ![]u32 {
    var coords = try cuda.alloc(c_uint, len);
    const rangeFn = try cuda.Function("range").init();
    try rangeFn.launch(stream, cuda.Grid.init1D(len, 64), .{ coords.ptr, @intCast(c_uint, len) });
    return coords;
}

test "range" {
    var stream = try cuda.Stream.init(0);
    defer stream.deinit();
    var numbers = try range(&stream, 5);
    try expectEqualDeviceSlices(u32, &[_]u32{ 0, 1, 2, 3, 4 }, numbers);
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

pub fn sortNetwork(stream: *const cuda.Stream, d_data: []f32, n_threads: usize) !void {
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

pub fn inPlaceCdf(stream: *const cuda.Stream, d_values: []u32, n_threads: u32) cuda.CudaError!void {
    const n = d_values.len;
    const grid_N = cuda.Grid.init1D(n, n_threads);
    const N = grid_N.blocks.x;
    var d_grid_bins = try cuda.alloc(u32, N);
    defer cuda.free(d_grid_bins);
    log.debug("cdf({}, {})", .{ n, N });
    const cdfIncremental = try cuda.Function("cdf_incremental").init();
    try cdfIncremental.launchWithSharedMem(
        stream,
        grid_N,
        n_threads * @sizeOf(u32),
        .{ d_values.ptr, d_grid_bins.ptr, @intCast(c_int, n) },
    );
    if (N == 1) return;

    log.debug("cdf_shift({}, {})", .{ n, N });
    try inPlaceCdf(stream, d_grid_bins, n_threads);
    const cdfShift = try cuda.Function("cdf_incremental_shift").init();
    try cdfShift.launch(
        stream,
        grid_N,
        .{ d_values.ptr, d_grid_bins.ptr, @intCast(c_int, n) },
    );
}

test "inPlaceCdf" {
    var stream = try cuda.Stream.init(0);
    defer stream.deinit();
    const h_x = [_]u32{ 0, 2, 1, 1, 0, 1, 3, 0 };
    var h_out = [_]u32{0} ** h_x.len;
    const h_cdf = [_]u32{ 0, 0, 2, 3, 4, 4, 5, 8 };
    const d_x = try cuda.alloc(u32, h_x.len);
    defer cuda.free(d_x);

    try cuda.memcpyHtoD(u32, d_x, &h_x);
    try inPlaceCdf(&stream, d_x, 8);
    try cuda.memcpyDtoH(u32, &h_out, d_x);
    try testing.expectEqual(h_cdf, h_out);

    try cuda.memcpyHtoD(u32, d_x, &h_x);
    try inPlaceCdf(&stream, d_x, 16);
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
    const mask_bits: u8 = 8 - @clz(u8, mask);

    const d_radix = try cuda.alloc(u32, n * (mask + 1));
    defer cuda.free(d_radix);
    var d_perm0 = try range(stream, n);
    errdefer cuda.free(d_perm0);
    var d_perm1 = try cuda.alloc(u32, n);
    errdefer cuda.free(d_perm1);

    log.warn("radixSort(n={}, mask_bits={}, mask={})", .{ n, mask_bits, mask });
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
    // TODO: this should be done only once
    const findRadixSplitted = try cuda.Function("find_radix_splitted").init();
    const updatePermutation = try cuda.Function("update_permutation").init();
    try cuda.memset(u32, d_radix, 0);
    debugDevice("d_values", d_values);
    debugDevice("d_perm0", d_perm0);
    log.debug("radixSort(n={}, shift={}, mask={})", .{ n, shift, mask });
    try findRadixSplitted.launch(
        stream,
        cuda.Grid.init1D(n, 1024),
        .{
            d_radix.ptr,
            d_values.ptr,
            d_perm0.ptr,
            shift,
            mask,
            @intCast(c_int, n),
        },
    );
    debugDevice("d_radix", d_radix);
    try inPlaceCdf(stream, d_radix, 1024);
    debugDevice("d_radix + cdf", d_radix);
    try updatePermutation.launch(
        stream,
        cuda.Grid.init1D(n, 1024),
        .{
            d_perm1.ptr,
            d_radix.ptr,
            d_values.ptr,
            d_perm0.ptr,
            shift,
            mask,
            @intCast(c_int, n),
        },
    );
    debugDevice("d_perm1", d_perm1);
}

test "findRadixSplitted" {
    var stream = try cuda.Stream.init(0);
    defer stream.deinit();
    const h_x0 = [_]u32{ 0b10, 0b01, 0b00 };
    const n = h_x0.len;
    const d_values = try cuda.allocAndCopy(u32, &h_x0);
    defer cuda.free(d_values);
    const findRadixSplitted = try cuda.Function("find_radix_splitted").init();
    const d_radix = try cuda.alloc(u32, 2 * n);
    defer cuda.free(d_radix);
    try cuda.memset(u32, d_radix, 0);
    const d_perm0 = try range(&stream, n);
    defer cuda.free(d_perm0);

    try findRadixSplitted.launch(
        &stream,
        cuda.Grid.init1D(n, 1024),
        .{ d_radix.ptr, d_values.ptr, d_perm0.ptr, 0, @intCast(c_uint, 0b1), @intCast(c_int, n) },
    );
    try expectEqualDeviceSlices(u32, &[_]u32{ 1, 0, 1, 0, 1, 0 }, d_radix);

    try cuda.memset(u32, d_radix, 0);
    try findRadixSplitted.launch(
        &stream,
        cuda.Grid.init1D(n, 1024),
        .{ d_radix.ptr, d_values.ptr, d_perm0.ptr, 1, @intCast(c_uint, 0b1), @intCast(c_int, n) },
    );
    try expectEqualDeviceSlices(u32, &[_]u32{ 0, 1, 1, 1, 0, 0 }, d_radix);

    try cuda.memcpyHtoD(u32, d_perm0, &[_]u32{ 0, 2, 1 });
    // values: { 0b10, 0b01, 0b00 }; perm: {0, 2, 1}
    try cuda.memset(u32, d_radix, 0);
    try findRadixSplitted.launch(
        &stream,
        cuda.Grid.init1D(n, 1024),
        .{ d_radix.ptr, d_values.ptr, d_perm0.ptr, 0, @intCast(c_uint, 0b1), @intCast(c_int, n) },
    );
    try expectEqualDeviceSlices(u32, &[_]u32{ 1, 1, 0, 0, 0, 1 }, d_radix);

    try cuda.memset(u32, d_radix, 0);
    try findRadixSplitted.launch(
        &stream,
        cuda.Grid.init1D(n, 1024),
        .{ d_radix.ptr, d_values.ptr, d_perm0.ptr, 1, @intCast(c_uint, 0b1), @intCast(c_int, n) },
    );
    try expectEqualDeviceSlices(u32, &[_]u32{ 0, 1, 1, 1, 0, 0 }, d_radix);
}

test "_radixSort" {
    var stream = try cuda.Stream.init(0);
    defer stream.deinit();
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
    var stream = try cuda.Stream.init(0);
    defer stream.deinit();
    const updatePermutation = try cuda.Function("update_permutation").init();

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

    try updatePermutation.launch(
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
    // TODO test with a harder permutation
}

test "radixSort" {
    var stream = try cuda.Stream.init(0);
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
    const h_x2 = [_]f32{ 2, 3, 1, 0, 6, 7, 5, 4 };
    try cuda.memcpyHtoD(u32, d_x, bitCastU32(&h_x2));
    var d_perm2 = try radixSortAlloc(&stream, d_x);
    try expectEqualDeviceSlices(u32, &expected, d_perm2);
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
        log.debug("Expected: {any}, got: {any}", .{ h_expected, h_values });
        return err;
    };
}

fn debugDevice(
    name: []const u8,
    d_values: anytype,
) void {
    const DType = std.meta.Elem(@TypeOf(d_values));
    var h = cuda.allocAndCopyResult(DType, testing.allocator, d_values) catch unreachable;
    defer testing.allocator.free(h);
    log.debug("{s} -> {any}", .{ name, h });
}
