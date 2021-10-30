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
    const d_img = try cuda.allocAndCopy(cu.uchar3, asUchar3(img));
    const d_template = try cuda.allocAndCopy(cu.uchar3, asUchar3(template));
    const d_scores = try cuda.alloc(f32, num_rows * num_cols);

    // Create a 2D grid for the image and use the last dimension for the channel (R, G, B)
    var stream = try cuda.Stream.init(0);
    defer stream.deinit();
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

    try cuda.memcpyDtoH(cu.uchar3, asUchar3(img), d_img);
    try png.writePngToFilePath(img, resources_dir ++ "output.png");
    try utils.validate_output(allocator, resources_dir);
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

    // TODO
    _ = d_out_vals;
    return d_permutation;
}

pub fn range(stream: *const cuda.Stream, len: usize) ![]c_uint {
    var coords = try cuda.alloc(c_uint, len);
    const rangeFn = try cuda.Function("range").init();
    try rangeFn.launch(stream, cuda.Grid.init1D(len, 64), .{ coords.ptr, @intCast(c_uint, len) });
    return coords;
}

pub fn asUchar3(img: zigimg.Image) []cu.uchar3 {
    var ptr: [*]cu.uchar3 = @ptrCast([*]cu.uchar3, img.pixels.?.Rgb24);
    const num_pixels = img.width * img.height;
    return ptr[0..num_pixels];
}
