const std = @import("std");
const assert = std.debug.assert;
const log = std.log;

const cuda = @import("cudaz");
const cu = cuda.cu;

const png = @import("png.zig");
const Image = png.Image;

const builtin = @import("builtin");
const CallingConvention = @import("std").builtin.CallingConvention;
pub const is_nvptx = builtin.cpu.arch == .nvptx64;
pub const kernel: CallingConvention = if (is_nvptx) .PtxKernel else .Unspecified;

pub fn validate_output(alloc: std.mem.Allocator, comptime dir: []const u8, threshold: f32) !void {
    const output = try Image.fromFilePath(alloc, dir ++ "output.png");
    const reference = try Image.fromFilePath(alloc, dir ++ "reference.png");

    log.info("Loaded output image and reference image for comparison", .{});
    assert(output.width == reference.width);
    assert(output.height == reference.height);
    // assert(output.image_format == reference.image_format);
    assert(output.raw().len == reference.raw().len);

    const avg_diff = try eq_and_show_diff(alloc, dir, output, reference);
    if (avg_diff < threshold) {
        log.info("*** The image matches, Congrats ! ***", .{});
    }
}

pub fn eq_and_show_diff(alloc: std.mem.Allocator, comptime dir: []const u8, output: Image, reference: Image) !f32 {
    var diff = try png.Image.init(alloc, reference.width, reference.height, .gray8);
    var out_pxls = output.iterator();
    var ref_pxls = reference.iterator();

    const num_pixels = reference.width * reference.height;
    var i: usize = 0;
    var min_val: f32 = 255;
    var max_val: f32 = -255;
    var total: f32 = 0;
    while (true) {
        var ref_pxl = ref_pxls.next();
        if (ref_pxl == null) break;
        var out_pxl = out_pxls.next();
        var d = ref_pxl.?.r - out_pxl.?.r;
        d = std.math.fabs(d);
        min_val = std.math.min(min_val, d);
        max_val = std.math.max(max_val, d);
        i += 1;
        total += d;
    }
    var avg_diff = 255.0 * total / @intToFloat(f32, num_pixels);
    i = 0;
    var diff_pxls = diff.px.gray8;
    while (true) {
        var ref_pxl = ref_pxls.next();
        if (ref_pxl == null) break;
        var out_pxl = out_pxls.next();
        var d = ref_pxl.?.r - out_pxl.?.r;
        d = std.math.fabs(d);
        const centered_d = 255.0 * (d - min_val) / (max_val - min_val);
        diff_pxls[i] = @floatToInt(u8, centered_d);
        i += 1;
    }

    try diff.writeToFilePath(dir ++ "output_diff.png");
    if (min_val != 0 or max_val != 0) {
        std.log.err("Found diffs between two images, avg: {d:.3}, ranging from {d:.1} to {d:.1} pixel value.", .{ avg_diff, 255 * min_val, 255 * max_val });
    }
    return avg_diff;
}

pub fn asUchar3(img: Image) []cu.uchar3 {
    var ptr: [*]cu.uchar3 = @ptrCast([*]cu.uchar3, img.px.rgb24);
    const num_pixels = img.width * img.height;
    return ptr[0..num_pixels];
}

pub fn expectEqualDeviceSlices(
    comptime DType: type,
    h_expected: []const DType,
    d_values: []const DType,
) !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    const allocator = gpa.allocator();
    const h_values = try cuda.allocAndCopyResult(DType, allocator, d_values);
    defer allocator.free(h_values);
    std.testing.expectEqualSlices(DType, h_expected, h_values) catch |err| {
        if (h_expected.len < 80) {
            log.err("Expected: {any}, got: {any}", .{ h_expected, h_values });
        }
        return err;
    };
}
