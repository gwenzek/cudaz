const std = @import("std");
const assert = std.debug.assert;
const log = std.log;

const zigimg = @import("zigimg");
const Image = zigimg.Image;

const png = @import("png.zig");

pub fn grayscale(allocator: *std.mem.Allocator, width: usize, height: usize) !Image {
    return try Image.create(
        allocator,
        width,
        height,
        .Grayscale8,
        .Png,
    );
}

pub fn validate_output(alloc: *std.mem.Allocator, comptime dir: []const u8, threshold: f32) !void {
    const output = try Image.fromFilePath(alloc, dir ++ "output.png");
    const reference = try Image.fromFilePath(alloc, dir ++ "reference.png");

    log.info("Loaded output image and reference image for comparison", .{});
    assert(output.width == reference.width);
    assert(output.height == reference.height);
    assert(output.image_format == reference.image_format);
    assert(output.pixels.?.len() == reference.pixels.?.len());

    const avg_diff = try eq_and_show_diff(alloc, dir, output, reference);
    if (avg_diff < threshold) {
        log.info("*** The image matches, Congrats ! ***", .{});
    }
}

pub fn eq_and_show_diff(alloc: *std.mem.Allocator, comptime dir: []const u8, output: Image, reference: Image) !f32 {
    var diff = try grayscale(alloc, reference.width, reference.height);
    var out_pxls = output.iterator();
    var ref_pxls = reference.iterator();

    const num_pixels = reference.pixels.?.len();
    var i: usize = 0;
    var min_val: f32 = 255;
    var max_val: f32 = -255;
    var total: f32 = 0;
    while (true) {
        var ref_pxl = ref_pxls.next();
        if (ref_pxl == null) break;
        var out_pxl = out_pxls.next();
        var d = ref_pxl.?.R - out_pxl.?.R;
        d = std.math.absFloat(d);
        min_val = std.math.min(min_val, d);
        max_val = std.math.max(max_val, d);
        i += 1;
        total += d;
    }
    var avg_diff = 255.0 * total / @intToFloat(f32, num_pixels);
    i = 0;
    var diff_pxls = diff.pixels.?.Grayscale8;
    while (true) {
        var ref_pxl = ref_pxls.next();
        if (ref_pxl == null) break;
        var out_pxl = out_pxls.next();
        var d = ref_pxl.?.R - out_pxl.?.R;
        d = std.math.absFloat(d);
        const centered_d = 255.0 * (d - min_val) / (max_val - min_val);
        diff_pxls[i] = .{ .value = @floatToInt(u8, centered_d) };
        i += 1;
    }

    try png.writePngToFilePath(diff, dir ++ "output_diff.png");
    if (min_val != 0 or max_val != 0) {
        std.log.err("Found diffs between two images, avg: {d:.3}, ranging from {d:.1} to {d:.1} pixel value.", .{ avg_diff, 255 * min_val, 255 * max_val });
    }
    return avg_diff;
}
