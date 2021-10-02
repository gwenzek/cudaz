const std = @import("std");
const assert = std.debug.assert;
const log = std.log;

const zigimg = @import("zigimg");
const Image = zigimg.Image;

const png = @import("png.zig");

pub fn grayscale(allocator: *std.mem.Allocator, width: usize, height: usize) !Image {
    var img = Image{
        .allocator = allocator,
        .width = width,
        .height = height,
        .pixels = try zigimg.color.ColorStorage.init(allocator, .Grayscale8, width * height),
        .image_format = .Png,
    };

    return img;
}

pub fn validate_output(alloc: *std.mem.Allocator, comptime dir: []const u8) !void {
    const output = try Image.fromFilePath(alloc, dir ++ "output.png");
    const reference = try Image.fromFilePath(alloc, dir ++ "reference.png");

    log.info("Loaded output image and reference image for comparison", .{});
    assert(output.width == reference.width);
    assert(output.height == reference.height);
    assert(output.image_format == reference.image_format);
    assert(output.pixels.?.Grayscale8.len == reference.pixels.?.Grayscale8.len);

    const img_match = try eq_and_show_diff(alloc, dir, output, reference);
    if (img_match) {
        log.info("*** The image matches, Congrats ! ***", .{});
    } else {
        std.os.exit(1);
    }
}

pub fn eq_and_show_diff(alloc: *std.mem.Allocator, comptime dir: []const u8, output: Image, reference: Image) !bool {
    var diff = try grayscale(alloc, reference.width, reference.height);
    var diff_pxls = diff.pixels.?.Grayscale8;
    var out_pxls = output.pixels.?.Grayscale8;
    var ref_pxls = reference.pixels.?.Grayscale8;

    const num_pixels = ref_pxls.len;
    var i: usize = 0;
    var min_val: i16 = 255;
    var max_val: i16 = -255;
    while (i < diff_pxls.len) : (i += 1) {
        var d: i16 = @intCast(i16, ref_pxls[i].value) - @intCast(i16, out_pxls[i].value);
        // d = try std.math.absInt(d);
        min_val = std.math.min(min_val, d);
        max_val = std.math.max(max_val, d);
    }
    i = 0;
    while (i < diff_pxls.len) : (i += 1) {
        var d: i16 = @intCast(i16, ref_pxls[i].value) - @intCast(i16, out_pxls[i].value);
        // d = try std.math.absInt(d);
        const centered_d = (255.0 * @intToFloat(f32, d - min_val)) / @intToFloat(f32, max_val - min_val);
        diff_pxls[i] = .{ .value = @floatToInt(u8, centered_d) };
    }

    try png.writePngToFilePath(diff, dir ++ "output_diff.png");
    if (min_val != 0 or max_val != 0) {
        std.log.warn("Found diffs between two images, ranging from {} to {} pixel value.", .{ min_val, max_val });
        return false;
    }
    return true;
}
