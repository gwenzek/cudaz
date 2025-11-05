//! Simplify bindings for zigimg that only deal with .png
const std = @import("std");
const testing = std.testing;

const zigimg = @import("zigimg");
pub const Rgb24 = zigimg.color.Rgb24;
pub const Gray8 = zigimg.color.Grayscale8;
pub const Image = zigimg.Image;

const log = std.log.scoped(.png);

/// Control how many pixels are printed when formatting an image.
pub const PRINT_PIXELS = 30;

pub fn grayscale(allocator: std.mem.Allocator, width: usize, height: usize) !Image {
    return .{
        .width = width,
        .height = height,
        .pixels = .{ .grayscale8 = @ptrCast(try allocator.alloc(u8, width * height)) },
    };
}

pub fn fromFilePath(allocator: std.mem.Allocator, path: []const u8) !Image {
    const file = try std.fs.cwd().openFile(path, .{});

    var file_content: std.Io.Writer.Allocating = .init(allocator);
    defer file_content.deinit();

    var file_reader = file.reader(&.{});
    _ = try file_reader.interface.stream(&file_content.writer, .unlimited);
    return try Image.fromMemory(allocator, file_content.written());
}

pub fn writeToFilePath(img: Image, path: []const u8) !void {
    var buf: [1024]u8 = undefined;
    try img.writeToFilePath(std.heap.smp_allocator, path, &buf, .{ .png = .{} });
}

pub fn img_eq(output: Image, reference: Image) bool {
    const same_dim = (output.width == reference.width) and (output.height == reference.height);
    if (!same_dim) return false;

    const same_len = output.raw().len == reference.raw().len;
    if (!same_len) return false;

    var i: usize = 0;
    while (i < output.raw().len) : (i += 1) {
        if (output.raw()[i] != reference.raw()[i]) return false;
    }
    return true;
}

test "read/write/read" {
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();

    var base = try Image.fromFilePath(testing.allocator, "resources/hw1_resources/cinque_terre_small.png");
    defer base.deinit();

    try tmp.dir.writeFile(.{ .sub_path = "out.png", .data = "hello" });
    const tmp_img = try tmp.dir.realpathAlloc(testing.allocator, "out.png");
    log.warn("will write image ({}x{}) to {s}", .{ base.width, base.height, tmp_img });
    defer testing.allocator.free(tmp_img);
    try base.writeToFilePath(tmp_img);

    var loaded = try Image.fromFilePath(testing.allocator, tmp_img);
    defer loaded.deinit();
    try testing.expectEqualSlices(u8, base.raw(), loaded.raw());
    try testing.expect(img_eq(base, loaded));
}

pub const KittyFmt = struct {
    img: Image,
    cols: u32 = 120,
    rows: ?u32 = null,

    pub fn format(fmt: KittyFmt, writer: *std.Io.Writer) std.Io.Writer.Error!void {
        const img = fmt.img;

        const pixel_type: u8 = switch (img.pixels) {
            .rgb24 => 24,
            .rgba32 => 32,
            else => return try writer.print("Img {d}x{d} {t} -> not supported", .{ img.width, img.height, img.pixels }),
        };

        const scaled_rows = std.math.divCeil(usize, fmt.cols * img.height, img.width * 2) catch unreachable;

        try writer.print("\x1b_Gf={d},a=T,t=d,s={d},v={d},c={d},r={d},m=1;", .{
            pixel_type,
            img.width,
            img.height,
            fmt.cols,
            fmt.rows orelse scaled_rows,
        });

        var it = std.mem.window(u8, img.rawBytes(), 4096, 4096);
        const first_chunk = it.first();
        try std.base64.standard.Encoder.encodeWriter(writer, first_chunk);
        try writer.writeAll("\x1b\\");

        while (it.next()) |chunk| {
            const has_next = it.index != null;
            try writer.print("\x1b_Gm={d};", .{@intFromBool(has_next)});
            try std.base64.standard.Encoder.encodeWriter(writer, chunk);
            try writer.writeAll("\x1b\\");
        }
        try writer.writeAll("\n");
        try writer.flush();
    }
};
