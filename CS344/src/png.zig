const std = @import("std");
const Allocator = std.mem.Allocator;
pub const png = @cImport({
    @cDefine("LODEPNG_NO_COMPILE_CPP", "1");
    @cDefine("LODEPNG_COMPILE_ERROR_TEXT", "1");
    // TODO remove libc dependency
    // @cDefine("LODEPNG_NO_COMPILE_DISK", "1");
    // @cDefine("LODEPNG_NO_COMPILE_ALLOCATORS", "1");
    @cInclude("lodepng.h");
});

const testing = std.testing;

const Rgb24 = extern struct { r: u8, g: u8, b: u8 };
const Gray = u8;
const ColorType = enum {
    rgb24,
    gray,
};

const log = std.log.scoped(.png);

const Image = struct {
    width: u32,
    height: u32,
    type: ColorType,
    px: union(ColorType) { rgb24: []Rgb24, gray: []u8 },

    pub fn raw(self: Image) []u8 {
        return switch (self.px) {
            .rgb24 => |px| std.mem.sliceAsBytes(px),
            .gray => |px| std.mem.sliceAsBytes(px),
        };
    }

    pub fn fromFile(allocator: Allocator, file_path: []const u8) !Image {
        var img: Image = undefined;
        // TODO: use lodepng_inspect to get the image size and handle allocations ourselves
        var buffer: [*c]u8 = undefined;

        var resolved_path = try std.fs.path.resolve(allocator, &[_][]const u8{file_path});
        defer allocator.free(resolved_path);
        var resolved_pathZ: []u8 = try allocator.dupeZ(u8, resolved_path);
        defer allocator.free(resolved_pathZ);

        try check(png.lodepng_decode24_file(
            &buffer,
            &img.width,
            &img.height,
            @ptrCast([*c]const u8, resolved_pathZ),
        ));
        std.debug.assert(buffer != null);

        const n = img.width * img.height;
        // TODO: handle different color encoding
        img.px = .{ .rgb24 = @ptrCast([*]Rgb24, buffer.?)[0..n] };
        return img;
    }

    fn lodeBitDepth(self: Image) c_uint {
        _ = self;
        return 8;
    }

    pub fn writeToFilePath(self: Image, file_path: []const u8) !void {
        var resolved_path = try std.fs.path.resolve(testing.allocator, &[_][]const u8{file_path});
        defer testing.allocator.free(resolved_path);
        var resolved_pathZ: []u8 = try testing.allocator.dupeZ(u8, resolved_path);
        defer testing.allocator.free(resolved_pathZ);
        // Write image data
        // TODO: adapt to different storage
        try check(png.lodepng_encode24_file(
            @ptrCast([*c]const u8, resolved_pathZ),
            self.raw().ptr,
            @intCast(c_uint, self.width),
            @intCast(c_uint, self.height),
        ));

        log.info("Wrote full image {s}", .{resolved_pathZ});
        return;
    }

    pub fn deinit(self: Image) void {
        std.heap.c_allocator.free(self.raw());
    }
};

pub fn check(err: c_uint) !void {
    if (err != 0) {
        log.err("Error {s}({})", .{ png.lodepng_error_text(err), err });
        return error.PngError;
    }
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

    var base = try Image.fromFile(testing.allocator, "resources/hw1_resources/cinque_terre_small.png");
    defer base.deinit();

    try tmp.dir.writeFile("out.png", "hello");
    var tmp_img = try tmp.dir.realpathAlloc(testing.allocator, "out.png");
    log.warn("will write image ({}x{}) to {s}", .{ base.width, base.height, tmp_img });
    defer testing.allocator.free(tmp_img);
<<<<<<< HEAD
    try base.writePngToFile(tmp_img);
=======
    try base.writeToFilePath(tmp_img);
>>>>>>> bb5c855 (fixup! use lodepng instead of libpng / zigimg)

    var loaded = try Image.fromFile(testing.allocator, tmp_img);
    defer loaded.deinit();
    try testing.expectEqualSlices(u8, base.raw(), loaded.raw());
    try testing.expect(img_eq(base, loaded));
}
