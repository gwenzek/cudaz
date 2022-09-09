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
const log = std.log.scoped(.png);

/// Control how many pixels are printed when formatting an image.
pub const PRINT_PIXELS = 30;

pub const Rgb24 = extern struct { r: u8, g: u8, b: u8 };
pub const Gray8 = u8;
pub const Rgb_f32 = struct { r: f32, g: f32, b: f32, alpha: f32 = 1.0 };

// TODO enable all types
pub const ColorType = enum(c_uint) {
    rgb24 = png.LCT_RGB,
    gray8 = png.LCT_GREY,
};

pub fn PixelType(t: ColorType) type {
    return switch (t) {
        .rgb24 => Rgb24,
        .gray8 => Gray8,
    };
}

pub fn grayscale(allocator: std.mem.Allocator, width: u32, height: u32) !Image {
    return Image.init(allocator, width, height, .gray8);
}

pub const Image = struct {
    width: u32,
    height: u32,
    px: union(ColorType) { rgb24: []Rgb24, gray8: []u8 },
    allocator: Allocator,

    pub fn init(allocator: Allocator, width: u32, height: u32, color_type: ColorType) !Image {
        const n = @as(usize, width * height);
        return Image{
            .width = width,
            .height = height,
            .allocator = allocator,
            .px = switch (color_type) {
                .rgb24 => .{ .rgb24 = try allocator.alloc(Rgb24, n) },
                .gray8 => .{ .gray8 = try allocator.alloc(Gray8, n) },
            },
        };
    }

    pub fn deinit(self: Image) void {
        self.allocator.free(self.raw());
    }

    pub fn raw(self: Image) []u8 {
        return switch (self.px) {
            .rgb24 => |px| std.mem.sliceAsBytes(px),
            .gray8 => |px| std.mem.sliceAsBytes(px),
        };
    }

    pub fn len(self: Image) usize {
        return self.width * self.height;
    }

    pub fn fromFilePath(allocator: Allocator, file_path: []const u8) !Image {
        var img: Image = undefined;
        // TODO: use lodepng_inspect to get the image size and handle allocations ourselves
        img.allocator = std.heap.c_allocator;
        var buffer: [*c]u8 = undefined;

        var resolved_path = try std.fs.path.resolve(allocator, &[_][]const u8{file_path});
        defer allocator.free(resolved_path);
        var resolved_pathZ: []u8 = try allocator.dupeZ(u8, resolved_path);
        defer allocator.free(resolved_pathZ);

        // TODO: handle different color encoding
        try check(png.lodepng_decode24_file(
            &buffer,
            &img.width,
            &img.height,
            @ptrCast([*c]const u8, resolved_pathZ),
        ));
        std.debug.assert(buffer != null);

        img.px = .{ .rgb24 = @ptrCast([*]Rgb24, buffer.?)[0..img.len()] };
        return img;
    }

    fn lodeBitDepth(self: Image) c_uint {
        _ = self;
        return 8;
    }

    pub fn writeToFilePath(self: Image, file_path: []const u8) !void {
        var resolved_path = try std.fs.path.resolve(self.allocator, &[_][]const u8{file_path});
        defer self.allocator.free(resolved_path);
        var resolved_pathZ: []u8 = try self.allocator.dupeZ(u8, resolved_path);
        defer self.allocator.free(resolved_pathZ);
        // Write image data
        try check(png.lodepng_encode_file(
            @ptrCast([*c]const u8, resolved_pathZ),
            self.raw().ptr,
            @intCast(c_uint, self.width),
            @intCast(c_uint, self.height),
            @enumToInt(self.px),
            self.lodeBitDepth(),
        ));

        log.info("Wrote full image {s}", .{resolved_pathZ});
        return;
    }

    // TODO: does it make sense to use f32 here ? shouldn't we stick with
    pub const Iterator = struct {
        image: Image,
        i: usize,

        fn u8_to_f32(value: u8) f32 {
            return @intToFloat(f32, value) / 255.0;
        }

        pub fn next(self: *Iterator) ?Rgb_f32 {
            if (self.i >= self.image.width * self.image.height) return null;
            const px_f32 = switch (self.image.px) {
                .rgb24 => |pixels| blk: {
                    const px = pixels[self.i];
                    break :blk Rgb_f32{ .r = u8_to_f32(px.r), .g = u8_to_f32(px.g), .b = u8_to_f32(px.b) };
                },
                .gray8 => |pixels| blk: {
                    const gray = u8_to_f32(pixels[self.i]);
                    break :blk Rgb_f32{ .r = gray, .g = gray, .b = gray };
                },
            };
            self.i += 1;
            return px_f32;
        }
    };

    pub fn iterator(self: Image) Iterator {
        return .{ .image = self, .i = 0 };
    }

    pub fn format(
        self: *const Image,
        comptime fmt: []const u8,
        options: std.fmt.FormatOptions,
        writer: anytype,
    ) !void {
        _ = fmt;
        _ = options;
        try std.fmt.format(writer, "Image ({s}){}x{}: (...{any}...)", .{ @tagName(self.px), self.width, self.height, self.raw()[200 .. 200 + PRINT_PIXELS] });
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

    var base = try Image.fromFilePath(testing.allocator, "resources/hw1_resources/cinque_terre_small.png");
    defer base.deinit();

    try tmp.dir.writeFile("out.png", "hello");
    var tmp_img = try tmp.dir.realpathAlloc(testing.allocator, "out.png");
    log.warn("will write image ({}x{}) to {s}", .{ base.width, base.height, tmp_img });
    defer testing.allocator.free(tmp_img);
    try base.writeToFilePath(tmp_img);

    var loaded = try Image.fromFilePath(testing.allocator, tmp_img);
    defer loaded.deinit();
    try testing.expectEqualSlices(u8, base.raw(), loaded.raw());
    try testing.expect(img_eq(base, loaded));
}
