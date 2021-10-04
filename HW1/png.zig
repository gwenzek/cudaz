const std = @import("std");
pub const png = @cImport({
    @cInclude("png.h");
    @cInclude("stdio.h");
});
const zigimg = @import("zigimg");
const Image = zigimg.Image;
const testing = std.testing;

// pub fn readImage(alloc: std.mem.Allocator, filename: []const u8) PngError!PngImg {
//     var header: [8]u8 = undefined;    // 8 is the maximum size that can be checked
//         // open file and test for it being a png
//         FILE *fp = fopen(filename, "rb");
//   assert(fp);
//         fread(header, 1, 8, fp);
//         assert(!png_sig_cmp(header, 0, 8));

//         // initialize stuff
//         png_structp png_ptr = png_create_read_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
//   assert(png_ptr);

//         png_infop info_ptr = png_create_info_struct(png_ptr);
//   assert(info_ptr);

//         assert(!setjmp(png_jmpbuf(png_ptr)));

//         png_init_io(png_ptr, fp);
//         png_set_sig_bytes(png_ptr, 8);

//         png_read_info(png_ptr, info_ptr);

//         *width = png_get_image_width(png_ptr, info_ptr);
//         *height = png_get_image_height(png_ptr, info_ptr);
//         int color_type = png_get_color_type(png_ptr, info_ptr);
//   assert(color_type == PNG_COLOR_TYPE_RGB);
//         int bit_depth = png_get_bit_depth(png_ptr, info_ptr);
//   assert(bit_depth == 8);
//   int pitch = png_get_rowbytes(png_ptr, info_ptr);

//         int number_of_passes = png_set_interlace_handling(png_ptr);
//         png_read_update_info(png_ptr, info_ptr);

//         // read file
//         assert(!setjmp(png_jmpbuf(png_ptr)));

//   png_bytep buffer = malloc(*height * pitch);
//   void *ret = buffer;
//   assert(buffer);
//         png_bytep *row_pointers = malloc(sizeof(png_bytep) * *height);
//   assert(row_pointers);
//         for (int i = 0; i < *height; i++) {
//                 row_pointers[i] = buffer;
//     buffer += pitch;
//   }

//         png_read_image(png_ptr, row_pointers);

//         fclose(fp);
//   free(row_pointers);
//   return ret;
// }

pub fn writePngToFilePath(self: zigimg.Image, file_path: []const u8) !void {
    // char* filename, int width, int height, void *buffer, char* title)
    if (self.pixels == null) {
        return error.NoPixelData;
    }

    const cwd = std.fs.cwd();

    var resolved_path = try std.fs.path.resolve(self.allocator, &[_][]const u8{file_path});
    defer self.allocator.free(resolved_path);
    try writePngToFile(self, resolved_path);
}

pub fn writePngToFile(self: zigimg.Image, resolved_path: []const u8) !void {
    var code: i32 = 0;
    // var png_ptr: png.png_structp = undefined;
    // var info_ptr: png.png_infop = undefined;
    //
    var resolved_pathZ: []u8 = try self.allocator.dupeZ(u8, resolved_path);
    defer self.allocator.free(resolved_pathZ);

    // Initialize write structure
    var fp = png.fopen(resolved_pathZ.ptr, "wb");
    if (fp == null) {
        return error.InputOutput;
    }
    defer _ = png.fclose(fp);
    std.log.info("Opened: {s} ({*})", .{ resolved_pathZ, fp });
    var png_ptr = png.png_create_write_struct(png.PNG_LIBPNG_VER_STRING, null, null, null);
    if (png_ptr == null) {
        // fprintf(stderr, "Could not allocate write struct\n");
        return error.OutOfMemory;
    }
    defer png.png_destroy_write_struct(&png_ptr, null);

    // Initialize info structure
    var info_ptr = png.png_create_info_struct(png_ptr);
    if (info_ptr == null) {
        return error.OutOfMemory;
    }
    defer png.png_free_data(png_ptr, info_ptr, png.PNG_FREE_ALL, -1);

    // // Setup Exception handling
    // if (setjmp(png_jmpbuf(png_ptr))) {
    //   fprintf(stderr, "Error during png creation\n");
    //   code = 1;
    //   goto finalise;
    // }

    png.png_init_io(png_ptr, fp);
    std.log.info("Init IO: {s}", .{png_ptr});

    // Write header (8 bit colour depth)
    const color_type = switch (self.pixels.?) {
        .Rgb24 => png.PNG_COLOR_TYPE_RGB,
        .Grayscale8 => png.PNG_COLOR_TYPE_GRAY,
        else => return error.ColorNotFound,
    };
    png.png_set_IHDR(
        png_ptr,
        info_ptr,
        @intCast(c_uint, self.width),
        @intCast(c_uint, self.height),
        // TODO: read color depth from image
        8,
        color_type,
        png.PNG_INTERLACE_NONE,
        png.PNG_COMPRESSION_TYPE_DEFAULT,
        png.PNG_FILTER_TYPE_DEFAULT,
    );

    // Set title
    // if (title != null) {
    //     var title_text = png.png_text{
    //         .compression = png.PNG_TEXT_COMPRESSION_NONE,
    //         .key = "Title",
    //         .text = title,
    //     };
    //     png.png_set_text(png_ptr, info_ptr, &title_text, 1);
    // }

    png.png_write_info(png_ptr, info_ptr);
    std.log.info("Wrote IDHR {s}", .{info_ptr});

    // Write image data
    // TODO: adapt to different storage
    const step: usize = switch (self.pixels.?) {
        .Rgb24 => |px| @sizeOf(@TypeOf(px[0])),
        .Grayscale8 => |px| @sizeOf(@TypeOf(px[0])),
        else => return error.ColorNotFound,
    };
    var pixels = switch (self.pixels.?) {
        .Rgb24 => |px| std.mem.sliceAsBytes(px),
        .Grayscale8 => |px| std.mem.sliceAsBytes(px),
        else => return error.ColorNotFound,
    };
    var row_id: usize = 0;
    while (row_id < self.height) : (row_id += 1) {
        var pixel_row = pixels[row_id * self.width * step .. (row_id + 1) * self.width * step];
        png.png_write_row(png_ptr, @ptrCast([*c]u8, pixel_row.ptr));
    }

    // End write
    png.png_write_end(png_ptr, null);
    std.log.info("Wrote full image {s}", .{resolved_path});
    return;
}

pub fn img_eq(output: Image, reference: Image) bool {
    var out_pxls = output.iterator();
    var ref_pxls = reference.iterator();

    const num_pixels = reference.pixels.?.len();
    while (true) {
        var ref_pxl = ref_pxls.next();
        if (ref_pxl == null) return true;
        var out_pxl = out_pxls.next();
        if (out_pxl == null) return false;
        var r = ref_pxl.?;
        var o = out_pxl.?;
        if (o.R != r.R) return false;
        if (o.G != r.G) return false;
        if (o.B != r.B) return false;
        if (o.A != r.A) return false;
    }
    return true;
}

test "read/write/read" {
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();

    var base = try Image.fromFilePath(testing.allocator, "HW1/cinque_terre_small.png");
    defer base.deinit();

    try tmp.dir.writeFile("out.png", "hello");
    var tmp_img = try tmp.dir.realpathAlloc(testing.allocator, "out.png");
    std.log.warn("will write image ({}x{}) to {s}", .{ base.width, base.height, tmp_img });
    defer testing.allocator.free(tmp_img);
    try writePngToFilePath(base, tmp_img);

    var loaded = try Image.fromFilePath(testing.allocator, tmp_img);
    defer loaded.deinit();
    try testing.expectEqualSlices(zigimg.color.Rgb24, base.pixels.?.Rgb24, loaded.pixels.?.Rgb24);
    try testing.expect(img_eq(base, loaded));
}
