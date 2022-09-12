const ptx = @import("kernel_utils.zig");
pub const panic = ptx.panic;

pub fn rgba_to_greyscale(rgbaImage: []u8, greyImage: []u8) callconv(ptx.Kernel) void {
    const i = ptx.getIdX();
    if (i >= greyImage.len) return;
    const px = rgbaImage[i * 3 .. i * 3 + 3];
    const R = @intCast(u32, px[0]);
    const G = @intCast(u32, px[1]);
    const B = @intCast(u32, px[2]);
    var grey = @divFloor(299 * R + 587 * G + 114 * B, 1000);
    greyImage[i] = @intCast(u8, grey);
}

comptime {
    if (ptx.is_nvptx) {
        @export(rgba_to_greyscale, .{ .name = "rgba_to_greyscale" });
    }
}
