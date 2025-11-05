const ptx = @import("nvptx");
pub const panic = ptx.panic;

pub fn rgba_to_greyscale(rgbaImage: [*]const u8, greyImage: [*]u8, len: u64) callconv(ptx.kernel) void {
    const i = ptx.getIdX();
    if (i >= len) return;
    const px = rgbaImage[i * 3 .. i * 3 + 3];
    const R: u32 = @intCast(px[0]);
    const G: u32 = @intCast(px[1]);
    const B: u32 = @intCast(px[2]);
    const grey: u32 = @divFloor(299 * R + 587 * G + 114 * B, 1000);
    greyImage[i] = @intCast(grey);
}

comptime {
    if (ptx.is_nvptx) {
        @export(&rgba_to_greyscale, .{ .name = "rgba_to_greyscale" });
    }
}
