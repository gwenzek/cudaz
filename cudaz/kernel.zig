const ptx = @import("nvptx.zig");
const message = []u8{ 72, 101, 108, 108, 111, 32, 119, 111, 114, 108, 100, 33, 13, 10 };

export fn hello(out: []u8) void {
    const i = ptx.getId_1D();
    if (i > message.len or i > out.len) return;
    ptx.syncThreads();
    out[i] = message[i];
}

// export fn rgba_to_greyscale(rgbaImage: [*][3]u8, greyImage: [*]u8) void {
//     const i = blockIdx[0] * lockDim[0] + threadIdx[0];
//     const px = rgbaImage[i];
//     // const R = @intToFloat(f32, px[0]);
//     // const G = @intToFloat(f32, px[1]);
//     // const B = @intToFloat(f32, px[2]);
//     // const output = (0.299 * R + 0.587 * G + 0.114 * B);
//     greyImage[i] = px[0];
// }
