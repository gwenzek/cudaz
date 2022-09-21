const std = @import("std");
const math = std.math;

const ptx = @import("kernel_utils.zig");
pub const panic = ptx.panic;

pub fn rgb2xyY(
    d_rgb: []f32,
    d_xyY: []f32,
    delta: f32,
) callconv(ptx.Kernel) void {
    const offset = ptx.getId_1D() * 3;
    if (offset >= d_rgb.len)
        return;

    const rgb = d_rgb[offset .. offset + 3];
    const r = rgb[0];
    const g = rgb[1];
    const b = rgb[2];

    const X = (r * 0.4124) + (g * 0.3576) + (b * 0.1805);
    const Y = (r * 0.2126) + (g * 0.7152) + (b * 0.0722);
    const Z = (r * 0.0193) + (g * 0.1192) + (b * 0.9505);
    const L = X + Y + Z;

    d_xyY[offset + 0] = X / L;
    d_xyY[offset + 1] = Y / L;
    // TODO: investigate why @log10 resolve to `log10f` and not some asm.
    const delta_p_Y = delta + Y;
    const log2_delta_p_Y = asm ("lg2.approx.f32 \t%[r], %[x];"
        : [r] "=r" (-> f32),
        : [x] "r" (delta_p_Y),
    );
    d_xyY[offset + 2] = log2_delta_p_Y / @log2(10.0);
}

pub const MinMax = struct { min: f32, max: f32 };
extern var _reduceMinmaxLum_sdata: opaque {} align(8) addrspace(.shared);

pub fn reduceMinmaxLum(d_xyY: []const f32, d_out: []MinMax) callconv(ptx.Kernel) void {
    var sdata = @ptrCast([*]addrspace(.shared) MinMax, &_reduceMinmaxLum_sdata);
    const myId = ptx.getId_1D();
    if (myId * 3 >= d_xyY.len) {
        return;
    }
    const tid = ptx.threadIdX();
    // load shared mem from global mem
    var minmax: MinMax = .{ .min = d_xyY[myId * 3 + 2], .max = d_xyY[myId * 3 + 2] };
    sdata[tid] = minmax;
    // make sure entire block is loaded!
    ptx.syncThreads();

    // do reduction in shared mem
    var s = ptx.blockDimX() / 2;
    while (s > 0) : (s >>= 1) {
        if (tid < s) {
            minmax.min = math.min(sdata[tid].min, sdata[tid + s].min);
            minmax.max = math.max(sdata[tid].max, sdata[tid + s].max);
            sdata[tid] = minmax;
        }
        ptx.syncThreads(); // make sure all adds at one stage are done!
    }

    // only thread 0 writes result for this block back to global mem
    if (tid == 0) {
        d_out[ptx.blockIdX()] = sdata[0];
    }
}

extern var _reduceMinmax_sdata: opaque {} align(8) addrspace(.shared);

pub fn reduceMinmax(d_in: []const MinMax, d_out: []MinMax) callconv(ptx.Kernel) void {
    var sdata = @ptrCast([*]addrspace(.shared) MinMax, &_reduceMinmax_sdata);
    const myId = ptx.getId_1D();
    if (myId >= d_in.len) {
        return;
    }
    const tid = ptx.threadIdX();
    // load shared mem from global mem
    sdata[tid] = d_in[myId];
    ptx.syncThreads(); // make sure entire block is loaded!

    var minmax: MinMax = undefined;
    // do reduction in shared mem
    var s = ptx.blockDimX() / 2;
    while (s > 0) : (s >>= 1) {
        if (tid < s) {
            minmax.min = math.min(sdata[tid].min, sdata[tid + s].min);
            minmax.max = math.max(sdata[tid].max, sdata[tid + s].max);
            sdata[tid] = minmax;
        }
        ptx.syncThreads(); // make sure all adds at one stage are done!
    }

    // only thread 0 writes result for this block back to global mem
    if (tid == 0) {
        d_out[ptx.blockIdX()] = sdata[0];
    }
}

pub fn lumHisto(d_bins: []u32, d_xyY: []const f32, lum_minmax: MinMax) callconv(ptx.Kernel) void {
    const myId = ptx.getId_1D();
    if (myId * 3 >= d_xyY.len) return;

    var lum = d_xyY[3 * myId + 2];
    const lum_range = lum_minmax.max - lum_minmax.min;
    // Normalize lum to [0, 1]
    lum = (lum - lum_minmax.min) / lum_range;
    const bin_count = d_bins.len;
    var bin = @floatToInt(u32, lum) * bin_count;
    bin = math.clamp(bin, 0, bin_count - 1);
    ptx.atomicAdd(&d_bins[bin], 1);
}

pub fn naiveComputeCdf(d_cdf: []f32, d_bins: []u32) callconv(ptx.Kernel) void {
    const myId = ptx.getId_1D();
    if (myId >= d_bins.len) return;

    // That's probably not the best way of using a GPU :-)
    var prefix_sum: u32 = 0;
    var i: u32 = 0;
    while (i < myId) : (i += 1) {
        prefix_sum += d_bins[i];
    }

    var total = prefix_sum;
    while (i < d_bins.len) : (i += 1) {
        total += d_bins[i];
    }
    d_cdf[myId] = @intToFloat(f32, prefix_sum) * @intToFloat(f32, total);
}

pub fn blellochCdf(d_cdf: []f32, d_bins: []u32) callconv(ptx.Kernel) void {
    // We need synchronization across all threads so only one block
    if (ptx.gridDimX() > 1) return;
    const n = d_bins.len;
    const tid = ptx.threadIdX();
    if (tid >= n) return;

    // Reduce
    var step: u32 = 1;
    while (step < n) : (step *= 2) {
        if (tid >= step and (n - 1 - tid) % (step * 2) == 0) {
            d_bins[tid] += d_bins[tid - step];
        }
        ptx.syncThreads();
    }

    const total = d_bins[n - 1];
    if (tid == n - 1) d_bins[tid] = 0;

    // Downsweep
    step /= 2;
    while (step > 0) : (step /= 2) {
        if (tid >= step and (n - 1 - tid) % (step * 2) == 0) {
            const left = d_bins[tid - step];
            const right = d_bins[tid];
            d_bins[tid] = left + right;
            d_bins[tid - step] = right;
        }
        ptx.syncThreads();
    }

    // Normalization
    d_cdf[tid] = @intToFloat(f32, d_bins[tid]) / @intToFloat(f32, total);
}

pub fn toneMap(d_xyY: []const f32, d_cdf_norm: []const f32, d_rgb_new: []f32, lum_minmax: MinMax, num_bins: u32) callconv(ptx.Kernel) void {
    const id = ptx.getId_1D();
    if (id * 3 >= d_xyY.len) return;

    const x = d_xyY[id * 3 + 0];
    const y = d_xyY[id * 3 + 1];
    const lum = d_xyY[id * 3 + 2];

    const lum_range = (lum_minmax.max - lum_minmax.min) / @intToFloat(f32, num_bins);
    var bin_index = @floatToInt(u32, (lum - lum_minmax.min) / lum_range);
    bin_index = math.clamp(bin_index, 0, num_bins - 1);
    const Y_new = d_cdf_norm[bin_index];

    const X_new = x * (Y_new / y);
    const Z_new = (1 - x - y) * (Y_new / y);

    d_rgb_new[id * 3 + 0] = (X_new * 3.2406) + (Y_new * -1.5372) + (Z_new * -0.4986);
    d_rgb_new[id * 3 + 1] = (X_new * -0.9689) + (Y_new * 1.8758) + (Z_new * 0.0415);
    d_rgb_new[id * 3 + 2] = (X_new * 0.0557) + (Y_new * -0.2040) + (Z_new * 1.0570);
}

comptime {
    if (ptx.is_device) {
        @export(rgb2xyY, .{ .name = "rgb2xyY" });
        @export(lumHisto, .{ .name = "lumHisto" });
        @export(reduceMinmaxLum, .{ .name = "reduceMinmaxLum" });
        @export(reduceMinmax, .{ .name = "reduceMinmax" });
        @export(naiveComputeCdf, .{ .name = "naiveComputeCdf" });
        @export(blellochCdf, .{ .name = "blellochCdf" });
        @export(toneMap, .{ .name = "toneMap" });
    }
}
