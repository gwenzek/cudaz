const ku = @import("kernel_utils.zig");
const cu = @import("cudaz").cu;
const Kernel = ku.Kernel;

pub export fn quicksort(d_out: []f32) callconv(.Kernel) void {
    // TODO: this should be a "no-divergence point", mark it as such
    if (d_out.len == 1) return;

    const tid = ku.getId_1D();

    // TODO actual run partitionning
    const pivot_id = d_out.len / 2;

    const sub_stream: cu.CUstream = undefined;
    if (tid == 0 and pivot_id > 0) {
        // Left stream
        cu.cudaStreamCreateWithFlags(&sub_stream, CU_STREAM_NON_BLOCKING);

        // quicksort<<<(n + 1023) / 1024, 1024, 0, sub_stream>>>(&d_out[pivot_id], n - pivot_id);
    }
    // TODO: right stream
    // Even better we can start the two streams at once.

}
