const cudaz_options = @import("cudaz_options");

pub const cu = @cImport({
    @cInclude("cuda.h");
    @cInclude("cuda_globals.h");
    if (cudaz_options.cuda_kernel) {
        @cInclude(cudaz_options.kernel_name);
    }
});
