const cudaz_options = @import("cudaz_options");

pub const cu = @cImport({
    @cInclude("cuda.h");
    @cInclude("cuda_globals.h");
    if (cudaz_options.cuda_kernel) {
        @cInclude(cudaz_options.kernel_name);
    }
});

// pub const kernels = if (cudaz_options.cuda_cimports) struct {} else @import(cudaz_options.kernel_name);
