const std = @import("std");
const sdk = @import("sdk.zig");
const CUDA_PATH = "/usr/local/cuda";

const Builder = std.build.Builder;
const LibExeObjStep = std.build.LibExeObjStep;

pub fn build(b: *Builder) void {
    // Standard target options allows the person running `zig build` to choose
    // what target to build for. Here we do not override the defaults, which
    // means any target is allowed, and the default is native. Other options
    // for restricting supported target set are available.
    // const target = b.standardTargetOptions(.{});

    // Standard release options allow the person running `zig build` to select
    // between Debug, ReleaseSafe, ReleaseFast, and ReleaseSmall.
    // const mode = b.standardReleaseOptions();

    // This isn't very useful, because we still have to declare `extern` symbols
    // const kernel = b.addObject("kernel", "src/kernel.o");
    // kernel.linkLibC();
    // kernel.addLibPath("/usr/local/cuda/lib64");
    // kernel.linkSystemLibraryName("cudart");
    var cuda_path = std.os.getenv("CUDA_HOME");
    if (cuda_path == null) cuda_path = &CUDA_PATH;

    var tests_nvcc = b.step("test_nvcc", "Tests");
    const test_cuda = b.addTest("src/cuda.zig");
    sdk.addCudazWithNvcc(b, test_cuda, cuda_path, "src/test.cu");
    tests_nvcc.dependOn(&test_cuda.step);

    var tests = b.step("test", "Tests");
    const test0 = b.addTest("src/test.zig");
    sdk.addCudazWithZigKernel(b, test0, cuda_path, "src/test_kernel.zig");
    tests.dependOn(&test0.step);
}
