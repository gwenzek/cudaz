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

    var tests = b.step("test", "Tests");
    const test_cuda = b.addTest("src/cuda.zig");
    sdk.addCudaz(b, test_cuda, CUDA_PATH, "src/test.cu");
    tests.dependOn(&test_cuda.step);

    const test_nvptx = b.addTest("src/test_nvptx.zig");
    sdk.addCudazWithZigKernel(b, test_nvptx, CUDA_PATH, "src/nvptx.zig");
    tests.dependOn(&test_nvptx.step);
}
