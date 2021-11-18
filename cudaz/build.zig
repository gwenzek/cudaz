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
    // const kernel = b.addObject("kernel", "cudaz/kernel.o");
    // kernel.linkLibC();
    // kernel.addLibPath("/usr/local/cuda/lib64");
    // kernel.linkSystemLibraryName("cudart");

    var tests = b.step("test", "Tests");
    const test_cuda = b.addTest("cudaz/cuda.zig");
    sdk.addCudaz(b, test_cuda, CUDA_PATH, "cudaz/test.cu");
    tests.dependOn(&test_cuda.step);

    const test_nvptx = b.addTest("cudaz/test_nvptx.zig");
    sdk.addCudazWithZigKernel(b, test_nvptx, CUDA_PATH, "cudaz/nvptx.zig");
    tests.dependOn(&test_nvptx.step);

    // TODO (Jan 2022): try zig build -ofmt=c (with master branch)
    // maybe we could write a kernel in Zig instead of cuda,
    // which will maybe simplify the type matching
    // const kernel_zig = b.addStaticLibrary("kernel_zig", "cudaz/kernel.zig");
    // kernel_zig.linkLibC();
    // kernel_zig.addLibPath("/usr/local/cuda/lib64");
    // kernel_zig.linkSystemLibraryName("cuda");
    // kernel_zig.addIncludeDir("/usr/local/cuda/include");
    // kernel_zig.setTarget(target);
    // kernel_zig.setBuildMode(mode);
    // kernel_zig.install();
}
