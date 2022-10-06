const std = @import("std");
const sdk = @import("sdk.zig");
const CUDA_PATH = "/usr/local/cuda";

const Builder = std.build.Builder;
const LibExeObjStep = std.build.LibExeObjStep;

pub fn build(b: *Builder) void {
    const mode = b.standardReleaseOptions();

    var cuda_path = std.os.getenv("CUDA_HOME");
    if (cuda_path == null) cuda_path = CUDA_PATH;

    var tests_nvcc = b.step("test_nvcc", "Tests");
    const test_cuda = b.addTest("src/cuda.zig");
    test_cuda.setBuildMode(mode);
    sdk.addCudazWithNvcc(b, test_cuda, cuda_path.?, "src/test.cu");
    tests_nvcc.dependOn(&test_cuda.step);

    var tests = b.step("test", "Tests");

    const test_zig = b.addTest("src/test.zig");
    test_zig.setBuildMode(mode);
    sdk.addCudazWithZigKernel(b, test_zig, cuda_path.?, "src/test_kernel.zig");
    tests.dependOn(&test_zig.step);

    const test_ptx = b.addTest("src/cuda.zig");
    test_ptx.setBuildMode(mode);
    sdk.addCudazWithPtxKernel(b, test_ptx, cuda_path.?, "src/test_kernel.ptx");
    tests.dependOn(&test_ptx.step);
}
