const std = @import("std");

const sdk = @import("sdk.zig");

const CUDA_PATH = "/usr/";

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    // This isn't very useful, because we still have to declare `extern` symbols
    // const kernel = b.addObject("kernel", "src/kernel.o");
    // kernel.linkLibC();
    // kernel.addLibPath("/usr/local/cuda/lib64");
    // kernel.linkSystemLibraryName("cudart");

    const target_nvptx = b.resolveTargetQuery(.{
        .os_tag = .cuda,
        .cpu_arch = .nvptx64,
        .cpu_model = .{ .explicit = &std.Target.nvptx.cpu.sm_32 },
    });

    const cuda_h_translate = b.addTranslateC(.{
        .root_source_file = b.path("src/cuda.h"),
        .target = target,
        .optimize = optimize,
    });
    const cuda_h = cuda_h_translate.addModule("cuda_h");

    const cudaz = b.addModule("cudaz", .{
        .root_source_file = b.path("src/cuda.zig"),
        .target = target,
        .optimize = optimize,
        .imports = &.{
            .{ .name = "cuda_h", .module = cuda_h },
        },
        .link_libc = true,
    });
    cudaz.linkSystemLibrary("cuda", .{ .needed = true });

    const nvptx_device = b.addModule("nvptx_device", .{
        .root_source_file = b.path("src/nvptx.zig"),
        .target = target_nvptx,
        .optimize = optimize,
    });

    const nvptx_cpu = b.addModule("nvptx_cpu", .{
        .root_source_file = b.path("src/nvptx.zig"),
        .target = target,
        .optimize = optimize,
    });

    var tests = b.step("test", "Tests");
    const test_cuda = b.addTest(.{ .root_module = cudaz });
    const run_test_cuda = b.addRunArtifact(test_cuda);
    tests.dependOn(&run_test_cuda.step);

    const test_kernel_kernel = b.createModule(.{
        .root_source_file = b.path("src/test_kernel.zig"),
        .target = target_nvptx,
        // ReleaseFast because the panic handler leads to a
        // external dso_local constant with a name to complex for PTX
        // TODO: try to sanitize name in the NvPtx Zig backend.
        .optimize = .ReleaseFast,
        .imports = &.{
            .{ .name = "nvptx", .module = nvptx_device },
        },
    });

    const test_kernel_obj = b.addObject(.{
        .name = "test_kernel",
        .root_module = test_kernel_kernel,
    });

    const test_kernel_obj_module = b.createModule(.{ .root_source_file = test_kernel_obj.getEmittedAsm() });
    // test_kernel_obj.addWriteFile()
    const test_kernel_obj_options = b.addOptions();
    test_kernel_obj_options.addOptionPath("ptx_path", test_kernel_obj.getEmittedAsm());

    // test_kernel_module.addEmbedPath(test_kernel_obj.getEmittedAsm());
    const test_kernel_module = b.createModule(.{
        .root_source_file = b.path("src/test_kernel.zig"),
        .target = target,
        .optimize = optimize,
        .imports = &.{
            .{ .name = "cudaz", .module = cudaz },
            .{ .name = "nvptx", .module = nvptx_cpu },
            .{ .name = "generated_ptx", .module = test_kernel_obj_module },
        },
    });
    // test_kernel_module.addOp([]const u8, "bootloader_exe_path", b.fmt("\"{}\"", .{bootloader.getOutputPath()}));
    const test_kernel = b.addTest(.{ .root_module = test_kernel_module });
    const run_test_kernel = b.addRunArtifact(test_kernel);
    tests.dependOn(&run_test_kernel.step);
    // var tests_nvcc = b.step("test_nvcc", "Tests");
    // sdk.addCudazWithNvcc(b, test_cuda, CUDA_PATH, "src/test.cu");
    // tests_nvcc.dependOn(&test_cuda.step);

    // const test_nvptx_module = b.createModule(.{
    //     .root_source_file = b.path("src/test_nvptx.zig"),
    //     .imports = &.{
    //         .{ .name = "cudaz", .module = cudaz },
    //         .{ .name = "nvptx", .module = nvptx },
    //     },
    //     .target = target,
    //     .optimize = optimize,
    // });
    // const test_nvptx = b.addTest(.{
    //     .name = "test_nvptx",
    //     .root_module = test_nvptx_module,
    // });
    // sdk.addCudazWithZigKernel(b, test_nvptx, CUDA_PATH, "src/nvptx.zig");
    // tests.dependOn(&test_nvptx.step);
}
