const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

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
    cudaz.linkSystemLibrary("cuda", .{});

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
    tests.dependOn(&b.addRunArtifact(test_cuda).step);

    const test_device_module = b.createModule(.{
        .root_source_file = b.path("src/test.zig"),
        .target = nvptx_device.resolved_target,
        .optimize = .ReleaseFast,
        .imports = &.{
            .{ .name = "nvptx", .module = nvptx_device },
        },
    });

    const test_ptx = createPtx(b, test_device_module);
    const test_ptx_install = b.addInstallFile(test_ptx.root_source_file.?, "test.ptx");
    b.getInstallStep().dependOn(&test_ptx_install.step);

    const test_cpu_module = b.createModule(.{
        .root_source_file = b.path("src/test.zig"),
        .target = target,
        .optimize = optimize,
        .imports = &.{
            .{ .name = "cudaz", .module = cudaz },
            .{ .name = "nvptx", .module = nvptx_cpu },
            .{ .name = "generated_ptx", .module = test_ptx },
        },
    });

    const test_cpu = b.addTest(.{ .root_module = test_cpu_module });
    tests.dependOn(&b.addRunArtifact(test_cpu).step);
}

/// Given a device module corresponding, create the corresponding PTX.
/// The returned module can be imported like a regular module.
pub fn createPtx(b: *std.Build, device_module: *std.Build.Module) *std.Build.Module {
    const kernel_obj = b.addObject(.{
        .name = "ptx",
        .root_module = device_module,
    });

    return b.createModule(.{ .root_source_file = kernel_obj.getEmittedAsm() });
}
