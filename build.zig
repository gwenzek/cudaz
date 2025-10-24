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
    cudaz.linkSystemLibrary("cuda", .{ .needed = true });

    const nvptx_device = b.addModule("nvptx_device", .{
        .root_source_file = b.path("src/nvptx.zig"),
        .target = target_nvptx,
        .optimize = optimize,
    });
    _ = nvptx_device; // used through registry.

    const nvptx_cpu = b.addModule("nvptx_cpu", .{
        .root_source_file = b.path("src/nvptx.zig"),
        .target = target,
        .optimize = optimize,
    });
    _ = nvptx_cpu; // used through registry.

    var tests = b.step("test", "Tests");
    const test_cuda = b.addTest(.{ .root_module = cudaz });
    const run_test_cuda = b.addRunArtifact(test_cuda);
    tests.dependOn(&run_test_cuda.step);

    const test_kernel_module = b.createModule(.{
        .root_source_file = b.path("src/test_kernel.zig"),
        .target = target,
        .optimize = optimize,
        .imports = &.{
            .{ .name = "cudaz", .module = cudaz },
        },
    });

    addCudaKernel(b, test_kernel_module, "generated_ptx");

    const test_kernel = b.addTest(.{ .root_module = test_kernel_module });
    const run_test_kernel = b.addRunArtifact(test_kernel);
    tests.dependOn(&run_test_kernel.step);
}

/// Given a regular Zig module, and module corresponding to a PTX kernel,
/// allows to `const generated_ptx = @embed(ptx_name);` from the Zig module.
pub fn addPtxEmbed(b: *std.Build, module: *std.Build.Module, kernel_import: std.Build.Module.Import) void {
    const kernel_obj = b.addObject(.{
        .name = "test_kernel",
        .root_module = kernel_import.module,
    });

    const kernel_ptx = b.createModule(.{ .root_source_file = kernel_obj.getEmittedAsm() });
    module.addImport(kernel_import.name, kernel_ptx);
}

/// Given a module containing the definition of a ptx kernel,
/// creates a ptx module, compiles it, and expose the PTX to the original module:
/// `const generated_ptx = @embed(ptx_name);`
///
/// Also automatically add nvptx imports
/// In case this is not desirable, you can create the kernel module manually,
/// Then call `addPtxEmbed`.
pub fn addCudaKernel(b: *std.Build, module: *std.Build.Module, ptx_name: []const u8) void {
    const nvptx_cpu = b.modules.get("nvptx_cpu") orelse @panic("nvptx_cpu not found");
    const nvptx_device = b.modules.get("nvptx_device") orelse @panic("nvptx_device not found");

    const kernel_module = b.createModule(.{
        .root_source_file = module.root_source_file,
        .target = nvptx_device.resolved_target,
        .optimize = .ReleaseFast,
        .imports = &.{
            .{ .name = "nvptx", .module = nvptx_device },
        },
    });

    module.addImport("nvptx", nvptx_cpu);
    addPtxEmbed(b, module, .{ .name = ptx_name, .module = kernel_module });
}
