const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    const cudaz_pkg = b.dependency("cudaz", .{});
    const nvptx = cudaz_pkg.module("nvptx_device");

    const matmul_mod = b.createModule(.{
        .root_source_file = b.path("src/root.zig"),
        .target = target,
        .optimize = optimize,
        .imports = &.{
            .{ .name = "nvptx", .module = cudaz_pkg.module("nvptx_cpu") },
        },
    });

    const matmul_kernel_mod = b.createModule(.{
        .root_source_file = b.path("src/root.zig"),
        .target = nvptx.resolved_target,
        .optimize = .ReleaseFast,
        .imports = &.{
            .{ .name = "nvptx", .module = nvptx },
        },
    });

    const exe = b.addExecutable(.{
        .name = "matmul",
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/main.zig"),
            .target = target,
            .optimize = optimize,
            .imports = &.{
                .{ .name = "cudaz", .module = cudaz_pkg.module("cudaz") },
                .{ .name = "matmul", .module = matmul_mod },
            },
        }),
    });
    addPtxEmbed(b, exe.root_module, .{ .name = "matmul_ptx", .module = matmul_kernel_mod });

    b.installArtifact(exe);

    const run_step = b.step("run", "Run the app");
    const run_cmd = b.addRunArtifact(exe);
    run_step.dependOn(&run_cmd.step);

    run_cmd.step.dependOn(b.getInstallStep());
    // This allows the user to pass arguments to the application in the build
    // command itself, like this: `zig build run -- arg1 arg2 etc`
    if (b.args) |args| {
        run_cmd.addArgs(args);
    }
}

// Copied from cudaz/build.zig before I have a better solution.

/// Given a regular Zig module, and module corresponding to a PTX kernel,
/// allows to `const generated_ptx = @embed(ptx_name);` from the Zig module.
pub fn addPtxEmbed(b: *std.Build, module: *std.Build.Module, kernel_import: std.Build.Module.Import) void {
    const kernel_obj = b.addObject(.{
        .name = "matmul",
        .root_module = kernel_import.module,
    });

    const kernel_ptx = b.createModule(.{ .root_source_file = kernel_obj.getEmittedAsm() });
    module.addImport(kernel_import.name, kernel_ptx);
}
