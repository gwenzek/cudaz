const std = @import("std");

const cudaz_sdk = @import("cudaz");

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

    const matmul_device_mod = b.createModule(.{
        .root_source_file = b.path("src/root.zig"),
        .target = nvptx.resolved_target,
        .optimize = .ReleaseFast,
        .imports = &.{
            .{ .name = "nvptx", .module = nvptx },
        },
    });

    const matmul_ptx = cudaz_sdk.createPtx(b, matmul_device_mod);

    // `zig build install` will materialize the .ptx in `zig-out` directory
    const matmul_ptx_install = b.addInstallFile(matmul_ptx.root_source_file.?, "matmul.ptx");
    b.getInstallStep().dependOn(&matmul_ptx_install.step);

    const exe = b.addExecutable(.{
        .name = "matmul",
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/main.zig"),
            .target = target,
            .optimize = optimize,
            .imports = &.{
                .{ .name = "cudaz", .module = cudaz_pkg.module("cudaz") },
                .{ .name = "matmul", .module = matmul_mod },
                .{ .name = "matmul_ptx", .module = matmul_ptx },
            },
        }),
    });

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
