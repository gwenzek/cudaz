const std = @import("std");

const cudaz_sdk = @import("cudaz");

// const LibExeObjStep = std.build.LibExeObjStep;
// const RunStep = std.build.RunStep;

var target: std.Build.ResolvedTarget = undefined;
var optimize: std.builtin.OptimizeMode = undefined;

pub fn build(b: *std.Build) void {
    target = b.standardTargetOptions(.{});
    optimize = b.standardOptimizeOption(.{});

    const tests = b.step("test", "Runs tests found inside homework code");

    // CS344 lessons and home works
    const run_step = b.step("run", "Run the examples");
    const hw1 = addZigHomework(b, tests, "hw1_pure");
    run_step.dependOn(&b.addRunArtifact(hw1).step);

    // addLesson(b, "lesson2");
    // const hw2 = addHomework(b, tests, "hw2");
    // addLesson(b, "lesson3");
    // const hw3 = addHomework(b, tests, "hw3");
    // const hw4 = addHomework(b, tests, "hw4");
    // // addZigLesson(b, "lesson5");

    const hw2 = addZigHomework(b, tests, "hw2_pure");
    b.getInstallStep().dependOn(&b.addInstallArtifact(hw2, .{}).step);
    run_step.dependOn(&b.addRunArtifact(hw2).step);

    // const run_hw3 = hw3.run();
    // run_hw3.step.dependOn(b.getInstallStep());
    // run_step.dependOn(&run_hw3.step);

    // const run_hw4 = hw4.run();
    // run_hw4.step.dependOn(b.getInstallStep());
    // run_step.dependOn(&run_hw4.step);

    // // Pure
    // const run_pure_step = b.step("run_pure", "Run the example");
    // const hw1_pure = addZigHomework(b, tests, "hw1_pure");
    // run_pure_step.dependOn(&hw1_pure.step);

    // const hw2_pure = addZigHomework(b, tests, "hw2_pure");
    // run_pure_step.dependOn(&hw2_pure.step);

    const hw5 = addZigHomework(b, tests, "hw5");
    run_step.dependOn(&b.addRunArtifact(hw5).step);
    b.getInstallStep().dependOn(&b.addInstallArtifact(hw5, .{}).step);
}

fn addLodePng(b: *std.Build, exe: *std.Build.Step.Compile) void {
    // TODO remove libc dependency
    exe.linkLibC();
    exe.addIncludePath(b.path("lodepng"));
    exe.addCSourceFile(.{ .file = b.path("lodepng/lodepng.c"), .flags = &.{"-DLODEPNG_COMPILE_ERROR_TEXT"} });
}

// fn addHomework(b: *std.Build, tests: *Step, comptime name: []const u8) *Step.Compile {
//     const hw = b.addModule(name, .{
//         .root_source_file = b.path("src/" ++ name ++ ".zig"),
//         .target = target,
//         .optimize = optimize,
//     });
//     const hw_exe = b.addExecutable(.{
//         .name = name ++ "_exe",
//         .root_module = hw,
//     });

//     cudaz.addCudazWithNvcc(b, hw_exe, CUDA_PATH, "src/" ++ name ++ ".cu");
//     addLodePng(b, hw);
//     // hw.install();

//     const test_hw = b.addTest(.{
//         .name = "test_" ++ name,
//         .root_source_file = b.path("src/" ++ name ++ ".zig"),
//         .target = target,
//         .optimize = optimize,
//     });
//     cudaz.addCudazWithNvcc(b, test_hw, CUDA_PATH, "src/" ++ name ++ ".cu");

//     tests.dependOn(&b.addRunArtifact(test_hw).step);

//     return hw;
// }

fn addZigHomework(b: *std.Build, tests: *std.Build.Step, name: []const u8) *std.Build.Step.Compile {
    _ = tests; // autofix
    const cudaz_pkg = b.dependency("cudaz", .{});
    const nvptx = cudaz_pkg.module("nvptx_device");

    const hw_zig = path(b, &.{ "src/", name, ".zig" });
    const hw_kernel_zig = path(b, &.{ "src/", name, "_kernel.zig" });

    const hw_kernel_device_mod = b.createModule(.{
        .root_source_file = hw_kernel_zig,
        .target = nvptx.resolved_target,
        .optimize = .ReleaseSafe,
        .imports = &.{
            .{ .name = "nvptx", .module = nvptx },
        },
        .strip = false,
    });

    const hw_ptx = cudaz_sdk.createPtx(b, hw_kernel_device_mod);
    const hw_ptx_install = b.addInstallFile(hw_ptx.root_source_file.?, join(b, &.{ name, ".ptx" }));
    b.getInstallStep().dependOn(&hw_ptx_install.step);

    const img_pkg = b.dependency("zigimg", .{});
    const hw_exe = b.addExecutable(.{
        .name = name,
        .root_module = b.createModule(.{
            .root_source_file = hw_zig,
            .target = target,
            .optimize = optimize,
            .imports = &.{
                .{ .name = "zigimg", .module = img_pkg.module("zigimg") },
                .{ .name = "cuda", .module = cudaz_pkg.module("cudaz") },
                .{ .name = join(b, &.{ name, "_ptx" }), .module = hw_ptx },
                .{ .name = "nvptx", .module = cudaz_pkg.module("nvptx_cpu") },
            },
        }),
        .use_llvm = true,
    });

    const run_step = b.step(name, join(b, &.{ "Run ", name }));
    run_step.dependOn(&b.addRunArtifact(hw_exe).step);

    return hw_exe;
}

// fn addLesson(b: *std.Build, comptime name: []const u8) void {
//     const lesson = b.addExecutable(name, "src/" ++ name ++ ".zig");
//     cudaz.addCudazWithNvcc(b, lesson, CUDA_PATH, "src/" ++ name ++ ".cu");
//     lesson.setTarget(target);
//     lesson.setBuildMode(mode);
//     lesson.install();

//     const run_lesson_step = b.step(name, "Run " ++ name);
//     const run_lesson = lesson.run();
//     run_lesson.step.dependOn(b.getInstallStep());
//     run_lesson_step.dependOn(&run_lesson.step);
// }

// fn addZigLesson(b: *std.Build, comptime name: []const u8) void {
//     const lesson = b.addExecutable(name, "src/" ++ name ++ ".zig");
//     cudaz.addCudazWithZigKernel(b, lesson, CUDA_PATH, "src/" ++ name ++ "_kernel.zig");
//     lesson.setTarget(target);
//     lesson.setBuildMode(mode);
//     lesson.install();

//     const run_lesson_step = b.step(name, "Run " ++ name);
//     const run_lesson = lesson.run();
//     run_lesson.step.dependOn(b.getInstallStep());
//     run_lesson_step.dependOn(&run_lesson.step);
// }

fn path(b: *std.Build, parts: []const []const u8) std.Build.LazyPath {
    return b.path(join(b, parts));
}

fn join(b: *std.Build, parts: []const []const u8) []const u8 {
    return std.mem.joinZ(b.allocator, "", parts) catch @panic("OOM");
}
