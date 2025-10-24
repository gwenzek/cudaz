const std = @import("std");
const Build = std.Build;
const Step = std.Build.Step;

const cuda_sdk = @import("cudaz/sdk.zig");

const CUDA_PATH = "/usr/";

// const LibExeObjStep = std.build.LibExeObjStep;
// const RunStep = std.build.RunStep;

var target: Build.ResolvedTarget = undefined;
var optimize: std.builtin.OptimizeMode = undefined;

pub fn build(b: *Build) void {
    // Standard target options allows the person running `zig build` to choose
    // what target to build for. Here we do not override the defaults, which
    // means any target is allowed, and the default is native. Other options
    // for restricting supported target set are available.
    target = b.standardTargetOptions(.{});

    // Standard release options allow the person running `zig build` to select
    // between Debug, ReleaseSafe, ReleaseFast, and ReleaseSmall.
    optimize = b.standardOptimizeOption(.{});

    // const lode_png = b.addTranslateC(.{
    //     .root_source_file
    //     })

    const png = b.addModule("png", .{
        .root_source_file = b.path("src/png.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
    });

    const test_png = b.addTest(.{
        .name = "test_png",
        .root_module = png,
    });
    addLodePng(b, test_png);

    const tests = b.step("test", "Runs the ZML test suite");
    tests.dependOn(&b.addRunArtifact(test_png).step);

    // CS344 lessons and home works
    const run_step = b.step("run", "Run the example");
    const hw1 = addZigHomework(b, tests, "hw1_pure");

    // addLesson(b, "lesson2");
    // const hw2 = addHomework(b, tests, "hw2");
    // addLesson(b, "lesson3");
    // const hw3 = addHomework(b, tests, "hw3");
    // const hw4 = addHomework(b, tests, "hw4");
    // // addZigLesson(b, "lesson5");

    run_step.dependOn(&b.addRunArtifact(hw1).step);
    // const run_hw2 = hw2.run();
    // run_hw2.step.dependOn(b.getInstallStep());
    // run_step.dependOn(&run_hw2.step);

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

    // const hw5 = addZigHomework(b, tests, "hw5");
    // run_pure_step.dependOn(&hw5.step);
}

fn addLodePng(b: *Build, exe: *Step.Compile) void {
    // TODO remove libc dependency
    exe.linkLibC();
    exe.addIncludePath(b.path("lodepng"));
    exe.addCSourceFile(.{ .file = b.path("lodepng/lodepng.c"), .flags = &.{"-DLODEPNG_COMPILE_ERROR_TEXT"} });
}

// fn addHomework(b: *Build, tests: *Step, comptime name: []const u8) *Step.Compile {
//     const hw = b.addModule(name, .{
//         .root_source_file = b.path("src/" ++ name ++ ".zig"),
//         .target = target,
//         .optimize = optimize,
//     });
//     const hw_exe = b.addExecutable(.{
//         .name = name ++ "_exe",
//         .root_module = hw,
//     });

//     cuda_sdk.addCudazWithNvcc(b, hw_exe, CUDA_PATH, "src/" ++ name ++ ".cu");
//     addLodePng(b, hw);
//     // hw.install();

//     const test_hw = b.addTest(.{
//         .name = "test_" ++ name,
//         .root_source_file = b.path("src/" ++ name ++ ".zig"),
//         .target = target,
//         .optimize = optimize,
//     });
//     cuda_sdk.addCudazWithNvcc(b, test_hw, CUDA_PATH, "src/" ++ name ++ ".cu");

//     tests.dependOn(&b.addRunArtifact(test_hw).step);

//     return hw;
// }

fn addZigHomework(b: *Build, tests: *std.Build.Step, comptime name: []const u8) *std.Build.Step.Compile {
    const hw = b.addModule(name, .{
        .root_source_file = b.path("src/" ++ name ++ ".zig"),
        .target = target,
        .optimize = optimize,
    });
    const hw_exe = b.addExecutable(.{
        .name = name ++ "_exe",
        .root_module = hw,
    });

    cuda_sdk.addCudazWithZigKernel(b, hw_exe, CUDA_PATH, "src/" ++ name ++ "_kernel.zig");
    addLodePng(b, hw_exe);
    b.installArtifact(hw_exe);

    const test_hw = b.addTest(.{
        .name = name ++ "__test",
        .root_module = hw,
    });
    cuda_sdk.addCudazWithZigKernel(b, test_hw, CUDA_PATH, "src/" ++ name ++ "_kernel.zig");
    tests.dependOn(&test_hw.step);

    return hw_exe;
}

// fn addLesson(b: *Build, comptime name: []const u8) void {
//     const lesson = b.addExecutable(name, "src/" ++ name ++ ".zig");
//     cuda_sdk.addCudazWithNvcc(b, lesson, CUDA_PATH, "src/" ++ name ++ ".cu");
//     lesson.setTarget(target);
//     lesson.setBuildMode(mode);
//     lesson.install();

//     const run_lesson_step = b.step(name, "Run " ++ name);
//     const run_lesson = lesson.run();
//     run_lesson.step.dependOn(b.getInstallStep());
//     run_lesson_step.dependOn(&run_lesson.step);
// }

// fn addZigLesson(b: *Build, comptime name: []const u8) void {
//     const lesson = b.addExecutable(name, "src/" ++ name ++ ".zig");
//     cuda_sdk.addCudazWithZigKernel(b, lesson, CUDA_PATH, "src/" ++ name ++ "_kernel.zig");
//     lesson.setTarget(target);
//     lesson.setBuildMode(mode);
//     lesson.install();

//     const run_lesson_step = b.step(name, "Run " ++ name);
//     const run_lesson = lesson.run();
//     run_lesson.step.dependOn(b.getInstallStep());
//     run_lesson_step.dependOn(&run_lesson.step);
// }
