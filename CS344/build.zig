const std = @import("std");
const cuda_sdk = @import("cudaz/sdk.zig");
const CUDA_PATH = "/usr/local/cuda";

const Builder = std.build.Builder;
const LibExeObjStep = std.build.LibExeObjStep;
const RunStep = std.build.RunStep;

var target: std.zig.CrossTarget = undefined;
var mode: std.builtin.Mode = undefined;

pub fn build(b: *Builder) void {
    // Standard target options allows the person running `zig build` to choose
    // what target to build for. Here we do not override the defaults, which
    // means any target is allowed, and the default is native. Other options
    // for restricting supported target set are available.
    target = b.standardTargetOptions(.{});

    // Standard release options allow the person running `zig build` to select
    // between Debug, ReleaseSafe, ReleaseFast, and ReleaseSmall.
    mode = b.standardReleaseOptions();

    var tests = b.step("test", "Tests");

    const test_png = b.addTest("src/png.zig");
    addLodePng(test_png);
    tests.dependOn(&test_png.step);

    // CS344 lessons and home works
    const hw1 = addHomework(b, tests, "hw1");
    addLesson(b, "lesson2");
    const hw2 = addHomework(b, tests, "hw2");
    addLesson(b, "lesson3");
    const hw3 = addHomework(b, tests, "hw3");
    const hw4 = addHomework(b, tests, "hw4");
    // addZigLesson(b, "lesson5");

    const run_step = b.step("run", "Run the example");
    const run_hw1 = hw1.run();
    run_hw1.step.dependOn(b.getInstallStep());
    run_step.dependOn(&run_hw1.step);

    const run_hw2 = hw2.run();
    run_hw2.step.dependOn(b.getInstallStep());
    run_step.dependOn(&run_hw2.step);

    const run_hw3 = hw3.run();
    run_hw3.step.dependOn(b.getInstallStep());
    run_step.dependOn(&run_hw3.step);

    const run_hw4 = hw4.run();
    run_hw4.step.dependOn(b.getInstallStep());
    run_step.dependOn(&run_hw4.step);

    // Pure
    const run_pure_step = b.step("run_pure", "Run the example");
    const hw1_pure = addZigHomework(b, tests, "hw1_pure");
    run_pure_step.dependOn(&hw1_pure.step);

    const hw2_pure = addZigHomework(b, tests, "hw2_pure");
    run_pure_step.dependOn(&hw2_pure.step);

    // const hw5 = addZigHomework(b, tests, "hw5");
    // run_pure_step.dependOn(&hw5.step);
}

fn addLodePng(exe: *LibExeObjStep) void {
    // TODO remove libc dependency
    exe.linkLibC();
    const lodepng_flags = [_][]const u8{
        "-DLODEPNG_COMPILE_ERROR_TEXT",
    };
    exe.addIncludeDir("lodepng/");
    exe.addCSourceFile("lodepng/lodepng.c", &lodepng_flags);
}

fn addHomework(b: *Builder, tests: *std.build.Step, comptime name: []const u8) *LibExeObjStep {
    const hw = b.addExecutable(name, "src/" ++ name ++ ".zig");
    hw.setTarget(target);
    hw.setBuildMode(mode);

    cuda_sdk.addCudazWithNvcc(b, hw, CUDA_PATH, "src/" ++ name ++ ".cu");
    addLodePng(hw);
    hw.install();

    const test_hw = b.addTest("src/" ++ name ++ ".zig");
    cuda_sdk.addCudazWithNvcc(b, test_hw, CUDA_PATH, "src/" ++ name ++ ".cu");
    tests.dependOn(&test_hw.step);
    return hw;
}

fn addZigHomework(b: *Builder, tests: *std.build.Step, comptime name: []const u8) *RunStep {
    const hw = b.addExecutable(name, "src/" ++ name ++ ".zig");
    hw.setTarget(target);
    hw.setBuildMode(mode);

    cuda_sdk.addCudazWithZigKernel(b, hw, CUDA_PATH, "src/" ++ name ++ "_kernel.zig");
    addLodePng(hw);
    hw.install();
    const hw_run = hw.run();
    hw_run.step.dependOn(b.getInstallStep());

    const test_hw = b.addTest("src/" ++ name ++ ".zig");
    cuda_sdk.addCudazWithZigKernel(b, test_hw, CUDA_PATH, "src/" ++ name ++ "_kernel.zig");
    tests.dependOn(&test_hw.step);

    return hw_run;
}

fn addLesson(b: *Builder, comptime name: []const u8) void {
    const lesson = b.addExecutable(name, "src/" ++ name ++ ".zig");
    cuda_sdk.addCudazWithNvcc(b, lesson, CUDA_PATH, "src/" ++ name ++ ".cu");
    lesson.setTarget(target);
    lesson.setBuildMode(mode);
    lesson.install();

    const run_lesson_step = b.step(name, "Run " ++ name);
    const run_lesson = lesson.run();
    run_lesson.step.dependOn(b.getInstallStep());
    run_lesson_step.dependOn(&run_lesson.step);
}

fn addZigLesson(b: *Builder, comptime name: []const u8) void {
    const lesson = b.addExecutable(name, "src/" ++ name ++ ".zig");
    cuda_sdk.addCudazWithZigKernel(b, lesson, CUDA_PATH, "src/" ++ name ++ "_kernel.zig");
    lesson.setTarget(target);
    lesson.setBuildMode(mode);
    lesson.install();

    const run_lesson_step = b.step(name, "Run " ++ name);
    const run_lesson = lesson.run();
    run_lesson.step.dependOn(b.getInstallStep());
    run_lesson_step.dependOn(&run_lesson.step);
}
