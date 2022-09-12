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

    // Pure
    const run_step = b.step("run", "Run the examples");
    const hw1 = addZigHomework(b, tests, "hw1");
    run_step.dependOn(&hw1.step);

    const hw2 = addZigHomework(b, tests, "hw2");
    run_step.dependOn(&hw2.step);

    const hw5 = addZigHomework(b, tests, "hw5");
    run_step.dependOn(&hw5.step);

    // CS344 lessons and home works
    const hw1_nvcc = addNvccHomework(b, tests, "hw1");
    addNvccLesson(b, "lesson2");
    const hw2_nvcc = addNvccHomework(b, tests, "hw2");
    addNvccLesson(b, "lesson3");
    const hw3_nvcc = addNvccHomework(b, tests, "hw3");
    const hw4_nvcc = addNvccHomework(b, tests, "hw4");
    // addZigLesson(b, "lesson5");

    const run_nvcc_step = b.step("run_nvcc", "Run the example");
    const run_hw1_nvcc = hw1_nvcc.run();
    run_hw1_nvcc.step.dependOn(b.getInstallStep());
    run_nvcc_step.dependOn(&run_hw1_nvcc.step);

    const run_hw2_nvcc = hw2_nvcc.run();
    run_hw2_nvcc.step.dependOn(b.getInstallStep());
    run_nvcc_step.dependOn(&run_hw2_nvcc.step);

    const run_hw3_nvcc = hw3_nvcc.run();
    run_hw3_nvcc.step.dependOn(b.getInstallStep());
    run_nvcc_step.dependOn(&run_hw3_nvcc.step);

    const run_hw4_nvcc = hw4_nvcc.run();
    run_hw4_nvcc.step.dependOn(b.getInstallStep());
    run_nvcc_step.dependOn(&run_hw4_nvcc.step);
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

fn addNvccHomework(b: *Builder, tests: *std.build.Step, comptime name: []const u8) *LibExeObjStep {
    const src = "src/nvcc/";
    const hw = b.addExecutable(name, src ++ name ++ ".zig");
    hw.setTarget(target);
    hw.setBuildMode(mode);

    cuda_sdk.addCudazWithNvcc(b, hw, CUDA_PATH, src ++ name ++ ".cu");
    addLodePng(hw);
    hw.install();

    const test_hw = b.addTest(src ++ name ++ ".zig");
    cuda_sdk.addCudazWithNvcc(b, test_hw, CUDA_PATH, src ++ name ++ ".cu");
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

fn addNvccLesson(b: *Builder, comptime name: []const u8) void {
    const src = "src/nvcc/";
    const lesson = b.addExecutable(name, src ++ name ++ ".zig");
    cuda_sdk.addCudazWithNvcc(b, lesson, CUDA_PATH, src ++ name ++ ".cu");
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
