const std = @import("std");

const Builder = std.build.Builder;
const LibExeObjStep = std.build.LibExeObjStep;

// Can be one of "ptx" or "fatbin". "cubin" don't work, because it implies a main.
const NVCC_OUTPUT_FORMAT = "ptx";
const CUDA_PATH = "/usr/local/cuda";

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

    // This isn't very useful, because we still have to declare `extern` symbols
    // const kernel = b.addObject("kernel", "cudaz/kernel.o");
    // kernel.linkLibC();
    // kernel.addLibPath("/usr/local/cuda/lib64");
    // kernel.linkSystemLibraryName("cudart");

    var tests = b.step("test", "Tests");
    const test_cuda = b.addTest("cudaz/cuda.zig");
    addCudaz(b, test_cuda, CUDA_PATH, "cudaz/kernel.cu");
    tests.dependOn(&test_cuda.step);

    const test_nvptx = b.addTest("cudaz/test_nvptx.zig");
    addCudazWithZigKernel(b, test_nvptx, CUDA_PATH, "cudaz/nvptx.zig");
    tests.dependOn(&test_nvptx.step);

    const test_png = b.addTest("CS344/png.zig");
    test_png.addPackagePath("zigimg", "zigimg/zigimg.zig");
    addLibpng(test_png);
    tests.dependOn(&test_png.step);

    // CS344 lessons and home works
    const hw1 = addHomework(b, tests, "hw1");
    addLesson(b, "lesson2");
    const hw2 = addHomework(b, tests, "hw2");
    addLesson(b, "lesson3");
    const hw3 = addHomework(b, tests, "hw3");
    const hw4 = addHomework(b, tests, "hw4");
    _ = hw1;
    _ = hw2;
    _ = hw3;
    _ = hw4;

    // TODO (Jan 2022): try zig build -ofmt=c (with master branch)
    // maybe we could write a kernel in Zig instead of cuda,
    // which will maybe simplify the type matching
    // const kernel_zig = b.addStaticLibrary("kernel_zig", "cudaz/kernel.zig");
    // kernel_zig.linkLibC();
    // kernel_zig.addLibPath("/usr/local/cuda/lib64");
    // kernel_zig.linkSystemLibraryName("cuda");
    // kernel_zig.addIncludeDir("/usr/local/cuda/include");
    // kernel_zig.setTarget(target);
    // kernel_zig.setBuildMode(mode);
    // kernel_zig.install();

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
}

/// For a given object:
///   1. Compile the given .cu file to a .ptx
///   2. Add lib C
///   3. Add cuda headers, and cuda lib path
///   4. Add Cudaz package with the given .cu file that will get imported as C code.
///
/// The .ptx file will have the same base name than the object (which is supposed to be unique)
/// and will appear in zig-out/bin folder next to the executable.
// TODO: allow to embed the .ptx file in the executable (and use another format probably)
fn addCudaz(
    b: *Builder,
    exe: *LibExeObjStep,
    comptime cuda_dir: []const u8,
    comptime kernel_path: [:0]const u8,
) void {
    const outfile = std.mem.concat(
        b.allocator,
        u8,
        &[_][]const u8{ exe.name, "." ++ NVCC_OUTPUT_FORMAT },
    ) catch unreachable;
    const kernel_ptx_path = std.fs.path.joinZ(
        b.allocator,
        &[_][]const u8{ b.exe_dir, outfile },
    ) catch unreachable;

    // Use nvcc to compile the .cu file
    const nvcc = b.addSystemCommand(&[_][]const u8{
        cuda_dir ++ "/bin/nvcc",
        // In Zig spirit, promote warnings to errors.
        "--Werror=all-warnings",
        "--display-error-number",
        "--" ++ NVCC_OUTPUT_FORMAT,
        kernel_path,
        "-o",
        kernel_ptx_path,
    });
    exe.step.dependOn(&nvcc.step);
    addCudazDeps(b, exe, cuda_dir, kernel_path, kernel_ptx_path);
}

fn addCudazDeps(
    b: *Builder,
    exe: *LibExeObjStep,
    comptime cuda_dir: []const u8,
    comptime kernel_path: [:0]const u8,
    kernel_ptx_path: [:0]const u8,
) void {
    const kernel_dir = std.fs.path.dirname(kernel_path).?;
    // Add libc and cuda headers / lib, and our own .cu files
    exe.linkLibC();
    exe.addLibPath(cuda_dir ++ "/lib64");
    exe.linkSystemLibraryName("cuda");
    exe.addIncludeDir(cuda_dir ++ "/include");
    exe.addIncludeDir("cudaz");
    exe.addIncludeDir(kernel_dir);

    // Add cudaz package with the kernel paths.
    const cudaz_options = b.addOptions();
    cudaz_options.addOption([:0]const u8, "kernel_path", kernel_path);
    cudaz_options.addOption([]const u8, "kernel_name", std.fs.path.basename(kernel_path));
    cudaz_options.addOption([:0]const u8, "kernel_ptx_path", kernel_ptx_path);
    cudaz_options.addOption([]const u8, "kernel_dir", kernel_dir);
    cudaz_options.addOption(bool, "cuda_kernel", std.mem.endsWith(u8, kernel_path, ".cu"));

    const cudaz_pkg = std.build.Pkg{
        .name = "cudaz",
        .path = .{ .path = "cudaz/cuda.zig" },
        .dependencies = &[_]std.build.Pkg{
            .{ .name = "cudaz_options", .path = cudaz_options.getSource() },
        },
    };
    exe.addPackage(cudaz_pkg);
    // TODO: this is only needed for the tests in cuda.zig
    exe.addOptions("cudaz_options", cudaz_options);
}

fn addLibpng(exe: *LibExeObjStep) void {
    exe.linkLibC();
    exe.linkSystemLibraryName("png");
    // exe.addIncludeDir("/usr/include");
    // exe.addCSourceFile("/usr/include/png.h", &[_][]const u8{});
}

fn addHomework(b: *Builder, tests: *std.build.Step, comptime name: []const u8) *LibExeObjStep {
    const hw = b.addExecutable(name, "CS344/" ++ name ++ ".zig");
    hw.setTarget(target);
    hw.setBuildMode(mode);

    addCudaz(b, hw, CUDA_PATH, "CS344/" ++ name ++ ".cu");
    hw.addPackagePath("zigimg", "zigimg/zigimg.zig");
    addLibpng(hw);
    hw.install();

    const test_hw = b.addTest("CS344/" ++ name ++ ".zig");
    addCudaz(b, test_hw, CUDA_PATH, "CS344/" ++ name ++ ".cu");
    tests.dependOn(&test_hw.step);
    return hw;
}

fn addZigHomework(b: *Builder, tests: *std.build.Step, comptime name: []const u8) *LibExeObjStep {
    const hw = b.addExecutable(name, "CS344/" ++ name ++ ".zig");
    hw.setTarget(target);
    hw.setBuildMode(mode);

    addCudazWithZigKernel(b, hw, CUDA_PATH, "CS344/" ++ name ++ "_kernel.zig");
    hw.addPackagePath("zigimg", "zigimg/zigimg.zig");
    addLibpng(hw);
    hw.install();

    _ = tests;
    return hw;
}

fn addLesson(b: *Builder, comptime name: []const u8) void {
    const lesson = b.addExecutable(name, "CS344/" ++ name ++ ".zig");
    addCudaz(b, lesson, CUDA_PATH, "CS344/" ++ name ++ ".cu");
    lesson.setTarget(target);
    lesson.setBuildMode(mode);
    lesson.install();

    const run_lesson_step = b.step(name, "Run " ++ name);
    const run_lesson = lesson.run();
    run_lesson.step.dependOn(b.getInstallStep());
    run_lesson_step.dependOn(&run_lesson.step);
}

fn addCudazWithZigKernel(
    b: *Builder,
    exe: *LibExeObjStep,
    comptime cuda_dir: []const u8,
    comptime kernel_path: [:0]const u8,
) void {
    const name = std.fs.path.basename(kernel_path);
    const dummy_zig_kernel = b.addObject(name, kernel_path);
    dummy_zig_kernel.setTarget(.{ .cpu_arch = .nvptx64, .os_tag = .cuda });
    const kernel_o_path = std.fs.path.joinZ(
        b.allocator,
        &[_][]const u8{ b.exe_dir, dummy_zig_kernel.out_filename },
    ) catch unreachable;

    // Actually we need to use Stage2 here, so don't use the dummy obj,
    // and manually run this command.
    const emit_bin = std.mem.join(b.allocator, "=", &[_][]const u8{ "-femit-bin", kernel_o_path }) catch unreachable;
    const zig_kernel = b.addSystemCommand(&[_][]const u8{
        "../zig/stage2/bin/zig", "build-obj",    kernel_path,
        "-target",               "nvptx64-cuda", "-OReleaseSafe",
        emit_bin,
        // TODO make "--verbose-llvm-ir" optional
    });
    const kernel_ptx_path = std.mem.joinZ(b.allocator, "", &[_][]const u8{ kernel_o_path, ".ptx" }) catch unreachable;
    // TODO: Fix this during LLVM IR generation
    // I think we need to add LLVM annotations to mark functions as kernel
    // !nvvm.annotations = !{!1}
    // !1 = !{void (i8*)* @hello, !"kernel", i32 1}
    const fix_zig_kernel = b.addSystemCommand(&[_][]const u8{
        "sed",
        "-i",
        "s/.visible .func/.visible .entry/g",
        kernel_ptx_path,
    });
    fix_zig_kernel.step.dependOn(&zig_kernel.step);
    exe.step.dependOn(&fix_zig_kernel.step);

    addCudazDeps(b, exe, cuda_dir, kernel_path, kernel_ptx_path);
}
