const std = @import("std");

const Builder = std.build.Builder;

fn addCudaz(b: *Builder, exe: *std.build.LibExeObjStep, comptime cuda_dir: []const u8) void {
    exe.linkLibC();
    exe.addLibPath(cuda_dir ++ "/lib64");
    exe.linkSystemLibraryName("cuda");
    exe.addIncludeDir(cuda_dir ++ "/include");
    exe.addIncludeDir("cudaz");
    exe.addPackagePath("cuda", "cudaz/cuda.zig");

    // TODO: allow to chose where to look at
    const nvcc = b.addSystemCommand(&[_][]const u8{
        cuda_dir ++ "/bin/nvcc",
        // In Zig spirit, promote warnings to errors.
        "--Werror=all-warnings",
        "--display-error-number",
        "--ptx",
        "cudaz/kernel.cu",
        "-o",
        "cudaz/kernel.ptx",
    });
    exe.step.dependOn(&nvcc.step);
}

fn addLibpng(exe: *std.build.LibExeObjStep) void {
    exe.linkSystemLibraryName("png");
    exe.addIncludeDir("/usr/include");
    // exe.addCSourceFile("/usr/include/png.h", &[_][]const u8{});
}

pub fn build(b: *Builder) void {
    // Standard target options allows the person running `zig build` to choose
    // what target to build for. Here we do not override the defaults, which
    // means any target is allowed, and the default is native. Other options
    // for restricting supported target set are available.
    const target = b.standardTargetOptions(.{});

    // Standard release options allow the person running `zig build` to select
    // between Debug, ReleaseSafe, ReleaseFast, and ReleaseSmall.
    const mode = b.standardReleaseOptions();

    // This isn't very useful, because we still have to declare `extern` symbols
    // const kernel = b.addObject("kernel", "cudaz/kernel.o");
    // kernel.linkLibC();
    // kernel.addLibPath("/usr/local/cuda/lib64");
    // kernel.linkSystemLibraryName("cudart");

    const exe = b.addExecutable("lesson3", "CS344/lesson3.zig");
    addCudaz(b, exe, "/usr/local/cuda");
    exe.addPackagePath("zigimg", "zigimg/zigimg.zig");
    addLibpng(exe);
    exe.setTarget(target);
    exe.setBuildMode(mode);
    exe.install();

    var tests = b.step("test", "Tests");
    const test_cuda = b.addTest("cudaz/cuda.zig");
    addCudaz(b, test_cuda, "/usr/local/cuda");
    tests.dependOn(&test_cuda.step);

    const test_png = b.addTest("CS344/png.zig");
    addCudaz(b, test_png, "/usr/local/cuda");
    test_png.addPackagePath("zigimg", "zigimg/zigimg.zig");
    addLibpng(test_png);
    tests.dependOn(&test_png.step);

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
    const run_cmd = exe.run();
    run_cmd.step.dependOn(b.getInstallStep());
    run_step.dependOn(&run_cmd.step);
}
