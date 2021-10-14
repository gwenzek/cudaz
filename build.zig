const std = @import("std");

const Builder = std.build.Builder;
const LibExeObjStep = std.build.LibExeObjStep;

const CUDA_PATH = "/usr/local/cuda";

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

    var tests = b.step("test", "Tests");
    const test_cuda = b.addTest("cudaz/cuda.zig");
    addCudaz(b, test_cuda, CUDA_PATH, "cudaz/kernel.cu");
    tests.dependOn(&test_cuda.step);

    const test_png = b.addTest("CS344/png.zig");
    test_png.addPackagePath("zigimg", "zigimg/zigimg.zig");
    addLibpng(test_png);
    tests.dependOn(&test_png.step);

    // CS344 lessons and home works
    const hw1 = addHomework(b, tests, "hw1");
    const hw2 = addHomework(b, tests, "hw2");
    const hw3 = addHomework(b, tests, "hw3");
    const hw4 = addHomework(b, tests, "hw4");
    _ = hw1;
    _ = hw2;
    _ = hw3;

    const lesson3 = b.addExecutable("lesson3", "CS344/lesson3.zig");
    addCudaz(b, lesson3, CUDA_PATH, "CS344/lesson3.cu");
    lesson3.setTarget(target);
    lesson3.setBuildMode(mode);
    lesson3.install();

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
    // const run_hw1 = hw1.run();
    // run_hw1.step.dependOn(b.getInstallStep());
    // run_step.dependOn(&run_hw1.step);

    // const run_hw2 = hw2.run();
    // run_hw2.step.dependOn(b.getInstallStep());
    // run_step.dependOn(&run_hw2.step);

    // const run_lesson3 = lesson3.run();
    // run_lesson3.step.dependOn(b.getInstallStep());
    // run_step.dependOn(&run_lesson3.step);

    // const run_hw3 = hw3.run();
    // run_hw3.step.dependOn(b.getInstallStep());
    // run_step.dependOn(&run_hw3.step);

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
    const kernel_dir = std.fs.path.dirname(kernel_path).?;
    const kernel_ptx_path = std.fs.path.joinZ(
        b.allocator,
        &[_][]const u8{
            b.exe_dir,
            std.mem.concat(b.allocator, u8, &[_][]const u8{ exe.name, ".ptx" }) catch unreachable,
        },
    ) catch unreachable;

    // Use nvcc to compile the .cu file
    const nvcc = b.addSystemCommand(&[_][]const u8{
        cuda_dir ++ "/bin/nvcc",
        // In Zig spirit, promote warnings to errors.
        "--Werror=all-warnings",
        "--display-error-number",
        "--ptx",
        kernel_path,
        "-o",
        kernel_ptx_path,
    });
    exe.step.dependOn(&nvcc.step);

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
    addCudaz(b, hw, CUDA_PATH, "CS344/" ++ name ++ ".cu");
    hw.addPackagePath("zigimg", "zigimg/zigimg.zig");
    addLibpng(hw);
    hw.install();

    const test_hw = b.addTest("CS344/" ++ name ++ ".zig");
    addCudaz(b, test_hw, CUDA_PATH, "CS344/" ++ name ++ ".cu");
    tests.dependOn(&test_hw.step);
    return hw;
}
