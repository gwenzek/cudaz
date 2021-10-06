const std = @import("std");

const Builder = std.build.Builder;

fn addCudaz(
    b: *Builder,
    exe: *std.build.LibExeObjStep,
    comptime cuda_dir: []const u8,
    comptime kernel_path: [:0]const u8,
) void {
    const kernel_ptx_path = kernel_path ++ ".ptx";
    const kernel_dir = std.fs.path.dirname(kernel_path).?;

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
}

fn addLibpng(exe: *std.build.LibExeObjStep) void {
    exe.linkLibC();
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
    addCudaz(b, exe, "/usr/local/cuda", "CS344/lesson3.cu");
    exe.addPackagePath("zigimg", "zigimg/zigimg.zig");
    addLibpng(exe);
    exe.setTarget(target);
    exe.setBuildMode(mode);
    exe.install();

    var tests = b.step("test", "Tests");
    const test_cuda = b.addTest("cudaz/cuda.zig");
    addCudaz(b, test_cuda, "/usr/local/cuda", "cudaz/kernel.cu");
    tests.dependOn(&test_cuda.step);

    const test_png = b.addTest("CS344/png.zig");

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
