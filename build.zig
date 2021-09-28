const std = @import("std");

const Builder = std.build.Builder;

fn addCudaz(exe: *std.build.LibExeObjStep, comptime cuda_dir: []const u8) void {
    exe.linkLibC();
    exe.addLibPath(cuda_dir ++ "/lib64");
    exe.linkSystemLibraryName("cuda");
    exe.addIncludeDir(cuda_dir ++ "/include");
    exe.addIncludeDir("cudaz");
    exe.addPackagePath("cuda", "cudaz/cuda.zig");
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

    const exe = b.addExecutable("main", "cudaz/atomic_example.zig");
    addCudaz(exe, "/usr/local/cuda");
    // exe.addObject(kernel);
    exe.setTarget(target);
    exe.setBuildMode(mode);
    exe.install();

    const tests = b.addTest("cudaz/cuda.zig");
    addCudaz(tests, "/usr/local/cuda");
    // tests.addCSourceFile("cudaz/kernel.h", &[_][]const u8{});
    // tests.addObject(kernel);

    b.step("test", "Tests").dependOn(&tests.step);

    // TODO try zig build -ofmt=c (with master branch)
    // maybe we could write a kernel in Zig instead of cuda,
    // which will maybe simplify the type matching

    const run_step = b.step("run", "Run the example");
    const run_cmd = exe.run();
    run_cmd.step.dependOn(b.getInstallStep());
    run_step.dependOn(&run_cmd.step);
}
