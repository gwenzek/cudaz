const std = @import("std");

const Builder = std.build.Builder;

fn addCudaSrc(b: *Builder, exe: *std.build.LibExeObjStep, cuda_src: []const u8) void {
    const cache_root = std.fs.path.join(b.allocator, &[_][]const u8{ b.build_root, b.cache_root }) catch unreachable;

    const cuda_output = std.fs.path.join(b.allocator, &[_][]const u8{ cache_root, "cuda" }) catch unreachable;

    const cuda_gen = b.addSystemCommand(&[_][]const u8{ "nvcc", std.meta.tagName(backend), "--library", cuda_src, "-o", cuda_output });

    exe.step.dependOn(&cuda_gen.step);

    exe.addObject(cuda_output, cuda_src);
    exe.addIncludeDir(cache_root);
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

    const exe = b.addExecutable("main", "cudaz/cuda.zig");
    exe.linkLibC();
    exe.addLibPath("/usr/local/cuda/lib64");
    exe.linkSystemLibraryName("cuda");
    exe.addIncludeDir("/usr/local/cuda/include/");
    exe.addCSourceFile("cudaz/kernel.c", &[_][]const u8{});
    // exe.addObject(kernel);
    exe.setTarget(target);
    exe.setBuildMode(mode);
    exe.install();

    const tests = b.addTest("cudaz/cuda.zig");
    tests.linkLibC();
    tests.addLibPath("/usr/local/cuda/lib64");
    tests.linkSystemLibraryName("cuda");
    tests.addIncludeDir("/usr/local/cuda/include/");
    tests.addIncludeDir("cudaz");
    // tests.addCSourceFile("cudaz/kernel.h", &[_][]const u8{});
    // tests.addObject(kernel);

    b.step("test", "Tests").dependOn(&tests.step);
}
