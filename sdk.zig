const std = @import("std");
const Build = std.Build;
const Step = std.Build.Step;

/// Can be one of "ptx" or "fatbin". "cubin" don't work, because it implies a main.
/// Fatbin contains device specialized assembly for all GPU arch supported
/// by this compiler. This provides faster startup time.
/// Ptx is a high-level text assembly that can converted to GPU specialized
/// instruction on loading.
/// We default to .ptx because it's more easy to distribute.
// TODO: make this a build option
const NVCC_OUTPUT_FORMAT = "ptx";
const ZIG_STAGE2 = "/home/guw/github/zig/build/stage3/bin/zig";
const SDK_ROOT = sdk_root() ++ "/";

/// For a given object:
///   1. Compile the given .cu file to a .ptx with `nvcc`
///   2. Add lib C
///   3. Add cuda headers, and cuda lib path
///   4. Add Cudaz package with the given .cu file that will get imported as C code.
///
/// The .ptx file will have the same base name than the executable
/// and will appear in zig-out/bin folder next to the executable.
/// In release mode the .ptx will be embedded inside the executable
/// so you can distribute it.
pub fn addCudazWithNvcc(
    b: *Build,
    exe: *Step.Compile,
    comptime cuda_dir: []const u8,
    comptime kernel_path: [:0]const u8,
) void {
    const outfile = std.mem.join(
        b.allocator,
        ".",
        &[_][]const u8{ exe.name, NVCC_OUTPUT_FORMAT },
    ) catch unreachable;
    // const kernel_ptx_path = std.fs.path.joinZ(
    //     b.allocator,
    //     &[_][]const u8{ b.exe_dir, outfile },
    // ) catch unreachable;
    // std.fs.cwd().makePath(b.exe_dir) catch @panic("Couldn't create zig-out output dir");

    // Use nvcc to compile the .cu file
    var nvcc = b.addSystemCommand(&[_][]const u8{
        cuda_dir ++ "/bin/nvcc",
        // In Zig spirit, promote warnings to errors.
        // "--Werror=all-warnings",
        "--display-error-number",
        // Don't require exactly gcc-11
        "-allow-unsupported-compiler",
        // TODO: try to use zig c++ here. For me it failed with:
        // zig: error: CUDA version is newer than the latest supported version 11.5 [-Werror,-Wunknown-cuda-version]
        // zig: error: cannot specify -o when generating multiple output files
        // "-ccbin",
        // "/usr/bin/gcc",
        "--" ++ NVCC_OUTPUT_FORMAT,
        "-I",
    });
    nvcc.addDirectoryArg(.{ .cwd_relative = SDK_ROOT ++ "src" });
    nvcc.addArg("-o");
    const kernel_ptx_path = nvcc.addOutputFileArg(outfile);
    nvcc.addFileArg(b.path(kernel_path));

    addCudazDeps(b, exe, cuda_dir, kernel_path, kernel_ptx_path);
}

/// Leverages stage2 to generate the .ptx from a .zig file.
/// This restricts the kernel to use the subset of Zig supported by stage2.
pub fn addCudazWithZigKernel(
    b: *Build,
    exe: *Step.Compile,
    comptime cuda_dir: []const u8,
    comptime kernel_path: [:0]const u8,
) void {
    const name = std.fs.path.basename(kernel_path);
    const zig_kernel = b.addModule(name, .{
        .root_source_file = b.path(kernel_path),
        .optimize = .ReleaseFast,
        .target = b.resolveTargetQuery(.{
            .cpu_arch = .nvptx64,
            .os_tag = .cuda,
            // .cpu_model = .{ .explicit = &sm32ptx75 },
            .cpu_model = .{ .explicit = &std.Target.nvptx.cpu.sm_32 },
            // .cpu_features_add = std.Target.nvptx.featureSet(&[_]std.Target.nvptx.Feature{
            //     .ptx75,
            // }),
        }),
    });

    const zig_kernel_obj = b.addObject(.{
        .name = name,
        .root_module = zig_kernel,
        // ReleaseFast because the panic handler leads to a
        // external dso_local constant with a name to complex for PTX
        // TODO: try to sanitize name in the NvPtx Zig backend.
    });

    b.getInstallStep().dependOn(&b.addInstallBinFile(
        zig_kernel_obj.getEmittedAsm(),
        std.mem.concat(b.allocator, u8, &.{ name, ".ptx" }) catch @panic("OOM"),
    ).step);
    // This yield an <error: unknown CPU: ''>
    // const sm32ptx75 = std.Target.Cpu.Model{
    //     .name = "sm_32",
    //     .llvm_name = "sm_32",
    //     .features = std.Target.nvptx.featureSet(&[_]std.Target.nvptx.Feature{
    //         .ptx75,
    //         .sm_32,
    //     }),
    // };

    // Adding the nvptx.zig package doesn't seem to work
    // const ptx_pkg = b.addModule("ptx", .{
    //     .root_source_file = .{ .cwd_relative = SDK_ROOT ++ "src/nvptx.zig" },
    // });
    // if (!std.mem.eql(u8, name, "nvptx.zig")) {
    //     // Don't include nvptx.zig in itself
    //     // TODO: find a more robust test
    //     zig_kernel_obj.addPackage(ptx_pkg);
    // }

    // Copy the .ptx next to the binary for easy review.
    // zig_kernel_obj.setOutputDir(b.exe_dir);
    // const kernel_ptx_path = std.mem.joinZ(
    //     b.allocator,
    //     "",
    //     &[_][]const u8{ b.exe_dir, "/", zig_kernel_obj.out_filename, ".ptx" },
    // ) catch unreachable;

    // // TODO: we should make this optional to allow compiling without a CUDA toolchain
    // const validate_ptx = b.addSystemCommand(
    //     &[_][]const u8{ cuda_dir ++ "/bin/ptxas", kernel_ptx_path },
    // );
    // validate_ptx.step.dependOn(&zig_kernel_obj.step);
    // exe.step.dependOn(&validate_ptx.step);
    exe.step.dependOn(&zig_kernel_obj.step);
    addCudazDeps(b, exe, cuda_dir, kernel_path, zig_kernel_obj.getEmittedAsm());
}

pub fn addCudazDeps(
    b: *Build,
    exe: *Step.Compile,
    comptime cuda_dir: []const u8,
    comptime kernel_path: [:0]const u8,
    kernel_ptx_path: Build.LazyPath,
) void {
    const kernel_dir = std.fs.path.dirname(kernel_path).?;
    // Add libc and cuda headers / lib, and our own .cu files
    exe.linkLibC();
    exe.addLibraryPath(.{ .cwd_relative = cuda_dir ++ "/lib64" });
    exe.linkSystemLibrary2("cuda", .{ .needed = true });
    // If nvidia-ptxjitcompiler is not found on your system,
    // check that there is a libnvidia-ptxjitcompiler.so, or create a symlink
    // to the right version.
    // We don't need to link ptxjit compiler, since it's loaded at runtime,
    // but this should warn the user that something is wrong.
    exe.linkSystemLibrary2("nvidia-ptxjitcompiler", .{ .needed = true });
    exe.addIncludePath(.{ .cwd_relative = SDK_ROOT ++ "src" });
    exe.addIncludePath(b.path(kernel_dir));
    exe.addSystemIncludePath(.{ .cwd_relative = cuda_dir ++ "/include" });

    // Add cudaz package with the kernel paths.
    const cudaz_options = b.addOptions();
    cudaz_options.step.dependOn(kernel_ptx_path.generated.file.step);
    cudaz_options.addOption([:0]const u8, "kernel_path", kernel_path);
    cudaz_options.addOption([]const u8, "kernel_name", std.fs.path.basename(kernel_path));
    cudaz_options.addOptionPath("kernel_ptx_path", kernel_ptx_path);
    cudaz_options.addOption([]const u8, "kernel_dir", kernel_dir);
    cudaz_options.addOption(bool, "cuda_kernel", std.mem.endsWith(u8, kernel_path, ".cu"));
    // Portable mode will embed the cuda modules inside the binary.
    // In debug mode we skip this step to have faster compilation.
    // But this makes the debug executable dependent on a hard-coded path.
    cudaz_options.addOption(bool, "portable", false);
    // TODO: restore previous behavior
    // cudaz_options.addOption(bool, "portable", exe.build_mode != .Debug);

    // const root_src = exe.root_module.root_source_file.?;
    // if (std.mem.eql(u8, root_src.path, "src/cuda.zig")) {
    //     // Don't include the package in itself
    //     exe.addOptions("cudaz_options", cudaz_options);
    // } else {
    // }
    var cudaz = b.addModule("cudaz", .{
        .root_source_file = .{ .cwd_relative = SDK_ROOT ++ "src/cuda.zig" },
    });
    cudaz.addIncludePath(.{ .cwd_relative = cuda_dir ++ "/include" });
    exe.root_module.addImport("cudaz", cudaz);
    exe.root_module.addOptions("cudaz_options", cudaz_options);
}

fn sdk_root() []const u8 {
    return std.fs.path.dirname(@src().file) orelse ".";
}

fn needRebuild(kernel_path: [:0]const u8, kernel_ptx_path: [:0]const u8) bool {
    var ptx_file = std.fs.openFileAbsoluteZ(kernel_ptx_path, .{}) catch return true;
    const ptx_stat = ptx_file.stat() catch return true;
    // detect empty .ptx files
    if (ptx_stat.size < 128) return true;

    var zig_file = (std.fs.cwd().openFileZ(kernel_path, .{}) catch return true);
    const zig_time = (zig_file.stat() catch return true).mtime;
    return zig_time >= ptx_stat.mtime + std.time.ns_per_s * 10;
}
