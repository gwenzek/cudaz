const std = @import("std");

const Builder = std.build.Builder;
const LibExeObjStep = std.build.LibExeObjStep;

/// Can be one of "ptx" or "fatbin". "cubin" don't work, because it implies a main.
/// Fatbin contains device specialized assembly for all GPU arch supported
/// by this compiler. This provides faster startup time.
/// Ptx is a high-level text assembly that can converted to GPU specialized
/// instruction on loading.
/// We default to .ptx because it's more easy to distribute.
// TODO: make this a build option
const NVCC_OUTPUT_FORMAT = "ptx";
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
    b: *Builder,
    exe: *LibExeObjStep,
    cuda_dir: []const u8,
    comptime kernel_path: [:0]const u8,
) void {
    const outfile = std.mem.join(
        b.allocator,
        ".",
        &[_][]const u8{ exe.name, NVCC_OUTPUT_FORMAT },
    ) catch unreachable;
    const kernel_ptx_path = std.fs.path.joinZ(
        b.allocator,
        &[_][]const u8{ b.exe_dir, outfile },
    ) catch unreachable;
    std.fs.cwd().makePath(b.exe_dir) catch @panic("Couldn't create zig-out output dir");

    const nvcc_bin = std.fs.path.join(b.allocator, &[_][]const u8{ cuda_dir, "bin/nvcc" }) catch unreachable;
    defer b.allocator.free(nvcc_bin);

    // Use nvcc to compile the .cu file
    const nvcc = b.addSystemCommand(&[_][]const u8{
        nvcc_bin,
        // In Zig spirit, promote warnings to errors.
        "--Werror=all-warnings",
        "--display-error-number",
        // Don't require exactly gcc-11
        "-allow-unsupported-compiler",
        // TODO: try to use zig c++ here. For me it failed with:
        // zig: error: CUDA version is newer than the latest supported version 11.5 [-Werror,-Wunknown-cuda-version]
        // zig: error: cannot specify -o when generating multiple output files
        "-ccbin",
        "/usr/bin/gcc",
        "--" ++ NVCC_OUTPUT_FORMAT,
        "-I",
        SDK_ROOT ++ "src",
        kernel_path,
        "-o",
        kernel_ptx_path,
    });
    exe.step.dependOn(&nvcc.step);
    addCudazDeps(b, exe, cuda_dir, kernel_path, kernel_ptx_path);
}

/// Leverages stage2 to generate the .ptx from a .zig file.
/// This restricts the kernel to use the subset of Zig supported by stage2.
pub fn addCudazWithZigKernel(
    b: *Builder,
    exe: *LibExeObjStep,
    cuda_dir: []const u8,
    comptime kernel_path: [:0]const u8,
) void {
    const name = std.fs.path.basename(kernel_path);
    const zig_kernel = b.addObject(name, kernel_path);
    zig_kernel.setTarget(.{
        .cpu_arch = .nvptx64,
        .os_tag = .cuda,
        .cpu_model = .{ .explicit = &std.Target.nvptx.cpu.sm_32 },
        .cpu_features_add = std.Target.nvptx.featureSet(&[_]std.Target.nvptx.Feature{
            .ptx75,
        }),
    });
    // * Debug doesn't compile, because some `u2` global will end-up in the ptx
    // which don't exist in PTX. It only seems to happen with globals.
    // A local `u2` should get lowered to a `u8`. This looks like a bug in LLVM backend.
    // TODO open bug with bugs/u2_in_ptx/u2_global.{ll, ptx}
    // * ReleaseSafe does compile but the panic handler are calling themselves recursively,
    // preventing the LLVM-PTX backend to inline the calls.
    // Later when ptxjit is compiling the ptx for the GPU it will think that the kernel
    // requires a crazy high number of local memory, making it impossible to run a kernel that also
    // need shared memory (local and shared use the same physical memory).
    // zig_kernel.setBuildMode(if (exe.build_mode == .Debug) .ReleaseSafe else exe.build_mode);
    zig_kernel.setBuildMode(.ReleaseFast);
    // Adding the nvptx.zig package doesn't seem to work
    const ptx_pkg = std.build.Pkg{
        .name = "ptx",
        .source = .{ .path = SDK_ROOT ++ "src/nvptx.zig" },
    };
    if (!std.mem.eql(u8, name, "nvptx.zig")) {
        // Don't include nvptx.zig in itself
        // TODO: find a more robust test
        zig_kernel.addPackage(ptx_pkg);
    }

    // Copy the .ptx next to the binary for easy review.
    zig_kernel.setOutputDir(b.exe_dir);
    const kernel_ptx_path = std.mem.joinZ(
        b.allocator,
        "",
        &[_][]const u8{ b.exe_dir, "/", zig_kernel.out_filename, ".ptx" },
    ) catch unreachable;

    const validate_ptx = validate_ptx_file(b, zig_kernel, cuda_dir, kernel_ptx_path);
    exe.step.dependOn(&validate_ptx.step);

    addCudazDeps(b, exe, cuda_dir, kernel_path, kernel_ptx_path);
}

/// Loads a kernel written in .ptx direcly or with another toolchain.
pub fn addCudazWithPtxKernel(
    b: *Builder,
    exe: *LibExeObjStep,
    cuda_dir: []const u8,
    comptime kernel_path: [:0]const u8,
) void {
    addCudazDeps(b, exe, cuda_dir, kernel_path, kernel_path);
}

pub fn addCudazDeps(
    b: *Builder,
    exe: *LibExeObjStep,
    cuda_dir: []const u8,
    comptime kernel_path: [:0]const u8,
    kernel_ptx_path: [:0]const u8,
) void {
    const kernel_dir = std.fs.path.dirname(kernel_path).?;
    // Add libc and cuda headers / lib, and our own .cu files
    exe.linkLibC();
    const cuda_lib64 = std.fs.path.join(b.allocator, &[_][]const u8{ cuda_dir, "lib64" }) catch unreachable;
    defer b.allocator.free(cuda_lib64);
    exe.addLibraryPath(cuda_lib64);
    exe.linkSystemLibraryNeeded("cuda");
    // If nvidia-ptxjitcompiler is not found on your system,
    // check that there is a libnvidia-ptxjitcompiler.so, or create a symlink
    // to the right version.
    // We don't need to link ptxjit compiler, since it's loaded at runtime,
    // but this should warn the user that something is wrong.
    exe.linkSystemLibraryNeeded("nvidia-ptxjitcompiler");
    exe.addIncludePath(SDK_ROOT ++ "src");
    const cuda_include = std.fs.path.join(b.allocator, &[_][]const u8{ cuda_dir, "include" }) catch unreachable;
    defer b.allocator.free(cuda_include);
    exe.addIncludePath(cuda_include);
    exe.addIncludePath(kernel_dir);

    // Add cudaz package with the kernel paths.
    const cudaz_options = b.addOptions();
    cudaz_options.addOption([:0]const u8, "kernel_path", kernel_path);
    cudaz_options.addOption([]const u8, "kernel_name", std.fs.path.basename(kernel_path));
    cudaz_options.addOption([:0]const u8, "kernel_ptx_path", kernel_ptx_path);
    cudaz_options.addOption([]const u8, "kernel_dir", kernel_dir);
    cudaz_options.addOption(bool, "cuda_kernel", std.mem.endsWith(u8, kernel_path, ".cu"));
    // Portable mode will embed the cuda modules inside the binary.
    // In debug mode we skip this step to have faster compilation.
    // But this makes the debug executable dependent on a hard-coded path.
    // TODO: currently this is forbidden by Zig. See https://github.com/ziglang/zig/issues/6662
    cudaz_options.addOption(bool, "portable", false);

    const cudaz_pkg = std.build.Pkg{
        .name = "cudaz",
        .source = .{ .path = SDK_ROOT ++ "src/cuda.zig" },
        .dependencies = &[_]std.build.Pkg{
            .{ .name = "cudaz_options", .source = cudaz_options.getSource() },
        },
    };
    const root_src = exe.root_src.?;
    if (std.mem.eql(u8, root_src.path, "src/cuda.zig")) {
        // Don't include the package in itself
        exe.addOptions("cudaz_options", cudaz_options);
    } else {
        exe.addPackage(cudaz_pkg);
    }
}

pub fn addCudazNoKernel(b: *Builder, exe: *LibExeObjStep) void {
    addCudazDeps(b, exe, "/cuda_dir/", "./nope.zig", "./nope.ptx");
}

fn sdk_root() []const u8 {
    return std.fs.path.dirname(@src().file).?;
}

/// Uses ptxas to validate the file
fn validate_ptx_file(
    b: *Builder,
    zig_kernel: *LibExeObjStep,
    cuda_dir: []const u8,
    kernel_ptx_path: []const u8,
) *std.build.RunStep {
    const suppress_stack_size_warning = "--suppress-stack-size-warning";

    const ptxas_bin = std.fs.path.join(b.allocator, &[_][]const u8{ cuda_dir, "bin/ptxas" }) catch unreachable;
    defer b.allocator.free(ptxas_bin);
    var full_ptxas_cmd = [_][]const u8{
        ptxas_bin,
        kernel_ptx_path,
        // This might be a little bit aggressive
        "--warning-as-error",
        "--warn-on-double-precision-use",
        "--warn-on-spills",
    };
    if (zig_kernel.build_mode == .ReleaseSafe or zig_kernel.build_mode == .Debug) {
        // The default panicOutOfBound handler always trigger the register spill and stack-size warnings.
        // TODO: fix panicExtra to not call itself
        full_ptxas_cmd[4] = suppress_stack_size_warning;
    }
    // TODO: we should make this optional to allow compiling without a CUDA toolchain
    const validate_ptx = b.addSystemCommand(&full_ptxas_cmd);
    validate_ptx.step.dependOn(&zig_kernel.step);
    return validate_ptx;
}
