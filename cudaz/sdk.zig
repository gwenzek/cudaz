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
const ZIG_STAGE2 = "zig2";
const SDK_ROOT = sdk_root() ++ "/";

/// For a given object:
///   1. Compile the given .cu file to a .ptx
///   2. Add lib C
///   3. Add cuda headers, and cuda lib path
///   4. Add Cudaz package with the given .cu file that will get imported as C code.
///
/// The .ptx file will have the same base name than the executable
/// and will appear in zig-out/bin folder next to the executable.
/// In release mode the .ptx will be embedded inside the executable
/// so you can distribute it.
pub fn addCudaz(
    b: *Builder,
    exe: *LibExeObjStep,
    comptime cuda_dir: []const u8,
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

    // Use nvcc to compile the .cu file
    const nvcc = b.addSystemCommand(&[_][]const u8{
        cuda_dir ++ "/bin/nvcc",
        // In Zig spirit, promote warnings to errors.
        "--Werror=all-warnings",
        "--display-error-number",
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
    comptime cuda_dir: []const u8,
    comptime kernel_path: [:0]const u8,
) void {
    const name = std.fs.path.basename(kernel_path);
    const dummy_zig_kernel = b.addObject(name, kernel_path);
    dummy_zig_kernel.setTarget(.{ .cpu_arch = .nvptx64, .os_tag = .cuda });
    const kernel_ptx_path = std.mem.joinZ(
        b.allocator,
        "",
        &[_][]const u8{ b.exe_dir, "/", dummy_zig_kernel.out_filename, ".ptx" },
    ) catch unreachable;

    if (needRebuild(kernel_path, kernel_ptx_path)) {
        // Actually we need to use Stage2 here, so don't use the dummy obj,
        // and manually run this command.
        const emit_bin = std.mem.join(b.allocator, "=", &[_][]const u8{ "-femit-bin", kernel_ptx_path[0 .. kernel_ptx_path.len - 4] }) catch unreachable;
        const zig_kernel = b.addSystemCommand(&[_][]const u8{
            ZIG_STAGE2,    "build-obj",    kernel_path,
            "-target",     "nvptx64-cuda", "-OReleaseSafe",
            "-Dcpu=sm_30", emit_bin,
        });
        // "--verbose-llvm-ir",
        const validate_ptx = b.addSystemCommand(
            &[_][]const u8{ cuda_dir ++ "/bin/ptxas", kernel_ptx_path },
        );
        validate_ptx.step.dependOn(enableAddrspace(b, &zig_kernel.step, kernel_path));
        exe.step.dependOn(&validate_ptx.step);
    } else {
        std.log.warn("Kernel up-to-date {s}", .{kernel_ptx_path});
    }

    addCudazDeps(b, exe, cuda_dir, kernel_path, kernel_ptx_path);
}

pub fn addCudazDeps(
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
    exe.addIncludeDir(SDK_ROOT ++ "src");
    exe.addIncludeDir(kernel_dir);

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
    cudaz_options.addOption(bool, "portable", exe.build_mode != .Debug);

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

fn sdk_root() []const u8 {
    return std.fs.path.dirname(@src().file).?;
}

fn needRebuild(kernel_path: [:0]const u8, kernel_ptx_path: [:0]const u8) bool {
    var ptx_file = std.fs.openFileAbsoluteZ(kernel_ptx_path, .{}) catch return true;
    var ptx_stat = ptx_file.stat() catch return true;
    // detect empty .ptx files
    if (ptx_stat.size < 128) return true;

    var zig_file = (std.fs.cwd().openFileZ(kernel_path, .{}) catch return true);
    var zig_time = (zig_file.stat() catch return true).mtime;
    return zig_time >= ptx_stat.mtime + std.time.ns_per_s * 10;
}

/// Wraps the given step by a small text processing
/// that enables then disables Stage2 only code.
fn enableAddrspace(b: *Builder, step: *std.build.Step, src: [:0]const u8) *std.build.Step {
    _ = b;
    _ = step;
    _ = src;
    const enable_stage2 = b.addSystemCommand(&[_][]const u8{
        "perl", "-ni", "-e", "s:^( *)// (.* // stage2):$1$2:g ; print", src,
    });
    const disable_stage1 = b.addSystemCommand(&[_][]const u8{
        "perl", "-ni", "-e", "s:^( *)([^/]+ // stage1):$1// $2:g ; print", src,
    });

    disable_stage1.step.dependOn(&enable_stage2.step);
    step.dependOn(&disable_stage1.step);
    // return step;
    const enable_stage1 = b.addSystemCommand(&[_][]const u8{
        "perl", "-ni", "-e", "s:^(\\s*)// (.* // stage1):$1$2:g ; print", src,
    });
    const disable_stage2 = b.addSystemCommand(&[_][]const u8{
        "perl", "-ni", "-e", "s:^(\\s*)([^/]* // stage2):$1// $2:g ; print", src,
    });

    enable_stage1.step.dependOn(step);
    disable_stage2.step.dependOn(&enable_stage1.step);
    return &disable_stage2.step;
}
