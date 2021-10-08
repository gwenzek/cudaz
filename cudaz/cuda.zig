const std = @import("std");
const meta = std.meta;
const testing = std.testing;
const TypeInfo = std.builtin.TypeInfo;

pub const cudaz_options = @import("cudaz_options");

pub const cu = @cImport({
    @cInclude("cuda.h");
    @cInclude("cuda_globals.h");
    @cInclude(cudaz_options.kernel_name);
});

pub const Dim3 = struct {
    x: c_uint = 1,
    y: c_uint = 1,
    z: c_uint = 1,

    pub fn init(x: usize, y: usize, z: usize) Dim3 {
        return .{
            .x = @intCast(c_uint, x),
            .y = @intCast(c_uint, y),
            .z = @intCast(c_uint, z),
        };
    }
};

/// Represents how kernel are execut
pub const Grid = struct {
    blocks: Dim3 = .{},
    threads: Dim3 = .{},

    pub fn init1D(len: usize, threads: usize) Grid {
        var t_x = threads;
        if (threads == 0) {
            // This correspond to having one thread per item.
            // This is likely to crash at runtime, unless for very small arrays.
            // Because there is a max number of threads supported by each GPU.
            t_x = len;
        }
        return Grid{
            .blocks = .{ .x = @intCast(c_uint, std.math.divCeil(usize, len, t_x) catch unreachable) },
            .threads = .{ .x = @intCast(c_uint, t_x) },
        };
    }
    pub fn init2D(rows: usize, cols: usize, threads_x: usize, threads_y: usize) Grid {
        var t_x = if (threads_x == 0) rows else threads_x;
        var t_y = if (threads_y == 0) cols else threads_y;
        return Grid{
            .blocks = Dim3.init(
                std.math.divCeil(usize, cols, t_x) catch unreachable,
                std.math.divCeil(usize, rows, t_y) catch unreachable,
                1,
            ),
            .threads = Dim3.init(t_x, t_y, 1),
        };
    }
};

pub const CudaError = error{
    InvalidValue,
    OutOfMemory,
    NotInitialized,
    Deinitialized,
    ProfilerDisabled,
    ProfilerNotInitialized,
    ProfilerAlreadyStarted,
    ProfilerAlreadyStopped,
    StubLibrary,
    NoDevice,
    InvalidDevice,
    DeviceNotLicensed,
    InvalidImage,
    InvalidContext,
    ContextAlreadyCurrent,
    MapFailed,
    UnmapFailed,
    ArrayIsMapped,
    AlreadyMapped,
    NoBinaryForGpu,
    AlreadyAcquired,
    NotMapped,
    NotMappedAsArray,
    NotMappedAsPointer,
    EccUncorrectable,
    UnsupportedLimit,
    ContextAlreadyInUse,
    PeerAccessUnsupported,
    InvalidPtx,
    InvalidGraphicsContext,
    NvlinkUncorrectable,
    JitCompilerNotFound,
    UnsupportedPtxVersion,
    JitCompilationDisabled,
    UnsupportedExecAffinity,
    InvalidSource,
    FileNotFound,
    SharedObjectSymbolNotFound,
    SharedObjectInitFailed,
    OperatingSystem,
    InvalidHandle,
    IllegalState,
    NotFound,
    NotReady,
    IllegalAddress,
    LaunchOutOfResources,
    LaunchTimeout,
    LaunchIncompatibleTexturing,
    PeerAccessAlreadyEnabled,
    PeerAccessNotEnabled,
    PrimaryContextActive,
    ContextIsDestroyed,
    Assert,
    TooManyPeers,
    HostMemoryAlreadyRegistered,
    HostMemoryNotRegistered,
    HardwareStackError,
    IllegalInstruction,
    MisalignedAddress,
    InvalidAddressSpace,
    InvalidPc,
    LaunchFailed,
    CooperativeLaunchTooLarge,
    NotPermitted,
    NotSupported,
    SystemNotReady,
    SystemDriverMismatch,
    CompatNotSupportedOnDevice,
    MpsConnectionFailed,
    MpsRpcFailure,
    MpsServerNotReady,
    MpsMaxClientsReached,
    MpsMaxConnectionsReached,
    StreamCaptureUnsupported,
    StreamCaptureInvalidated,
    StreamCaptureMerge,
    StreamCaptureUnmatched,
    StreamCaptureUnjoined,
    StreamCaptureIsolation,
    StreamCaptureImplicit,
    CapturedEvent,
    StreamCaptureWrongThread,
    Timeout,
    GraphExecUpdateFailure,
    ExternalDevice,
    Unknown,
    UnexpectedByCudaZig,
};

pub fn check(result: cu.CUresult) CudaError!void {
    var err = switch (result) {
        cu.CUDA_SUCCESS => return,
        cu.CUDA_ERROR_INVALID_VALUE => error.InvalidValue,
        cu.CUDA_ERROR_OUT_OF_MEMORY => error.OutOfMemory,
        cu.CUDA_ERROR_NOT_INITIALIZED => error.NotInitialized,
        cu.CUDA_ERROR_DEINITIALIZED => error.Deinitialized,
        cu.CUDA_ERROR_PROFILER_DISABLED => error.ProfilerDisabled,
        cu.CUDA_ERROR_PROFILER_NOT_INITIALIZED => error.ProfilerNotInitialized,
        cu.CUDA_ERROR_PROFILER_ALREADY_STARTED => error.ProfilerAlreadyStarted,
        cu.CUDA_ERROR_PROFILER_ALREADY_STOPPED => error.ProfilerAlreadyStopped,
        cu.CUDA_ERROR_STUB_LIBRARY => error.StubLibrary,
        cu.CUDA_ERROR_NO_DEVICE => error.NoDevice,
        cu.CUDA_ERROR_INVALID_DEVICE => error.InvalidDevice,
        cu.CUDA_ERROR_DEVICE_NOT_LICENSED => error.DeviceNotLicensed,
        cu.CUDA_ERROR_INVALID_IMAGE => error.InvalidImage,
        cu.CUDA_ERROR_INVALID_CONTEXT => error.InvalidContext,
        cu.CUDA_ERROR_CONTEXT_ALREADY_CURRENT => error.ContextAlreadyCurrent,
        cu.CUDA_ERROR_MAP_FAILED => error.MapFailed,
        cu.CUDA_ERROR_UNMAP_FAILED => error.UnmapFailed,
        cu.CUDA_ERROR_ARRAY_IS_MAPPED => error.ArrayIsMapped,
        cu.CUDA_ERROR_ALREADY_MAPPED => error.AlreadyMapped,
        cu.CUDA_ERROR_NO_BINARY_FOR_GPU => error.NoBinaryForGpu,
        cu.CUDA_ERROR_ALREADY_ACQUIRED => error.AlreadyAcquired,
        cu.CUDA_ERROR_NOT_MAPPED => error.NotMapped,
        cu.CUDA_ERROR_NOT_MAPPED_AS_ARRAY => error.NotMappedAsArray,
        cu.CUDA_ERROR_NOT_MAPPED_AS_POINTER => error.NotMappedAsPointer,
        cu.CUDA_ERROR_ECC_UNCORRECTABLE => error.EccUncorrectable,
        cu.CUDA_ERROR_UNSUPPORTED_LIMIT => error.UnsupportedLimit,
        cu.CUDA_ERROR_CONTEXT_ALREADY_IN_USE => error.ContextAlreadyInUse,
        cu.CUDA_ERROR_PEER_ACCESS_UNSUPPORTED => error.PeerAccessUnsupported,
        cu.CUDA_ERROR_INVALID_PTX => error.InvalidPtx,
        cu.CUDA_ERROR_INVALID_GRAPHICS_CONTEXT => error.InvalidGraphicsContext,
        cu.CUDA_ERROR_NVLINK_UNCORRECTABLE => error.NvlinkUncorrectable,
        cu.CUDA_ERROR_JIT_COMPILER_NOT_FOUND => error.JitCompilerNotFound,
        cu.CUDA_ERROR_UNSUPPORTED_PTX_VERSION => error.UnsupportedPtxVersion,
        cu.CUDA_ERROR_JIT_COMPILATION_DISABLED => error.JitCompilationDisabled,
        cu.CUDA_ERROR_UNSUPPORTED_EXEC_AFFINITY => error.UnsupportedExecAffinity,
        cu.CUDA_ERROR_INVALID_SOURCE => error.InvalidSource,
        cu.CUDA_ERROR_FILE_NOT_FOUND => error.FileNotFound,
        cu.CUDA_ERROR_SHARED_OBJECT_SYMBOL_NOT_FOUND => error.SharedObjectSymbolNotFound,
        cu.CUDA_ERROR_SHARED_OBJECT_INIT_FAILED => error.SharedObjectInitFailed,
        cu.CUDA_ERROR_OPERATING_SYSTEM => error.OperatingSystem,
        cu.CUDA_ERROR_INVALID_HANDLE => error.InvalidHandle,
        cu.CUDA_ERROR_ILLEGAL_STATE => error.IllegalState,
        cu.CUDA_ERROR_NOT_FOUND => error.NotFound,
        cu.CUDA_ERROR_NOT_READY => error.NotReady,
        cu.CUDA_ERROR_ILLEGAL_ADDRESS => error.IllegalAddress,
        cu.CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES => error.LaunchOutOfResources,
        cu.CUDA_ERROR_LAUNCH_TIMEOUT => error.LaunchTimeout,
        cu.CUDA_ERROR_LAUNCH_INCOMPATIBLE_TEXTURING => error.LaunchIncompatibleTexturing,
        cu.CUDA_ERROR_PEER_ACCESS_ALREADY_ENABLED => error.PeerAccessAlreadyEnabled,
        cu.CUDA_ERROR_PEER_ACCESS_NOT_ENABLED => error.PeerAccessNotEnabled,
        cu.CUDA_ERROR_PRIMARY_CONTEXT_ACTIVE => error.PrimaryContextActive,
        cu.CUDA_ERROR_CONTEXT_IS_DESTROYED => error.ContextIsDestroyed,
        cu.CUDA_ERROR_ASSERT => error.Assert,
        cu.CUDA_ERROR_TOO_MANY_PEERS => error.TooManyPeers,
        cu.CUDA_ERROR_HOST_MEMORY_ALREADY_REGISTERED => error.HostMemoryAlreadyRegistered,
        cu.CUDA_ERROR_HOST_MEMORY_NOT_REGISTERED => error.HostMemoryNotRegistered,
        cu.CUDA_ERROR_HARDWARE_STACK_ERROR => error.HardwareStackError,
        cu.CUDA_ERROR_ILLEGAL_INSTRUCTION => error.IllegalInstruction,
        cu.CUDA_ERROR_MISALIGNED_ADDRESS => error.MisalignedAddress,
        cu.CUDA_ERROR_INVALID_ADDRESS_SPACE => error.InvalidAddressSpace,
        cu.CUDA_ERROR_INVALID_PC => error.InvalidPc,
        cu.CUDA_ERROR_LAUNCH_FAILED => error.LaunchFailed,
        cu.CUDA_ERROR_COOPERATIVE_LAUNCH_TOO_LARGE => error.CooperativeLaunchTooLarge,
        cu.CUDA_ERROR_NOT_PERMITTED => error.NotPermitted,
        cu.CUDA_ERROR_NOT_SUPPORTED => error.NotSupported,
        cu.CUDA_ERROR_SYSTEM_NOT_READY => error.SystemNotReady,
        cu.CUDA_ERROR_SYSTEM_DRIVER_MISMATCH => error.SystemDriverMismatch,
        cu.CUDA_ERROR_COMPAT_NOT_SUPPORTED_ON_DEVICE => error.CompatNotSupportedOnDevice,
        cu.CUDA_ERROR_MPS_CONNECTION_FAILED => error.MpsConnectionFailed,
        cu.CUDA_ERROR_MPS_RPC_FAILURE => error.MpsRpcFailure,
        cu.CUDA_ERROR_MPS_SERVER_NOT_READY => error.MpsServerNotReady,
        cu.CUDA_ERROR_MPS_MAX_CLIENTS_REACHED => error.MpsMaxClientsReached,
        cu.CUDA_ERROR_MPS_MAX_CONNECTIONS_REACHED => error.MpsMaxConnectionsReached,
        cu.CUDA_ERROR_STREAM_CAPTURE_UNSUPPORTED => error.StreamCaptureUnsupported,
        cu.CUDA_ERROR_STREAM_CAPTURE_INVALIDATED => error.StreamCaptureInvalidated,
        cu.CUDA_ERROR_STREAM_CAPTURE_MERGE => error.StreamCaptureMerge,
        cu.CUDA_ERROR_STREAM_CAPTURE_UNMATCHED => error.StreamCaptureUnmatched,
        cu.CUDA_ERROR_STREAM_CAPTURE_UNJOINED => error.StreamCaptureUnjoined,
        cu.CUDA_ERROR_STREAM_CAPTURE_ISOLATION => error.StreamCaptureIsolation,
        cu.CUDA_ERROR_STREAM_CAPTURE_IMPLICIT => error.StreamCaptureImplicit,
        cu.CUDA_ERROR_CAPTURED_EVENT => error.CapturedEvent,
        cu.CUDA_ERROR_STREAM_CAPTURE_WRONG_THREAD => error.StreamCaptureWrongThread,
        cu.CUDA_ERROR_TIMEOUT => error.Timeout,
        cu.CUDA_ERROR_GRAPH_EXEC_UPDATE_FAILURE => error.GraphExecUpdateFailure,
        cu.CUDA_ERROR_EXTERNAL_DEVICE => error.ExternalDevice,
        cu.CUDA_ERROR_UNKNOWN => error.Unknown,
        else => error.UnexpectedByCudaZig,
        // TODO: take inspiration on zig.std.os on how to handle unexpected errors
        // https://github.com/ziglang/zig/blob/c4f97d336528d5b795c6584053f072cf8e28495e/lib/std/os.zig#L4889
    };
    var err_message: [*c]const u8 = undefined;
    const error_string_res = cu.cuGetErrorString(result, &err_message);
    if (error_string_res == cu.CUDA_SUCCESS) {
        std.log.err("Cuda error {d}: {s}", .{ err, err_message });
    } else {
        std.log.err("Cuda error {d} (no error string)", .{err});
    }
    return err;
}

pub const Cuda = struct {
    arena: std.heap.ArenaAllocator,
    stream: cu.CUstream = undefined,
    device: cu.CUdevice = undefined,
    ctx: cu.CUcontext = undefined,

    pub fn init(device: u8) CudaError!Cuda {
        var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
        var cuda = Cuda{ .arena = arena };
        try check(cu.cuInit(0));
        try check(cu.cuDeviceGet(&cuda.device, device));
        try check(cu.cuCtxCreate(&cuda.ctx, 0, cuda.device));
        try check(cu.cuStreamCreate(&cuda.stream, cu.CU_STREAM_DEFAULT));
        return cuda;
    }

    pub fn deinit(self: *Cuda) void {
        // Don't handle CUDA errors here
        _ = cu.cuStreamDestroy(self.stream);
        _ = cu.cuCtxDestroy(self.ctx);
        self.arena.deinit();
    }

    // TODO: return a device pointer
    pub fn alloc(self: *Cuda, comptime DestType: type, size: usize) ![]DestType {
        _ = self;
        var int_ptr: cu.CUdeviceptr = undefined;
        try check(cu.cuMemAlloc(&int_ptr, size * @sizeOf(DestType)));
        var ptr = @intToPtr([*]DestType, int_ptr);
        return ptr[0..size];
    }

    // TODO:
    pub fn free(self: *Cuda, device_ptr: anytype) void {
        _ = self;
        var raw_ptr: *c_void = if (meta.trait.isSlice(@TypeOf(device_ptr)))
            @ptrCast(*c_void, device_ptr.ptr)
        else
            @ptrCast(*c_void, device_ptr);
        _ = cu.cuMemFree(@ptrToInt(raw_ptr));
    }

    pub fn memset(self: *Cuda, comptime DestType: type, slice: []DestType, value: DestType) !void {
        _ = self;
        var d_ptr = @ptrToInt(slice.ptr);
        var n = slice.len;
        var memset_res = switch (@sizeOf(DestType)) {
            1 => cu.cuMemsetD8(d_ptr, @bitCast(u8, value), n),
            2 => cu.cuMemsetD16(d_ptr, @bitCast(u16, value), n),
            4 => cu.cuMemsetD32(d_ptr, @bitCast(u32, value), n),
            else => @compileError("cuda.memset doesn't support type: " ++ @typeName(DestType)),
        };
        try check(memset_res);
    }

    pub fn memsetD8(self: *Cuda, comptime DestType: type, slice: []DestType, value: u8) !void {
        _ = self;
        var d_ptr = @ptrToInt(slice.ptr);
        var n = slice.len * @sizeOf(DestType);
        try check(cu.cuMemsetD8(d_ptr, value, n));
    }

    pub fn allocAndCopy(self: *Cuda, comptime DestType: type, h_source: []const DestType) ![]DestType {
        var ptr = try self.alloc(DestType, h_source.len);
        try self.memcpyHtoD(DestType, ptr, h_source);
        return ptr;
    }

    pub fn allocAndCopyResult(self: *Cuda, comptime DestType: type, allocator: *std.mem.Allocator, d_source: []const DestType) ![]DestType {
        var h_tgt = try allocator.alloc(DestType, d_source.len);
        try self.memcpyDtoH(DestType, h_tgt, d_source);
        return h_tgt;
    }

    pub fn readResult(self: *Cuda, comptime DestType: type, d_source: []const DestType) !DestType {
        var h_res: [1]DestType = undefined;
        try self.memcpyDtoH(DestType, &h_res, d_source);
        return h_res[0];
    }

    pub fn memcpyHtoD(self: *Cuda, comptime DestType: type, d_target: []DestType, h_source: []const DestType) !void {
        _ = self;
        std.debug.assert(h_source.len == d_target.len);
        check(cu.cuMemcpyHtoD(
            @ptrToInt(d_target.ptr),
            @ptrCast(*const c_void, h_source.ptr),
            h_source.len * @sizeOf(DestType),
        )) catch |err| switch (err) {
            // TODO: leverage adress spaces to make this a comptime check
            error.InvalidValue => std.log.warn("InvalidValue error while memcpyHtoD! Usage is memcpyHtoD(d_tgt, h_src)", .{}),
            else => return err,
        };
    }
    pub fn memcpyDtoH(self: *Cuda, comptime DestType: type, h_target: []DestType, d_source: []const DestType) !void {
        _ = self;
        std.debug.assert(d_source.len == h_target.len);
        check(cu.cuMemcpyDtoH(
            @ptrCast(*c_void, h_target.ptr),
            @ptrToInt(d_source.ptr),
            d_source.len * @sizeOf(DestType),
        )) catch |err| switch (err) {
            // TODO: leverage adress spaces to make this a comptime check
            error.InvalidValue => std.log.warn("InvalidValue error while memcpyDtoH! Usage is memcpyDtoH(h_tgt, d_src).", .{}),
            else => return err,
        };
    }

    pub fn loadFunction(self: *Cuda, file: [*:0]const u8, name: [*:0]const u8) !cu.CUfunction {
        // std.fs.accessAbsoluteZ(file, std.fs.File.OpenFlags{ .read = true }) catch @panic("can't open kernel file: " ++ file);
        var module = self.arena.allocator.create(cu.CUmodule) catch unreachable;
        // TODO: save the module so we can destroy it in deinit
        // TODO: cache module objects (unless Cuda does it for us)
        check(cu.cuModuleLoad(module, file)) catch |err| switch (err) {
            error.FileNotFound => {
                std.log.err("FileNotFound: {s}", .{file});
                return err;
            },
            else => return err,
        };
        var function: cu.CUfunction = undefined;
        try check(cu.cuModuleGetFunction(&function, module.*, name));
        std.log.info("Loaded function {s}:{s} ({})", .{ file, name, function });
        return function;
    }

    pub fn launch(self: *Cuda, f: cu.CUfunction, grid: Grid, args: anytype) !void {
        try self.launchWithSharedMem(f, grid, 0, args);
    }

    pub fn launchWithSharedMem(self: *Cuda, f: cu.CUfunction, grid: Grid, shared_mem: usize, args: anytype) !void {
        // Create an array of pointers pointing to the given args.
        const fields: []const TypeInfo.StructField = meta.fields(@TypeOf(args));
        var args_ptrs: [fields.len:0]usize = undefined;
        inline for (fields) |field, i| {
            args_ptrs[i] = @ptrToInt(&@field(args, field.name));
        }
        const res = cu.cuLaunchKernel(
            f,
            grid.blocks.x,
            grid.blocks.y,
            grid.blocks.z,
            grid.threads.x,
            grid.threads.y,
            grid.threads.z,
            @intCast(c_uint, shared_mem),
            self.stream,
            @ptrCast([*c]?*c_void, &args_ptrs),
            null,
        );
        try check(res);
    }

    pub fn synchronize(self: *Cuda) !void {
        // TODO: add a debug_sync to catch error in debug code
        try check(cu.cuStreamSynchronize(self.stream));
    }

    pub fn format(
        self: *const Cuda,
        comptime fmt: []const u8,
        options: std.fmt.FormatOptions,
        writer: anytype,
    ) !void {
        _ = fmt;
        _ = options;
        try std.fmt.format(writer, "Cuda(device={}, stream={*})", .{ self.device, self.stream });
    }
};

pub const GpuTimer = struct {
    _start: cu.CUevent,
    _stop: cu.CUevent,
    stream: cu.CUstream,

    pub fn init(cuda: *Cuda) GpuTimer {
        var timer = GpuTimer{ ._start = undefined, ._stop = undefined, .stream = cuda.stream };
        _ = cu.cuEventCreate(&timer._start, 0);
        _ = cu.cuEventCreate(&timer._stop, 0);
        return timer;
    }

    pub fn deinit(self: *GpuTimer) void {
        _ = cu.cuEventDestroy(self._start);
        _ = cu.cuEventDestroy(self._stop);
    }

    pub fn start(self: *GpuTimer) void {
        check(cu.cuEventRecord(self._start, self.stream)) catch unreachable;
    }

    pub fn stop(self: *GpuTimer) void {
        check(cu.cuEventRecord(self._stop, self.stream)) catch unreachable;
    }

    pub fn elapsed(self: *GpuTimer) f32 {
        var _elapsed: f32 = undefined;
        _ = cu.cuEventSynchronize(self._stop);
        _ = cu.cuEventElapsedTime(&_elapsed, self._start, self._stop);
        return _elapsed;
    }
};

pub fn main() anyerror!void {
    var general_purpose_allocator = std.heap.GeneralPurposeAllocator(.{}){};
    const gpa = &general_purpose_allocator.allocator;
    const args = try std.process.argsAlloc(gpa);
    defer std.process.argsFree(gpa, args);

    std.log.info("All your codebase are belong to us.", .{});

    std.log.info("cuda: {}", .{cu.cuInit});
    std.log.info("cuInit: {}", .{cu.cuInit(0)});
}

test "cuda version" {
    std.log.warn("Cuda version: {d}", .{cu.CUDA_VERSION});
    try testing.expect(cu.CUDA_VERSION > 11000);
    try testing.expectEqual(cu.cuInit(0), cu.CUDA_SUCCESS);
}

test "cuda init" {
    var cuda = try Cuda.init(0);
    defer cuda.deinit();
}

test "HW1" {
    var cuda = try Cuda.init(0);
    defer cuda.deinit();
    std.log.warn("cuda: {}", .{cuda});
    const rgba_to_greyscale = try cuda.loadFunction(cudaz_options.kernel_ptx_path, "rgba_to_greyscale");
    const numRows: u32 = 10;
    const numCols: u32 = 20;
    const d_rgbaImage = try cuda.alloc([4]u8, numRows * numCols);
    // try cuda.memset([4]u8, d_rgbaImage, [4]u8{ 0xaa, 0, 0, 255 });
    const d_greyImage = try cuda.alloc(u8, numRows * numCols);
    try cuda.memset(u8, d_greyImage, 0);

    // copy input array to the GPU
    // checkCudaErrors(cudaMemcpy(*d_rgbaImage, *inputImage, sizeof(uchar3) * numPixels, cudaMemcpyHostToDevice));

    try cuda.launch(
        rgba_to_greyscale,
        .{ .blocks = Dim3.init(numRows, numCols, 1) },
        .{ d_rgbaImage, d_greyImage, numRows, numCols },
    );
}

/// Create a function with the correct signature for a cuda Kernel.
/// The kernel must come from the default .cu file
pub fn Function(comptime name: [:0]const u8) type {
    return struct {
        const Self = @This();
        const Args = meta.ArgsTuple(@TypeOf(@field(cu, name)));

        f: cu.CUfunction,
        cuda: *Cuda,

        pub fn init(cuda: *Cuda) !Self {
            var k = Self{ .f = undefined, .cuda = cuda };
            k.f = try cuda.loadFunction(cudaz_options.kernel_ptx_path, name);
            return k;
        }

        // TODO: deinit -> CUDestroy

        pub fn launch(self: *const Self, grid: Grid, args: Args) !void {
            try self.cuda.launch(self.f, grid, args);
        }

        pub fn launchWithSharedMem(self: *const Self, grid: Grid, shared_mem: usize, args: Args) !void {
            // TODO: this seems error prone, could we make the type of the shared buffer
            // part of the function signature ?
            try self.cuda.launchWithSharedMem(self.f, grid, shared_mem, args);
        }
    };
}

test "kernel.cu" {
    std.log.warn("My kernel: {s}", .{@TypeOf(cu.rgba_to_greyscale)});
}

test "safe kernel" {
    var cuda = try Cuda.init(0);
    defer cuda.deinit();
    std.log.warn("cuda: {}", .{cuda});
    _ = try cuda.loadFunction(cudaz_options.kernel_ptx_path, "rgba_to_greyscale");
    const numRows: u32 = 10;
    const numCols: u32 = 20;
    var d_rgbaImage = try cuda.alloc(cu.uchar3, numRows * numCols);
    // cuda.memset(cu.uchar3, d_rgbaImage, 0xaa);
    const d_greyImage = try cuda.alloc(u8, numRows * numCols);
    try cuda.memset(u8, d_greyImage, 0);

    const rgba_to_greyscale_safe = try Function("rgba_to_greyscale").init(&cuda);
    std.log.warn("kernel args: {s}", .{@TypeOf(rgba_to_greyscale_safe).Args});
    try rgba_to_greyscale_safe.launch(
        .{ .blocks = Dim3.init(numCols, numRows, 1) },
        // TODO: we should accept slices
        .{ d_rgbaImage.ptr, d_greyImage.ptr, numRows, numCols },
    );
}

test "cuda alloc" {
    var cuda = try Cuda.init(0);
    defer cuda.deinit();

    const d_greyImage = try cuda.alloc(u8, 128);
    try cuda.memset(u8, d_greyImage, 0);
    defer cuda.free(d_greyImage);
}
