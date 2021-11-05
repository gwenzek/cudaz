const std = @import("std");
const meta = std.meta;
const testing = std.testing;
const TypeInfo = std.builtin.TypeInfo;

pub const cudaz_options = @import("cudaz_options");

pub const cu = @cImport({
    @cInclude("cuda.h");
    @cInclude("cuda_globals.h");
    if (cudaz_options.cuda_kernel) {
        @cInclude(cudaz_options.kernel_name);
    }
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

    pub fn dim3(self: *const Dim3) cu.dim3 {
        return cu.dim3{ .x = self.x, .y = self.y, .z = self.z };
    }
};

/// Represents how kernel are execut
pub const Grid = struct {
    blocks: Dim3 = .{},
    threads: Dim3 = .{},

    pub fn init1D(len: usize, threads: usize) Grid {
        return init3D(len, 1, 1, threads, 1, 1);
    }

    pub fn init2D(cols: usize, rows: usize, threads_x: usize, threads_y: usize) Grid {
        return init3D(cols, rows, 1, threads_x, threads_y, 1);
    }

    pub fn init3D(
        cols: usize,
        rows: usize,
        depth: usize,
        threads_x: usize,
        threads_y: usize,
        threads_z: usize,
    ) Grid {
        var t_x = if (threads_x == 0) cols else threads_x;
        var t_y = if (threads_y == 0) rows else threads_y;
        var t_z = if (threads_z == 0) depth else threads_z;
        return Grid{
            .blocks = Dim3.init(
                std.math.divCeil(usize, cols, t_x) catch unreachable,
                std.math.divCeil(usize, rows, t_y) catch unreachable,
                std.math.divCeil(usize, depth, t_z) catch unreachable,
            ),
            .threads = Dim3.init(t_x, t_y, t_z),
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
    UnexpectedByCudaz,
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
        else => error.UnexpectedByCudaz,
    };
    var err_message: [*c]const u8 = undefined;
    const error_string_res = cu.cuGetErrorString(result, &err_message);
    if (error_string_res == cu.CUDA_SUCCESS) {
        std.log.err("Cuda error {d}: {s}", .{ err, err_message });
    } else {
        std.log.err("Cuda error {d} (no error string)", .{err});
    }
    if (err == error.UnexpectedByCudaz) {
        std.log.err("Unknown cuda error {d}. Please open a bug against Cudaz.", .{result});
    }
    return err;
}

pub const Stream = struct {
    device: u8,
    _stream: *cu.CUstream_st,

    pub fn init(device: u8) CudaError!Stream {
        _ = try getCtx(device);
        var stream: cu.CUstream = undefined;
        try check(cu.cuStreamCreate(&stream, cu.CU_STREAM_DEFAULT));
        return Stream{ .device = device, ._stream = stream.? };
    }

    pub fn deinit(self: *Stream) void {
        // Don't handle CUDA errors here
        _ = cu.cuStreamDestroy(self._stream);
        self._stream = undefined;
    }

    // TODO: add a typesafe launch, so that we can remove launch from the Function itself
    pub inline fn launch(self: *const Stream, f: cu.CUfunction, grid: Grid, args: anytype) !void {
        try self.launchWithSharedMem(f, grid, 0, args);
    }

    pub fn launchWithSharedMem(self: *const Stream, f: cu.CUfunction, grid: Grid, shared_mem: usize, args: anytype) !void {
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
            self._stream,
            @ptrCast([*c]?*c_void, &args_ptrs),
            null,
        );
        try check(res);
        try self.synchronize();
    }

    pub fn synchronize(self: *const Stream) !void {
        // TODO: add a debug_sync to catch error in debug code
        try check(cu.cuStreamSynchronize(self._stream));
    }

    pub fn format(
        self: *const Stream,
        comptime fmt: []const u8,
        options: std.fmt.FormatOptions,
        writer: anytype,
    ) !void {
        _ = fmt;
        _ = options;
        try std.fmt.format(writer, "CuStream(device={}, stream={*})", .{ self.device, self._stream });
    }
};

// TODO: return a device pointer
pub fn alloc(comptime DestType: type, size: usize) ![]DestType {
    var int_ptr: cu.CUdeviceptr = undefined;
    try check(cu.cuMemAlloc(&int_ptr, size * @sizeOf(DestType)));
    var ptr = @intToPtr([*]DestType, int_ptr);
    return ptr[0..size];
}

// TODO:
pub fn free(device_ptr: anytype) void {
    var raw_ptr: *c_void = if (meta.trait.isSlice(@TypeOf(device_ptr)))
        @ptrCast(*c_void, device_ptr.ptr)
    else
        @ptrCast(*c_void, device_ptr);
    _ = cu.cuMemFree(@ptrToInt(raw_ptr));
}

pub fn memset(comptime DestType: type, slice: []DestType, value: DestType) !void {
    var d_ptr = @ptrToInt(slice.ptr);
    var n = slice.len;
    var memset_res = switch (@sizeOf(DestType)) {
        1 => cu.cuMemsetD8(d_ptr, @bitCast(u8, value), n),
        2 => cu.cuMemsetD16(d_ptr, @bitCast(u16, value), n),
        4 => cu.cuMemsetD32(d_ptr, @bitCast(u32, value), n),
        else => @compileError("memset doesn't support type: " ++ @typeName(DestType)),
    };
    try check(memset_res);
}

pub fn memsetD8(comptime DestType: type, slice: []DestType, value: u8) !void {
    var d_ptr = @ptrToInt(slice.ptr);
    var n = slice.len * @sizeOf(DestType);
    try check(cu.cuMemsetD8(d_ptr, value, n));
}

pub fn allocAndCopy(comptime DestType: type, h_source: []const DestType) ![]DestType {
    var ptr = try alloc(DestType, h_source.len);
    try memcpyHtoD(DestType, ptr, h_source);
    return ptr;
}

pub fn allocAndCopyResult(comptime DestType: type, allocator: *std.mem.Allocator, d_source: []const DestType) ![]DestType {
    var h_tgt = try allocator.alloc(DestType, d_source.len);
    try memcpyDtoH(DestType, h_tgt, d_source);
    return h_tgt;
}

pub fn readResult(comptime DestType: type, d_source: []const DestType) !DestType {
    var h_res: [1]DestType = undefined;
    try memcpyDtoH(DestType, &h_res, d_source);
    return h_res[0];
}

pub fn memcpyHtoD(comptime DestType: type, d_target: []DestType, h_source: []const DestType) !void {
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
pub fn memcpyDtoH(comptime DestType: type, h_target: []DestType, d_source: []const DestType) !void {
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

/// Time gpu event.
/// `deinit` is called when `elapsed` is called.
/// Note: we don't check errors, you'll receive Nan if any error happens.
/// start and stop are asynchronous, only elapsed is blocking and will wait
/// for the underlying operations to be over.
pub const GpuTimer = struct {
    _start: cu.CUevent,
    _stop: cu.CUevent,
    // Here we take a pointer to the Zig struct.
    // This way we can detect if we try to use a timer on a deleted stream
    stream: *const Stream,
    _elapsed: f32 = std.math.nan_f32,

    pub fn init(stream: *const Stream) GpuTimer {
        // The cuEvent are implicitly reffering to the current context.
        // We don't know if the current context is the same than the stream context.
        // Typically I'm not sure what happens with 2 streams on 2 gpus.
        // We might need to restore the stream context before creating the events.
        var timer = GpuTimer{ ._start = undefined, ._stop = undefined, .stream = stream };
        _ = cu.cuEventCreate(&timer._start, 0);
        _ = cu.cuEventCreate(&timer._stop, 0);
        return timer;
    }

    pub fn deinit(self: *GpuTimer) void {
        // Double deinit is allowed
        if (self._stop == null) return;
        _ = cu.cuEventDestroy(self._start);
        self._start = null;
        _ = cu.cuEventDestroy(self._stop);
        self._stop = null;
    }

    pub fn start(self: *GpuTimer) void {
        _ = cu.cuEventRecord(self._start, self.stream._stream);
    }

    pub fn stop(self: *GpuTimer) void {
        _ = cu.cuEventRecord(self._stop, self.stream._stream);
    }

    /// Return the elapsed time in milliseconds.
    /// Resolution is around 0.5 microseconds.
    pub fn elapsed(self: *GpuTimer) f32 {
        if (!std.math.isNan(self._elapsed)) return self._elapsed;
        var _elapsed = std.math.nan_f32;
        _ = cu.cuEventSynchronize(self._stop);
        _ = cu.cuEventElapsedTime(&_elapsed, self._start, self._stop);
        self.deinit();
        self._elapsed = _elapsed;
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

// TODO: who is responsible for destroying the context ?
// Given that we already assume one program == one module,
// we can also assume one program == one context per GPU
var _ctx = [1]cu.CUcontext{null} ** 8;
fn getCtx(device: u8) !cu.CUcontext {
    if (_ctx[device]) |ctx| {
        return ctx;
    }
    try check(cu.cuInit(0));
    var cu_dev: cu.CUdevice = undefined;
    try check(cu.cuDeviceGet(&cu_dev, device));
    try check(cu.cuCtxCreate(&_ctx[device], 0, cu_dev));
    return _ctx[device];
}

var _default_module: cu.CUmodule = null;

fn defaultModule() cu.CUmodule {
    if (_default_module != null) return _default_module;
    const file = cudaz_options.kernel_ptx_path;

    check(cu.cuModuleLoad(&_default_module, file)) catch |err| {
        std.log.err("Couldn't load {s}: {}", .{ file, err });
        std.debug.panic("Couldn't load default ptx", .{});
    };
    if (_default_module == null) {
        std.debug.panic("Couldn't load default ptx", .{});
    }
    return _default_module;
}

/// Create a function with the correct signature for a cuda Kernel.
/// The kernel must come from the default .cu file
pub inline fn Function(comptime name: [:0]const u8) type {
    return FnStruct(name, @field(cu, name));
}

pub fn FnStruct(comptime name: [:0]const u8, comptime func: anytype) type {
    return struct {
        const Self = @This();
        const CpuFn = func;
        const Args = meta.ArgsTuple(@TypeOf(Self.CpuFn));

        f: cu.CUfunction,

        pub fn init() !Self {
            var f: cu.CUfunction = undefined;
            try check(cu.cuModuleGetFunction(&f, defaultModule(), name));
            var res = Self{ .f = f };
            std.log.info("Loaded function {}", .{res});
            return res;
        }

        // TODO: deinit -> CUDestroy

        pub fn launch(self: *const Self, stream: *const Stream, grid: Grid, args: Args) !void {
            try stream.launch(self.f, grid, args);
        }

        pub fn launchWithSharedMem(self: *const Self, stream: *const Stream, grid: Grid, shared_mem: usize, args: Args) !void {
            // TODO: this seems error prone, could we make the type of the shared buffer
            // part of the function signature ?
            try stream.launchWithSharedMem(self.f, grid, shared_mem, args);
        }

        pub fn debugCpuCall(grid: Grid, point: Grid, args: Args) void {
            cu.blockDim = grid.blocks.dim3();
            cu.threadDim = grid.threads.dim3();
            cu.blockIdx = point.blocks.dim3();
            cu.threadIdx = point.threads.dim3();
            _ = @call(.{}, CpuFn, args);
        }

        pub fn format(
            self: *const Self,
            comptime fmt: []const u8,
            options: std.fmt.FormatOptions,
            writer: anytype,
        ) !void {
            _ = self;
            _ = fmt;
            _ = options;
            try std.fmt.format(writer, "{s}(", .{name});
            inline for (@typeInfo(Args).Struct.fields) |arg| {
                const ArgT = arg.field_type;
                try std.fmt.format(writer, "{}, ", .{ArgT});
            }
            try std.fmt.format(writer, ")", .{});
        }
    };
}

test "can read function signature from .cu files" {
    std.log.warn("My kernel: {s}", .{@TypeOf(cu.rgba_to_greyscale)});
}

test "rgba_to_greyscale" {
    var stream = try Stream.init(0);
    defer stream.deinit();
    std.log.warn("cuda: {}", .{stream});
    const rgba_to_greyscale = try Function("rgba_to_greyscale").init();
    const numRows: u32 = 10;
    const numCols: u32 = 20;
    const d_rgbaImage = try alloc([4]u8, numRows * numCols);
    // try memset([4]u8, d_rgbaImage, [4]u8{ 0xaa, 0, 0, 255 });
    const d_greyImage = try alloc(u8, numRows * numCols);
    try memset(u8, d_greyImage, 0);

    try stream.launch(
        rgba_to_greyscale.f,
        .{ .blocks = Dim3.init(numRows, numCols, 1) },
        .{ d_rgbaImage, d_greyImage, numRows, numCols },
    );
    try stream.synchronize();
}

test "safe kernel" {
    const rgba_to_greyscale = try Function("rgba_to_greyscale").init();
    var stream = try Stream.init(0);
    defer stream.deinit();
    const numRows: u32 = 10;
    const numCols: u32 = 20;
    var d_rgbaImage = try alloc(cu.uchar3, numRows * numCols);
    // memset(cu.uchar3, d_rgbaImage, 0xaa);
    const d_greyImage = try alloc(u8, numRows * numCols);
    try memset(u8, d_greyImage, 0);
    try stream.synchronize();
    std.log.warn("stream: {}, fn: {}", .{ stream, rgba_to_greyscale.f });
    try rgba_to_greyscale.launch(
        &stream,
        .{ .blocks = Dim3.init(numCols, numRows, 1) },
        // TODO: we should accept slices
        .{ d_rgbaImage.ptr, d_greyImage.ptr, numRows, numCols },
    );
}

test "cuda alloc" {
    var stream = try Stream.init(0);
    defer stream.deinit();

    const d_greyImage = try alloc(u8, 128);
    try memset(u8, d_greyImage, 0);
    defer free(d_greyImage);
}

test "run the kernel on CPU" {
    // This isn't very ergonomic, but it's possible !
    // Also ironically it can't run in parallel because of the usage of the
    // globals blockIdx and threadIdx.
    // I think it could be useful to detect out of bound errors that Cuda
    // tend to ignore.
    const rgba_to_greyscale = Function("rgba_to_greyscale");
    const rgbImage = [_]cu.uchar3{
        .{ .x = 0x2D, .y = 0x24, .z = 0x1F },
        .{ .x = 0xEB, .y = 0x82, .z = 0x48 },
    };
    var gray = [_]u8{ 0, 0 };
    rgba_to_greyscale.debugCpuCall(
        Grid.init1D(2, 1),
        .{ .blocks = Dim3.init(0, 0, 0), .threads = Dim3.init(0, 0, 0) },
        .{ &rgbImage, &gray, 1, 2 },
    );
    rgba_to_greyscale.debugCpuCall(
        Grid.init1D(2, 1),
        .{ .blocks = Dim3.init(0, 0, 0), .threads = Dim3.init(1, 0, 0) },
        .{ &rgbImage, &gray, 1, 2 },
    );

    try testing.expectEqual([_]u8{ 38, 154 }, gray);
}

test "GpuTimer" {
    const rgba_to_greyscale = try Function("rgba_to_greyscale").init();
    var stream = try Stream.init(0);
    defer stream.deinit();
    const numRows: u32 = 10;
    const numCols: u32 = 20;
    var d_rgbaImage = try alloc(cu.uchar3, numRows * numCols);
    // memset(cu.uchar3, d_rgbaImage, 0xaa);
    const d_greyImage = try alloc(u8, numRows * numCols);
    try memset(u8, d_greyImage, 0);

    std.log.warn("stream: {}, fn: {}", .{ stream, rgba_to_greyscale.f });
    var timer = GpuTimer.init(&stream);
    timer.start();
    try rgba_to_greyscale.launch(
        &stream,
        .{ .blocks = Dim3.init(numCols, numRows, 1) },
        .{ d_rgbaImage.ptr, d_greyImage.ptr, numRows, numCols },
    );
    timer.stop();
    std.log.warn("rgba_to_greyscale took: {}", .{timer.elapsed()});
    try testing.expect(timer.elapsed() > 0);
}

test "we use only one context per GPU" {
    var stream = try Stream.init(0);
    var default_ctx: cu.CUcontext = undefined;
    var stream_ctx: cu.CUcontext = undefined;
    try check(cu.cuCtxGetCurrent(&default_ctx));
    try check(cu.cuStreamGetCtx(stream._stream, &stream_ctx));
}

fn cudaAllocFn(allocator: *std.mem.Allocator, n: usize, ptr_align: u29, len_align: u29, ra: usize) std.mem.Allocator.Error![]u8 {
    _ = allocator;
    _ = ra;
    // TODO implement alignment
    _ = ptr_align;
    _ = len_align;

    return alloc(u8, n) catch |err| switch (err) {
        error.OutOfMemory => error.OutOfMemory,
        else => {
            std.log.err("Cuda error while allocating memory: {}", .{err});
            return error.OutOfMemory;
        },
    };
}

fn cudaResizeFn(allocator: *std.mem.Allocator, buf: []u8, buf_align: u29, new_len: usize, len_align: u29, ra: usize) std.mem.Allocator.Error!usize {
    _ = allocator;
    _ = ra;
    _ = buf_align;
    _ = len_align;
    if (new_len == 0) {
        free(buf);
        return new_len;
    }

    return error.OutOfMemory;
}

pub const cuda_allocator = &cuda_allocator_state;
var cuda_allocator_state = std.mem.Allocator{
    .allocFn = cudaAllocFn,
    .resizeFn = cudaResizeFn,
};

// TODO: fix cuda_allocator
// test "cuda_allocator" {
//     _ = try Stream.init(0);
//     try std.heap.testAllocator(cuda_allocator);
// }
