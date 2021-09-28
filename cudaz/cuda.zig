const std = @import("std");
const meta = std.meta;
const testing = std.testing;
const TypeInfo = std.builtin.TypeInfo;

pub const cu = @cImport({
    @cInclude("cuda.h");
    // @cInclude("cuda_runtime.h");
    @cInclude("cuda_globals.h");
    @cInclude("kernel.cu");
});

pub const Dim3 = struct {
    x: c_uint = 1,
    y: c_uint = 1,
    z: c_uint = 1,
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
    const z = CudaError;
    return switch (result) {
        .CUDA_SUCCESS => .{},
        .CUDA_ERROR_INVALID_VALUE => error.InvalidValue,
        .CUDA_ERROR_OUT_OF_MEMORY => error.OutOfMemory,
        .CUDA_ERROR_NOT_INITIALIZED => error.NotInitialized,
        .CUDA_ERROR_DEINITIALIZED => error.Deinitialized,
        .CUDA_ERROR_PROFILER_DISABLED => error.ProfilerDisabled,
        .CUDA_ERROR_PROFILER_NOT_INITIALIZED => error.ProfilerNotInitialized,
        .CUDA_ERROR_PROFILER_ALREADY_STARTED => error.ProfilerAlreadyStarted,
        .CUDA_ERROR_PROFILER_ALREADY_STOPPED => error.ProfilerAlreadyStopped,
        .CUDA_ERROR_STUB_LIBRARY => error.StubLibrary,
        .CUDA_ERROR_NO_DEVICE => error.NoDevice,
        .CUDA_ERROR_INVALID_DEVICE => error.InvalidDevice,
        .CUDA_ERROR_DEVICE_NOT_LICENSED => error.DeviceNotLicensed,
        .CUDA_ERROR_INVALID_IMAGE => error.InvalidImage,
        .CUDA_ERROR_INVALID_CONTEXT => error.InvalidContext,
        .CUDA_ERROR_CONTEXT_ALREADY_CURRENT => error.ContextAlreadyCurrent,
        .CUDA_ERROR_MAP_FAILED => error.MapFailed,
        .CUDA_ERROR_UNMAP_FAILED => error.UnmapFailed,
        .CUDA_ERROR_ARRAY_IS_MAPPED => error.ArrayIsMapped,
        .CUDA_ERROR_ALREADY_MAPPED => error.AlreadyMapped,
        .CUDA_ERROR_NO_BINARY_FOR_GPU => error.NoBinaryForGpu,
        .CUDA_ERROR_ALREADY_ACQUIRED => error.AlreadyAcquired,
        .CUDA_ERROR_NOT_MAPPED => error.NotMapped,
        .CUDA_ERROR_NOT_MAPPED_AS_ARRAY => error.NotMappedAsArray,
        .CUDA_ERROR_NOT_MAPPED_AS_POINTER => error.NotMappedAsPointer,
        .CUDA_ERROR_ECC_UNCORRECTABLE => error.EccUncorrectable,
        .CUDA_ERROR_UNSUPPORTED_LIMIT => error.UnsupportedLimit,
        .CUDA_ERROR_CONTEXT_ALREADY_IN_USE => error.ContextAlreadyInUse,
        .CUDA_ERROR_PEER_ACCESS_UNSUPPORTED => error.PeerAccessUnsupported,
        .CUDA_ERROR_INVALID_PTX => error.InvalidPtx,
        .CUDA_ERROR_INVALID_GRAPHICS_CONTEXT => error.InvalidGraphicsContext,
        .CUDA_ERROR_NVLINK_UNCORRECTABLE => error.NvlinkUncorrectable,
        .CUDA_ERROR_JIT_COMPILER_NOT_FOUND => error.JitCompilerNotFound,
        .CUDA_ERROR_UNSUPPORTED_PTX_VERSION => error.UnsupportedPtxVersion,
        .CUDA_ERROR_JIT_COMPILATION_DISABLED => error.JitCompilationDisabled,
        .CUDA_ERROR_UNSUPPORTED_EXEC_AFFINITY => error.UnsupportedExecAffinity,
        .CUDA_ERROR_INVALID_SOURCE => error.InvalidSource,
        .CUDA_ERROR_FILE_NOT_FOUND => error.FileNotFound,
        .CUDA_ERROR_SHARED_OBJECT_SYMBOL_NOT_FOUND => error.SharedObjectSymbolNotFound,
        .CUDA_ERROR_SHARED_OBJECT_INIT_FAILED => error.SharedObjectInitFailed,
        .CUDA_ERROR_OPERATING_SYSTEM => error.OperatingSystem,
        .CUDA_ERROR_INVALID_HANDLE => error.InvalidHandle,
        .CUDA_ERROR_ILLEGAL_STATE => error.IllegalState,
        .CUDA_ERROR_NOT_FOUND => error.NotFound,
        .CUDA_ERROR_NOT_READY => error.NotReady,
        .CUDA_ERROR_ILLEGAL_ADDRESS => error.IllegalAddress,
        .CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES => error.LaunchOutOfResources,
        .CUDA_ERROR_LAUNCH_TIMEOUT => error.LaunchTimeout,
        .CUDA_ERROR_LAUNCH_INCOMPATIBLE_TEXTURING => error.LaunchIncompatibleTexturing,
        .CUDA_ERROR_PEER_ACCESS_ALREADY_ENABLED => error.PeerAccessAlreadyEnabled,
        .CUDA_ERROR_PEER_ACCESS_NOT_ENABLED => error.PeerAccessNotEnabled,
        .CUDA_ERROR_PRIMARY_CONTEXT_ACTIVE => error.PrimaryContextActive,
        .CUDA_ERROR_CONTEXT_IS_DESTROYED => error.ContextIsDestroyed,
        .CUDA_ERROR_ASSERT => error.Assert,
        .CUDA_ERROR_TOO_MANY_PEERS => error.TooManyPeers,
        .CUDA_ERROR_HOST_MEMORY_ALREADY_REGISTERED => error.HostMemoryAlreadyRegistered,
        .CUDA_ERROR_HOST_MEMORY_NOT_REGISTERED => error.HostMemoryNotRegistered,
        .CUDA_ERROR_HARDWARE_STACK_ERROR => error.HardwareStackError,
        .CUDA_ERROR_ILLEGAL_INSTRUCTION => error.IllegalInstruction,
        .CUDA_ERROR_MISALIGNED_ADDRESS => error.MisalignedAddress,
        .CUDA_ERROR_INVALID_ADDRESS_SPACE => error.InvalidAddressSpace,
        .CUDA_ERROR_INVALID_PC => error.InvalidPc,
        .CUDA_ERROR_LAUNCH_FAILED => error.LaunchFailed,
        .CUDA_ERROR_COOPERATIVE_LAUNCH_TOO_LARGE => error.CooperativeLaunchTooLarge,
        .CUDA_ERROR_NOT_PERMITTED => error.NotPermitted,
        .CUDA_ERROR_NOT_SUPPORTED => error.NotSupported,
        .CUDA_ERROR_SYSTEM_NOT_READY => error.SystemNotReady,
        .CUDA_ERROR_SYSTEM_DRIVER_MISMATCH => error.SystemDriverMismatch,
        .CUDA_ERROR_COMPAT_NOT_SUPPORTED_ON_DEVICE => error.CompatNotSupportedOnDevice,
        .CUDA_ERROR_MPS_CONNECTION_FAILED => error.MpsConnectionFailed,
        .CUDA_ERROR_MPS_RPC_FAILURE => error.MpsRpcFailure,
        .CUDA_ERROR_MPS_SERVER_NOT_READY => error.MpsServerNotReady,
        .CUDA_ERROR_MPS_MAX_CLIENTS_REACHED => error.MpsMaxClientsReached,
        .CUDA_ERROR_MPS_MAX_CONNECTIONS_REACHED => error.MpsMaxConnectionsReached,
        .CUDA_ERROR_STREAM_CAPTURE_UNSUPPORTED => error.StreamCaptureUnsupported,
        .CUDA_ERROR_STREAM_CAPTURE_INVALIDATED => error.StreamCaptureInvalidated,
        .CUDA_ERROR_STREAM_CAPTURE_MERGE => error.StreamCaptureMerge,
        .CUDA_ERROR_STREAM_CAPTURE_UNMATCHED => error.StreamCaptureUnmatched,
        .CUDA_ERROR_STREAM_CAPTURE_UNJOINED => error.StreamCaptureUnjoined,
        .CUDA_ERROR_STREAM_CAPTURE_ISOLATION => error.StreamCaptureIsolation,
        .CUDA_ERROR_STREAM_CAPTURE_IMPLICIT => error.StreamCaptureImplicit,
        .CUDA_ERROR_CAPTURED_EVENT => error.CapturedEvent,
        .CUDA_ERROR_STREAM_CAPTURE_WRONG_THREAD => error.StreamCaptureWrongThread,
        .CUDA_ERROR_TIMEOUT => error.Timeout,
        .CUDA_ERROR_GRAPH_EXEC_UPDATE_FAILURE => error.GraphExecUpdateFailure,
        .CUDA_ERROR_EXTERNAL_DEVICE => error.ExternalDevice,
        .CUDA_ERROR_UNKNOWN => error.Unknown,
        else => error.UnexpectedByCudaZig,
        // TODO: take inspiration on zig.std.os on how to handle unexpected errors
        // https://github.com/ziglang/zig/blob/c4f97d336528d5b795c6584053f072cf8e28495e/lib/std/os.zig#L4889
    };
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
        try check(cu.cuStreamCreate(
            &cuda.stream,
            @enumToInt(cu.CUstream_flags_enum.CU_STREAM_DEFAULT),
        ));
        return cuda;
    }

    pub fn deinit(self: *Cuda) void {
        // Don't handle CUDA errors here
        _ = cu.cuStreamDestroy(self.stream);
        _ = cu.cuCtxDestroy(self.ctx);
        self.arena.deinit();
    }

    // TODO: return a device pointer
    pub fn alloc(self: *Cuda, comptime DestType: type, size: usize) []DestType {
        var int_ptr: cu.CUdeviceptr = undefined;
        _ = cu.cuMemAlloc(&int_ptr, size * @sizeOf(DestType));
        var ptr = @intToPtr([*]DestType, int_ptr);
        return ptr[0..size];
    }

    pub fn free(self: *Cuda, device_ptr: anytype) void {
        var raw_ptr: *c_void = if (meta.trait.isSlice(@TypeOf(device_ptr)))
            @ptrCast(*c_void, device_ptr.ptr)
        else
            @ptrCast(*c_void, device_ptr);
        _ = cu.cuMemFree(@ptrToInt(raw_ptr));
    }

    pub fn memset(self: *Cuda, comptime DestType: type, slice: []const DestType, value: u8) void {
        _ = cu.cuMemsetD8(@ptrToInt(slice.ptr), value, slice.len * @sizeOf(DestType));
    }

    pub fn memcpyHtoD(self: *Cuda, comptime DestType: type, target: []DestType, slice: []const DestType) void {
        std.debug.assert(slice.len == target.len);
        _ = cu.cuMemcpyHtoD(@ptrToInt(target.ptr), @ptrToInt(slice.ptr), slice.len * @sizeOf(DestType));
    }
    pub fn memcpyDtoH(self: *Cuda, comptime DestType: type, target: []DestType, slice: []DestType) void {
        _ = cu.cuMemcpyDtoH(
            @ptrCast(*c_void, target.ptr),
            @ptrToInt(slice.ptr),
            slice.len * @sizeOf(DestType),
        );
    }

    pub fn kernel(self: *Cuda, file: [*:0]const u8, name: [*:0]const u8) !cu.CUfunction {
        // std.fs.accessAbsoluteZ(file, std.fs.File.OpenFlags{ .read = true }) catch @panic("can't open kernel file: " ++ file);
        var module = self.arena.allocator.create(cu.CUmodule) catch unreachable;

        try check(cu.cuModuleLoad(module, file));
        std.log.warn("module {s}: {s}", .{ file, module });

        var function: cu.CUfunction = undefined;
        try check(cu.cuModuleGetFunction(&function, module.*, name));
        std.log.warn("function {s}.{s}: {}", .{ file, name, function });
        return function;
    }

    pub fn launch(self: *Cuda, f: cu.CUfunction, gridDim: Dim3, blockDim: Dim3, args: anytype) !void {
        // Create an array of pointers pointing to the given args.
        const fields: []const TypeInfo.StructField = meta.fields(@TypeOf(args));
        var args_ptrs: [fields.len:0]usize = undefined;
        inline for (fields) |field, i| {
            args_ptrs[i] = @ptrToInt(&@field(args, field.name));
        }
        const res = cu.cuLaunchKernel(f, gridDim.x, gridDim.y, gridDim.z, blockDim.x, blockDim.y, blockDim.z, 0, null, @ptrCast([*c]?*c_void, &args_ptrs), null);
        try check(res);
    }

    pub fn format(
        self: *const Cuda,
        comptime fmt: []const u8,
        options: std.fmt.FormatOptions,
        writer: anytype,
    ) !void {
        try std.fmt.format(writer, "Cuda(device={})", .{self.device});
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

    pub fn deinit(self: *GpuTimer) GpuTimer {
        _ = cu.cuEventDestroy(&self._start);
        _ = cu.cuEventDestroy(&self._stop);
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

fn your_rgba_to_greyscale(h_rgbaImage: *[4]u8, d_rgbaImage: *[4]u8, d_greyImage: *u8, numRows: c_int, numCols: c_int) void {
    // You must fill in the correct sizes for the blockSize and gridSize
    // currently only one block with one thread is being launched
    const gridDim = Dim3{ numRows, numCols, 1 };
    var args = .{ d_rgbaImage, d_greyImage, numRows, numCols };
    // launchKernel(rgba_to_greyscale, gridDim, .{}, &args);

    cudaDeviceSynchronize();
    checkCudaErrors(cudaGetLastError());
}

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
    try testing.expectEqual(cu.CUresult.CUDA_SUCCESS, cu.cuInit(0));
}

test "cuda init" {
    var cuda = try Cuda.init(0);
    defer cuda.deinit();
}

test "HW1" {
    var cuda = try Cuda.init(0);
    defer cuda.deinit();
    std.log.warn("cuda: {}", .{cuda});
    const rgba_to_greyscale = try cuda.kernel("./cudaz/kernel.ptx", "rgba_to_greyscale");
    const numRows: u32 = 10;
    const numCols: u32 = 20;
    const d_rgbaImage = cuda.alloc([4]u8, numRows * numCols);
    cuda.memset([4]u8, d_rgbaImage, 0xaa);
    const d_greyImage = cuda.alloc(u8, numRows * numCols);
    cuda.memset(u8, d_greyImage, 0);

    // copy input array to the GPU
    // checkCudaErrors(cudaMemcpy(*d_rgbaImage, *inputImage, sizeof(uchar4) * numPixels, cudaMemcpyHostToDevice));

    try cuda.launch(
        rgba_to_greyscale,
        .{ .x = numRows, .y = numCols },
        .{},
        .{ d_rgbaImage, d_greyImage, numRows, numCols },
    );
}

pub fn ArgsStruct(comptime Function: type) type {
    const ArgsTuple = meta.ArgsTuple(Function);
    var info = @typeInfo(ArgsTuple);
    info.Struct.is_tuple = false;
    return @Type(info);
}

pub fn KernelSignature(comptime ptx_file: [:0]const u8, comptime name: [:0]const u8) type {
    // TODO: I'm not fond of passing .ptx files, I'd prefer if we could only talk about .cu files
    return struct {
        const Self = @This();
        // const Args = comptime ArgsStruct(@TypeOf(@field(cu, name)));
        const Args = meta.ArgsTuple(@TypeOf(@field(cu, name)));

        f: cu.CUfunction,
        cuda: *Cuda,

        pub fn init(cuda: *Cuda) !Self {
            var k = Self{ .f = undefined, .cuda = cuda };
            k.f = try cuda.kernel(ptx_file, name);
            return k;
        }

        // TODO: deinit -> CUDestroy

        pub fn launch(self: *const Self, gridDim: Dim3, blockDim: Dim3, args: Args) !void {
            try self.cuda.launch(self.f, gridDim, blockDim, args);
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
    const ptx_file = "./cudaz/kernel.ptx";
    const rgba_to_greyscale = try cuda.kernel(ptx_file, "rgba_to_greyscale");
    const numRows: u32 = 10;
    const numCols: u32 = 20;
    const d_rgbaImage = cuda.alloc(cu.uchar4, numRows * numCols);
    cuda.memset(cu.uchar4, d_rgbaImage, 0xaa);
    const d_greyImage = cuda.alloc(u8, numRows * numCols);
    cuda.memset(u8, d_greyImage, 0);

    const rgba_to_greyscale_safe = try KernelSignature(ptx_file, "rgba_to_greyscale").init(&cuda);
    std.log.warn("kernel args: {s}", .{@TypeOf(rgba_to_greyscale_safe).Args});
    try rgba_to_greyscale_safe.launch(
        .{ .x = numRows, .y = numCols },
        .{},
        // this is so ugly ! can I do something about it ?
        // https://github.com/ziglang/zig/issues/8136
        .{ .@"0" = d_rgbaImage.ptr, .@"1" = d_greyImage.ptr, .@"2" = numRows, .@"3" = numCols },
    );
}

test "cuda alloc" {
    var cuda = try Cuda.init(0);
    defer cuda.deinit();

    const d_greyImage = cuda.alloc(u8, 128);
    cuda.memset(u8, d_greyImage, 0);
    defer cuda.free(d_greyImage);
}
