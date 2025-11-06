const std = @import("std");
const builtin = @import("builtin");

pub const cu = @import("cuda_h");

pub const algorithms = @import("algorithms.zig");
const attributes = @import("attributes.zig");
pub const Attribute = attributes.Attribute;
pub const getAttr = attributes.getAttr;
pub const errors = @import("errors.zig");
pub const check = errors.check;
pub const Error = errors.Error;

const log = std.log.scoped(.cuda);

var _devices: [8]cu.CUdevice = @splat(-1);
var _device_info: [8]DeviceInfo = undefined;

/// Returns the ctx for the given device
/// We assume one program == one context per GPU.
/// From Nvidia doc:
/// A host thread may have only one device context current at a time.
// TODO: who is responsible for destroying the context ?
var _ctx: [8]cu.CUcontext = @splat(null);
fn getCtx(device: u3, cu_dev: cu.CUdevice) !cu.CUcontext {
    const cu_ctx = &_ctx[device];
    if (cu_ctx.* == null) {
        try check(cu.cuCtxCreate(cu_ctx, 0, cu_dev));
    }
    return cu_ctx.*;
}

/// Represents a Cuda stream associated to a specific device.
pub const Stream = struct {
    device: cu.CUdevice,
    _stream: *cu.CUstream_st,

    pub fn init(device: u3) !Stream {
        const cu_dev = try initDevice(device);
        _ = try getCtx(device, cu_dev);
        var stream: cu.CUstream = undefined;
        check(cu.cuStreamCreate(&stream, cu.CU_STREAM_DEFAULT)) catch |err| switch (err) {
            error.NotSupported => return error.NotSupported,
            else => unreachable,
        };
        return Stream{ .device = cu_dev, ._stream = stream.? };
    }

    pub fn deinit(stream: *Stream) void {
        // Don't handle CUDA errors here
        _ = stream.synchronize();
        _ = cu.cuStreamDestroy(stream._stream);
        stream._stream = undefined;
    }

    // TODO: can this OOM ? Or will the error be raised later ?
    pub fn alloc(stream: Stream, comptime DestType: type, size: usize) ![]DestType {
        var int_ptr: cu.CUdeviceptr = undefined;
        const byte_size = size * @sizeOf(DestType);
        check(cu.cuMemAllocAsync(&int_ptr, byte_size, stream._stream)) catch |err| {
            switch (err) {
                error.OutOfMemory => {
                    var free_mem: usize = undefined;
                    var total_mem: usize = undefined;
                    const mb = 1024 * 1024;
                    check(cu.cuMemGetInfo(&free_mem, &total_mem)) catch return err;
                    log.err(
                        "Cuda OutOfMemory: tried to allocate {d:.1}Mb, free {d:.1}Mb, total {d:.1}Mb",
                        .{ byte_size / mb, free_mem / mb, total_mem / mb },
                    );
                    return err;
                },
                else => unreachable,
            }
        };
        const ptr: [*]DestType = @ptrFromInt(int_ptr);
        return ptr[0..size];
    }

    pub fn free(stream: Stream, device_ptr: anytype) void {
        const raw_ptr: *anyopaque = if (@hasField(@TypeOf(device_ptr), "ptr"))
            @ptrCast(device_ptr.ptr)
        else
            @ptrCast(device_ptr);
        _ = cu.cuMemFreeAsync(@intFromPtr(raw_ptr), stream._stream);
    }

    pub fn memcpyHtoD(stream: Stream, comptime DestType: type, d_target: []DestType, h_source: []const DestType) void {
        std.debug.assert(h_source.len == d_target.len);
        check(cu.cuMemcpyHtoDAsync(
            @intFromPtr(d_target.ptr),
            @ptrCast(h_source.ptr),
            h_source.len * @sizeOf(DestType),
            stream._stream,
        )) catch unreachable;
    }

    pub fn memcpyDtoH(stream: Stream, comptime DestType: type, h_target: []DestType, d_source: []const DestType) void {
        std.debug.assert(d_source.len == h_target.len);
        check(cu.cuMemcpyDtoHAsync(
            @ptrCast(h_target.ptr),
            @intFromPtr(d_source.ptr),
            d_source.len * @sizeOf(DestType),
            stream._stream,
        )) catch unreachable;
        // The only cause of failures here are segfaults or hardware issues,
        // can't recover.
    }

    /// Allocate a device buffer and copy the give host data into it.
    pub fn allocAndCopy(stream: Stream, comptime DestType: type, h_source: []const DestType) ![]DestType {
        const ptr = try stream.alloc(DestType, h_source.len);
        stream.memcpyHtoD(DestType, ptr, h_source);
        return ptr;
    }

    pub fn allocAndCopyResult(
        stream: Stream,
        comptime DestType: type,
        host_allocator: std.mem.Allocator,
        d_source: []const DestType,
    ) ![]DestType {
        const h_tgt = try host_allocator.alloc(DestType, d_source.len);
        stream.memcpyDtoH(DestType, h_tgt, d_source);
        return h_tgt;
    }

    pub fn memset(stream: Stream, comptime DestType: type, slice: []DestType, value: DestType) void {
        const d_ptr = @intFromPtr(slice.ptr);
        const n = slice.len;
        const memset_res = switch (@sizeOf(DestType)) {
            1 => cu.cuMemsetD8Async(d_ptr, @bitCast(value), n, stream._stream),
            2 => cu.cuMemsetD16Async(d_ptr, @bitCast(value), n, stream._stream),
            4 => cu.cuMemsetD32Async(d_ptr, @bitCast(value), n, stream._stream),
            else => @compileError("memset doesn't support type: " ++ @typeName(DestType)),
        };
        check(memset_res) catch unreachable;
    }

    pub fn launch(stream: Stream, f: cu.CUfunction, grid: Grid, params_buffer: []const u8) !void {
        try stream.launchWithSharedMem(f, grid, 0, params_buffer);
    }

    /// Launch the given kernel.
    pub fn launchWithSharedMem(stream: Stream, f: cu.CUfunction, grid: Grid, shared_mem: usize, params_buffer: []const u8) !void {
        // This check is optional cause it's only here to provide a better error message than just CUDA_ERROR_INVALID_VALUE
        if (builtin.mode == .Debug) {
            const gpu_id = std.mem.indexOfScalar(cu.CUdevice, _devices[0..], stream.device);
            if (gpu_id) |d| {
                const info = _device_info[d];
                if (!info.gridIsOk(grid)) {
                    std.debug.panic("Cuda launch kernel failed ! Grid is too big. Device constraint: {any}, received: {any}", .{ info, grid });
                }
                // log.debug("Grid constraint: {}", .{info});
            }

            log.debug("Launching kernel {x} with grid: {any}, params: {x}", .{ @intFromPtr(f), grid, @intFromPtr(params_buffer.ptr) });
        }

        // Note: There are two ways to pass kernel arguments, this way aligns better with Zig,
        // telling the driver to copy the full args struct somewhere safe.
        var extras = [_]?*anyopaque{
            cu.CU_LAUNCH_PARAM_BUFFER_POINTER,
            @constCast(params_buffer.ptr),
            cu.CU_LAUNCH_PARAM_BUFFER_SIZE,
            @constCast(&params_buffer.len),
            cu.CU_LAUNCH_PARAM_END,
            null,
        };

        // TODO: switch to cuLaunchKernelEx and merge launch and launchWithSharedMem.
        const res = cu.cuLaunchKernel(
            f,
            grid.blocks.x,
            grid.blocks.y,
            grid.blocks.z,
            grid.threads.x,
            grid.threads.y,
            grid.threads.z,
            @intCast(shared_mem),
            stream._stream,
            null,
            &extras,
        );
        try check(res);
        // TODO use callback API to keep the asynchronous scheduling
    }

    pub fn synchronize(stream: Stream) void {
        check(cu.cuStreamSynchronize(stream._stream)) catch unreachable;
    }

    pub fn format(stream: Stream, writer: *std.Io.Writer) std.Io.Writer.Error!void {
        try writer.print("CuStream(device={}, stream={*})", .{ stream.device, stream._stream });
    }

    // TODO: I'd like to have an async method that suspends until the stream is over.
    pub fn done(stream: *Stream) bool {
        const res = cu.cuStreamQuery(stream._stream);
        return res != cu.CUDA_ERROR_NOT_READY;
    }
};

test "cuda version" {
    log.warn("Cuda version: {d}", .{cu.CUDA_VERSION});
    try std.testing.expect(cu.CUDA_VERSION > 11000);
    try std.testing.expectEqual(cu.cuInit(0), cu.CUDA_SUCCESS);
}

pub const Module = opaque {
    pub fn initFromData(ptx_data: [:0]const u8) *Module {
        var module: cu.CUmodule = undefined;
        check(cu.cuModuleLoadData(&module, ptx_data.ptr)) catch |err| {
            std.debug.panic("Couldn't load embedded cuda module: {}", .{err});
        };
        return @ptrCast(module);
    }

    pub fn initFromFile(path: [:0]const u8) *Module {
        var module: cu.CUmodule = undefined;
        check(cu.cuModuleLoad(&module, path.ptr)) catch |err| {
            std.debug.panic("Couldn't load cuda module {s}: {}", .{ path, err });
        };
        return @ptrCast(module);
    }

    pub fn deinit(module: *Module) void {
        _ = cu.cuModuleUnload(@ptrCast(module));
    }
};

pub fn initDevice(device: u3) !cu.CUdevice {
    const cu_dev = &_devices[device];
    if (cu_dev.* == -1) {
        try check(cu.cuInit(0));
        try check(cu.cuDeviceGet(cu_dev, device));

        _device_info[device] = .init(cu_dev.*);
    }
    return cu_dev.*;
}

const DeviceInfo = struct {
    max_num_blocks: Dim3,
    max_num_threads: Dim3,
    max_threads_per_block: u32,

    pub fn init(d: cu.CUdevice) DeviceInfo {
        return .{
            .max_num_blocks = .{
                .x = getAttr(d, .MaxGridDimX),
                .y = getAttr(d, .MaxGridDimY),
                .z = getAttr(d, .MaxGridDimZ),
            },
            .max_num_threads = .{
                .x = getAttr(d, .MaxBlockDimX),
                .y = getAttr(d, .MaxBlockDimY),
                .z = getAttr(d, .MaxBlockDimZ),
            },
            .max_threads_per_block = getAttr(d, .MaxThreadsPerBlock),
        };
    }

    pub fn gridIsOk(info: DeviceInfo, grid: Grid) bool {
        return (info.max_num_blocks.x >= grid.blocks.x //
        and info.max_num_blocks.y >= grid.blocks.y //
        and info.max_num_blocks.z >= grid.blocks.z //
        and info.max_num_threads.x >= grid.threads.x //
        and info.max_num_threads.y >= grid.threads.y //
        and info.max_num_threads.z >= grid.threads.z //
        and info.max_threads_per_block >= grid.threads.x * grid.threads.y * grid.threads.z //
        );
    }
};

pub inline fn Kernel(comptime ZigModule: anytype, comptime name: [:0]const u8) type {
    return TypedKernel(name, @field(ZigModule, name));
}

pub fn TypedKernel(comptime name: []const u8, comptime func: anytype) type {
    return struct {
        const K = @This();
        const CpuFn = *const @TypeOf(func);
        pub const Args = std.meta.ArgsTuple(@TypeOf(func));
        pub const num_args = @typeInfo(Args).@"struct".fields.len;

        f: cu.CUfunction,

        const arg_offsets: [num_args]usize = offs: {
            var offs: [num_args]usize = undefined;
            for (offs[0..], @typeInfo(Args).@"struct".fields) |*o, field| {
                o.* = @offsetOf(Args, field.name);
            }
            break :offs offs;
        };

        pub fn init(module: *Module) !K {
            var f: cu.CUfunction = undefined;
            const code = cu.cuModuleGetFunction(&f, @ptrCast(module), @ptrCast(name));
            if (code != cu.CUDA_SUCCESS) log.err("Couldn't load function {s}", .{name});
            try check(code);
            const kernel: K = .{ .f = f };

            if (builtin.mode == .Debug) kernel.assertParamsLayout();
            return kernel;
        }

        // TODO: deinit -> CUDestroy

        pub fn launch(self: K, stream: Stream, grid: Grid, args: Args) !void {
            try self.launchWithSharedMem(stream, grid, 0, args);
        }

        pub fn launchWithSharedMem(self: K, stream: Stream, grid: Grid, shared_mem: usize, args: Args) !void {
            try stream.launchWithSharedMem(self.f, grid, shared_mem, std.mem.asBytes(&args));
        }

        /// Returns the offset and size of a kernel parameter in the device-side parameter layout
        pub fn paramInfo(kernel: K, param_index: u32) !struct { usize, usize } {
            var param_offset: usize = undefined;
            var param_size: usize = undefined;
            const rc = cu.cuFuncGetParamInfo(kernel.f, param_index, &param_offset, &param_size);
            try check(rc);
            return .{ param_offset, param_size };
        }

        pub fn assertParamsLayout(kernel: K) void {
            inline for (0..num_args, arg_offsets, @typeInfo(Args).@"struct".fields) |i, zig_offset, arg| {
                const cuda_offset, const cuda_sizeof = kernel.paramInfo(i) catch @panic("too many arguments on Zig side");
                // log.debug("Argument {} ({s}): offset {}, size: {}", .{ i, @typeName(arg.type), cuda_offset, cuda_sizeof });
                std.debug.assert(cuda_offset == zig_offset); // layout mismatch
                std.debug.assert(cuda_sizeof == @sizeOf(arg.type)); // size mismatch
            }

            // Try one more time to get param, info. It should fail now.
            var param_offset: usize = undefined;
            const rc = cu.cuFuncGetParamInfo(kernel.f, num_args, &param_offset, null);
            std.debug.assert(rc == cu.CUDA_ERROR_INVALID_VALUE); // missing arguments on Zig side
        }

        // pub fn debugCpuCall(grid: Grid, point: Grid, args: Args) void {
        //     cu.threadIdx = point.threads.dim3();
        //     cu.blockDim = grid.threads.dim3();
        //     cu.blockIdx = point.blocks.dim3();
        //     cu.gridDim = grid.blocks.dim3();
        //     _ = @call(.{}, CpuFn, args);
        // }

        pub fn format(_: K, writer: *std.Io.Writer) std.Io.Writer.Error!void {
            try writer.print("{s}(", .{name});
            inline for (@typeInfo(Args).Struct.fields) |arg| {
                const ArgT = arg.field_type;
                try writer.print("{}, ", .{ArgT});
            }
            try writer.print(")", .{});
        }
    };
}

pub const Dim3 = extern struct {
    x: c_uint,
    y: c_uint,
    z: c_uint,

    pub fn init(x: usize, y: usize, z: usize) Dim3 {
        return .{
            .x = @intCast(x),
            .y = @intCast(y),
            .z = @intCast(z),
        };
    }
};

/// Represents how kernel are execut
pub const Grid = struct {
    blocks: Dim3,
    threads: Dim3,

    /// Divide in blocks the given len by the given number of threads.
    pub fn init1D(full_len: usize, threads: usize) Grid {
        return init3D(.{ full_len, 1, 1 }, .{ threads, 1, 1 });
    }

    /// Divide in blocks the given shape by the given number of threads.
    pub fn init2D(full_shape: [2]usize, threads: [2]usize) Grid {
        return init3D(.{ full_shape[0], full_shape[1], 1 }, .{ threads[0], threads[1], 1 });
    }

    /// Divide in blocks the given shape by the given number of threads.
    pub fn init3D(
        full_shape: [3]usize,
        threads: [3]usize,
    ) Grid {
        return .{
            .blocks = Dim3.init(
                std.math.divCeil(usize, full_shape[0], threads[0]) catch unreachable,
                std.math.divCeil(usize, full_shape[1], threads[1]) catch unreachable,
                std.math.divCeil(usize, full_shape[2], threads[2]) catch unreachable,
            ),
            .threads = Dim3.init(threads[0], threads[1], threads[2]),
        };
    }

    pub fn threadsPerBlock(self: Grid) usize {
        return self.threads.x * self.threads.y * self.threads.z;
    }

    pub fn sharedMem(self: Grid, comptime ty: type, per_thread: usize) usize {
        return self.threadsPerBlock() * @sizeOf(ty) * per_thread;
    }
};

test "we use only one context per GPU" {
    var default_ctx: cu.CUcontext = undefined;
    try check(cu.cuCtxGetCurrent(&default_ctx));
    // std.log.warn("default_ctx: {any}", .{std.mem.asBytes(&default_ctx).*});

    const stream = try Stream.init(0);
    var stream_ctx: cu.CUcontext = undefined;
    try check(cu.cuStreamGetCtx(stream._stream, &stream_ctx));
    // std.log.warn("stream_ctx: {any}", .{std.mem.asBytes(&stream_ctx).*});
    // try std.testing.expectEqual(default_ctx, stream_ctx);

    // Create a new stream
    const stream2 = try Stream.init(0);
    var stream2_ctx: cu.CUcontext = undefined;
    try check(cu.cuStreamGetCtx(stream2._stream, &stream2_ctx));
    // std.log.warn("stream2_ctx: {any}", .{std.mem.asBytes(&stream2_ctx).*});
    try std.testing.expectEqual(stream_ctx, stream2_ctx);
}

test "cuda alloc" {
    var stream = try Stream.init(0);
    defer stream.deinit();

    const d_greyImage = try stream.alloc(u8, 128);
    stream.memset(u8, d_greyImage, 0);
    defer stream.free(d_greyImage);
}

pub fn Kernels(comptime module: type) type {
    // @compileLog(@typeName(module));
    const decls = @typeInfo(module).Struct.decls;
    var kernels: [decls.len]std.builtin.Type.StructField = undefined;
    comptime var kernels_count = 0;
    inline for (decls) |decl| {
        if (decl.data != .Fn or !decl.data.Fn.is_export) continue;
        kernels[kernels_count] = .{
            .name = decl.name,
            .field_type = TypedKernel(decl.name, decl.data.Fn.fn_type),
            .default_value = null,
            .is_comptime = false,
            .alignment = @alignOf(cu.CUfunction),
        };
        kernels_count += 1;
    }
    // @compileLog(kernels_count);
    return @Type(.{
        .@"struct" = .{
            .is_tuple = false,
            .layout = .Auto,
            .decls = &.{},
            .fields = kernels[0..kernels_count],
        },
    });
}

pub fn loadKernels(comptime module: type) Kernels(module) {
    const KernelType = Kernels(module);

    var kernels: KernelType = undefined;
    inline for (std.meta.fields(KernelType)) |field| {
        @field(kernels, field.name) = field.field_type.init() catch unreachable;
    }
    return kernels;
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
    stream: Stream,
    _elapsed: f32 = std.math.nan(f32),

    pub fn start(stream: Stream) GpuTimer {
        // The cuEvent are implicitly reffering to the current context.
        // We don't know if the current context is the same than the stream context.
        // Typically I'm not sure what happens with 2 streams on 2 gpus.
        // We might need to restore the stream context before creating the events.
        var timer = GpuTimer{ ._start = undefined, ._stop = undefined, .stream = stream };
        _ = cu.cuEventCreate(&timer._start, 0);
        _ = cu.cuEventCreate(&timer._stop, 0);
        _ = cu.cuEventRecord(timer._start, stream._stream);
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

    pub fn stop(self: *GpuTimer) void {
        _ = cu.cuEventRecord(self._stop, self.stream._stream);
    }

    /// Return the elapsed time in milliseconds.
    /// Resolution is around 0.5 microseconds.
    pub fn elapsed(self: *GpuTimer) f32 {
        if (!std.math.isNan(self._elapsed)) return self._elapsed;
        var _elapsed = std.math.nan(f32);
        // _ = cu.cuEventSynchronize(self._stop);
        _ = cu.cuEventElapsedTime(&_elapsed, self._start, self._stop);
        self.deinit();
        self._elapsed = _elapsed;
        if (_elapsed < 1e-3) log.warn("Cuda events only have 0.5 microseconds of resolution, so this might not be precise", .{});
        return _elapsed;
    }
};
