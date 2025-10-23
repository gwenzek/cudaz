const std = @import("std");
const builtin = @import("builtin");
const root = @import("root");

pub const cu = @import("cuda_h");

pub const algorithms = @import("algorithms.zig");
const attributes = @import("attributes.zig");
pub const Attribute = attributes.Attribute;
pub const getAttr = attributes.getAttr;
pub const cuda_errors = @import("cuda_errors.zig");
pub const check = cuda_errors.check;
pub const Error = cuda_errors.Error;

const log = std.log.scoped(.Cuda);

pub const Dim3 = struct {
    x: c_uint = 1,
    y: c_uint = 1,
    z: c_uint = 1,

    pub fn init(x: usize, y: usize, z: usize) Dim3 {
        return .{
            .x = @intCast(x),
            .y = @intCast(y),
            .z = @intCast(z),
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
        const t_x = if (threads_x == 0) cols else threads_x;
        const t_y = if (threads_y == 0) rows else threads_y;
        const t_z = if (threads_z == 0) depth else threads_z;
        return Grid{
            .blocks = Dim3.init(
                std.math.divCeil(usize, cols, t_x) catch unreachable,
                std.math.divCeil(usize, rows, t_y) catch unreachable,
                std.math.divCeil(usize, depth, t_z) catch unreachable,
            ),
            .threads = Dim3.init(t_x, t_y, t_z),
        };
    }

    pub fn threadsPerBlock(self: *const Grid) usize {
        return self.threads.x * self.threads.y * self.threads.z;
    }

    pub fn sharedMem(self: *const Grid, comptime ty: type, per_thread: usize) usize {
        return self.threadsPerBlock() * @sizeOf(ty) * per_thread;
    }
};

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

    pub fn deinit(self: *Stream) void {
        // Don't handle CUDA errors here
        _ = self.synchronize();
        _ = cu.cuStreamDestroy(self._stream);
        self._stream = undefined;
    }

    // TODO: can this OOM ? Or will the error be raised later ?
    pub fn alloc(self: *const Stream, comptime DestType: type, size: usize) ![]DestType {
        var int_ptr: cu.CUdeviceptr = undefined;
        const byte_size = size * @sizeOf(DestType);
        check(cu.cuMemAllocAsync(&int_ptr, byte_size, self._stream)) catch |err| {
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

    pub fn free(self: *const Stream, device_ptr: anytype) void {
        const raw_ptr: *anyopaque = if (@hasField(@TypeOf(device_ptr), "ptr"))
            @ptrCast(device_ptr.ptr)
        else
            @ptrCast(device_ptr);
        _ = cu.cuMemFreeAsync(@intFromPtr(raw_ptr), self._stream);
    }

    pub fn memcpyHtoD(self: *const Stream, comptime DestType: type, d_target: []DestType, h_source: []const DestType) void {
        std.debug.assert(h_source.len == d_target.len);
        check(cu.cuMemcpyHtoDAsync(
            @intFromPtr(d_target.ptr),
            @ptrCast(h_source.ptr),
            h_source.len * @sizeOf(DestType),
            self._stream,
        )) catch unreachable;
    }

    pub fn memcpyDtoH(self: *const Stream, comptime DestType: type, h_target: []DestType, d_source: []const DestType) void {
        std.debug.assert(d_source.len == h_target.len);
        check(cu.cuMemcpyDtoHAsync(
            @ptrCast(h_target.ptr),
            @intFromPtr(d_source.ptr),
            d_source.len * @sizeOf(DestType),
            self._stream,
        )) catch unreachable;
        // The only cause of failures here are segfaults or hardware issues,
        // can't recover.
    }

    pub fn allocAndCopy(self: *const Stream, comptime DestType: type, h_source: []const DestType) ![]DestType {
        const ptr = try self.alloc(DestType, h_source.len);
        self.memcpyHtoD(DestType, ptr, h_source);
        return ptr;
    }

    pub fn allocAndCopyResult(
        self: *const Stream,
        comptime DestType: type,
        host_allocator: std.mem.Allocator,
        d_source: []const DestType,
    ) ![]DestType {
        const h_tgt = try host_allocator.alloc(DestType, d_source.len);
        self.memcpyDtoH(DestType, h_tgt, d_source);
        return h_tgt;
    }

    pub fn memset(self: *const Stream, comptime DestType: type, slice: []DestType, value: DestType) void {
        const d_ptr = @intFromPtr(slice.ptr);
        const n = slice.len;
        const memset_res = switch (@sizeOf(DestType)) {
            1 => cu.cuMemsetD8Async(d_ptr, @bitCast(value), n, self._stream),
            2 => cu.cuMemsetD16Async(d_ptr, @bitCast(value), n, self._stream),
            4 => cu.cuMemsetD32Async(d_ptr, @bitCast(value), n, self._stream),
            else => @compileError("memset doesn't support type: " ++ @typeName(DestType)),
        };
        check(memset_res) catch unreachable;
    }

    pub inline fn launch(self: *const Stream, f: cu.CUfunction, grid: Grid, args: anytype) !void {
        try self.launchWithSharedMem(f, grid, 0, args);
    }

    pub fn launchWithSharedMem(self: *const Stream, f: cu.CUfunction, grid: Grid, shared_mem: usize, args: anytype) !void {
        // Create an array of pointers pointing to the given args.
        const fields: []const std.builtin.Type.StructField = std.meta.fields(@TypeOf(args));
        var args_ptrs: [fields.len:0]usize = undefined;
        inline for (fields, 0..) |field, i| {
            args_ptrs[i] = @intFromPtr(&@field(args, field.name));
        }
        const res = cu.cuLaunchKernel(
            f,
            grid.blocks.x,
            grid.blocks.y,
            grid.blocks.z,
            grid.threads.x,
            grid.threads.y,
            grid.threads.z,
            @intCast(shared_mem),
            self._stream,
            @ptrCast(&args_ptrs),
            null,
        );
        try check(res);
        // TODO use callback API to keep the asynchronous scheduling
    }

    pub fn synchronize(self: *const Stream) void {
        check(cu.cuStreamSynchronize(self._stream)) catch unreachable;
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

    // TODO: I'd like to have an async method that suspends until the stream is over.
    // Currently the best way too achieve something like this is to `suspend {} stream.synchronize();`
    // once the stream is scheduled, and then `resume` once you are ready to wait
    // for the blocking `synchronize` call.
    // Ideally we would have an event loop that poll streams to check
    // if they are over.
    pub fn done(self: *Stream) bool {
        const res = cu.cuStreamQuery(self._stream);
        return res != cu.CUDA_ERROR_NOT_READY;
    }
};

// TODO: return a device pointer
pub fn alloc(comptime DestType: type, size: usize) ![]DestType {
    var int_ptr: cu.CUdeviceptr = undefined;
    const byte_size = size * @sizeOf(DestType);
    check(cu.cuMemAlloc(&int_ptr, byte_size)) catch |err| {
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

// TODO: move all this to stream using async variants
pub fn free(device_ptr: anytype) void {
    const raw_ptr: *anyopaque = if (@hasField(@TypeOf(device_ptr), "ptr"))
        @ptrCast(device_ptr.ptr)
    else
        @ptrCast(device_ptr);
    _ = cu.cuMemFree(@intFromPtr(raw_ptr));
}

pub fn memset(comptime DestType: type, slice: []DestType, value: DestType) !void {
    const d_ptr = @intFromPtr(slice.ptr);
    const n = slice.len;
    const memset_res = switch (@sizeOf(DestType)) {
        1 => cu.cuMemsetD8(d_ptr, @bitCast(value), n),
        2 => cu.cuMemsetD16(d_ptr, @bitCast(value), n),
        4 => cu.cuMemsetD32(d_ptr, @bitCast(value), n),
        else => @compileError("memset doesn't support type: " ++ @typeName(DestType)),
    };
    try check(memset_res);
}

pub fn memsetD8(comptime DestType: type, slice: []DestType, value: u8) !void {
    const d_ptr = @intFromPtr(slice.ptr);
    const n = slice.len * @sizeOf(DestType);
    try check(cu.cuMemsetD8(d_ptr, value, n));
}

// pub fn allocAndCopy(comptime DestType: type, h_source: []const DestType) ![]DestType {
//     const ptr = try alloc(DestType, h_source.len);
//     try memcpyHtoD(DestType, ptr, h_source);
//     return ptr;
// }

// pub fn allocAndCopyResult(
//     comptime DestType: type,
//     host_allocator: std.mem.Allocator,
//     d_source: []const DestType,
// ) ![]DestType {
//     const h_tgt = try host_allocator.alloc(DestType, d_source.len);
//     try memcpyDtoH(DestType, h_tgt, d_source);
//     return h_tgt;
// }

// pub fn readResult(comptime DestType: type, d_source: *const DestType) !DestType {
//     var h_res: [1]DestType = undefined;
//     try check(cu.cuMemcpyDtoH(
//         @ptrCast(&h_res),
//         @intFromPtr(d_source),
//         @sizeOf(DestType),
//     ));
//     return h_res[0];
// }

// pub fn memcpyHtoD(comptime DestType: type, d_target: []DestType, h_source: []const DestType) !void {
//     std.debug.assert(h_source.len == d_target.len);
//     try check(cu.cuMemcpyHtoD(
//         @intFromPtr(d_target.ptr),
//         @ptrCast(h_source.ptr),
//         h_source.len * @sizeOf(DestType),
//     ));
// }
// pub fn memcpyDtoH(comptime DestType: type, h_target: []DestType, d_source: []const DestType) !void {
//     std.debug.assert(d_source.len == h_target.len);
//     try check(cu.cuMemcpyDtoH(
//         @ptrCast(h_target.ptr),
//         @intFromPtr(d_source.ptr),
//         d_source.len * @sizeOf(DestType),
//     ));
// }

// pub fn push(value: anytype) !*@TypeOf(value) {
//     const DestType = @TypeOf(value);
//     const d_ptr = try alloc(DestType, 1);
//     try check(cu.cuMemcpyHtoD(
//         @intFromPtr(d_ptr.ptr),
//         @ptrCast(&value),
//         @sizeOf(DestType),
//     ));
//     return @ptrCast(d_ptr.ptr);
// }

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

    pub fn start(stream: *const Stream) GpuTimer {
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
        var _elapsed = std.math.nan_f32;
        // _ = cu.cuEventSynchronize(self._stop);
        _ = cu.cuEventElapsedTime(&_elapsed, self._start, self._stop);
        self.deinit();
        self._elapsed = _elapsed;
        if (_elapsed < 1e-3) log.warn("Cuda events only have 0.5 microseconds of resolution, so this might not be precise", .{});
        return _elapsed;
    }
};

pub fn main() anyerror!void {
    var general_purpose_allocator = std.heap.GeneralPurposeAllocator(.{}){};
    const gpa = &general_purpose_allocator.allocator;
    const args = try std.process.argsAlloc(gpa);
    defer std.process.argsFree(gpa, args);

    log.info("All your codebase are belong to us.", .{});

    log.info("cuda: {}", .{cu.cuInit});
    log.info("cuInit: {}", .{cu.cuInit(0)});
}

test "cuda version" {
    log.warn("Cuda version: {d}", .{cu.CUDA_VERSION});
    try std.testing.expect(cu.CUDA_VERSION > 11000);
    try std.testing.expectEqual(cu.cuInit(0), cu.CUDA_SUCCESS);
}

var _device = [_]cu.CUdevice{-1} ** 8;
pub fn initDevice(device: u3) !cu.CUdevice {
    const cu_dev = &_device[device];
    if (cu_dev.* == -1) {
        try check(cu.cuInit(0));
        try check(cu.cuDeviceGet(cu_dev, device));
    }
    return cu_dev.*;
}

/// Returns the ctx for the given device
/// Given that we already assume one program == one module,
/// we can also assume one program == one context per GPU.
/// From Nvidia doc:
/// A host thread may have only one device context current at a time.
// TODO: who is responsible for destroying the context ?
// we should use cuCtxAttach and cuCtxDetach in stream init/deinit
var _ctx = [1]cu.CUcontext{null} ** 8;
fn getCtx(device: u3, cu_dev: cu.CUdevice) !cu.CUcontext {
    const cu_ctx = &_ctx[device];
    if (cu_ctx.* == null) {
        try check(cu.cuCtxCreate(cu_ctx, 0, cu_dev));
    }
    return cu_ctx.*;
}

pub const PtxLocation = union(enum) {
    embed: [:0]const u8,
    file: [:0]const u8,
};

pub fn loadModule(location: PtxLocation) cu.CUmodule {
    var module: cu.CUmodule = undefined;
    switch (location) {
        .embed => |content| {
            // log.info("Loading Cuda module from embedded file.", .{});
            // TODO see if we can use nvPTXCompiler to explicitly compile it ourselve
            check(cu.cuModuleLoadData(&module, content.ptr)) catch |err| {
                std.debug.panic("Couldn't load embedded cuda module: {}", .{err});
            };
        },
        .file => |path| {
            // log.warn("Loading Cuda module from local file {s}", .{file});
            check(cu.cuModuleLoad(&module, path.ptr)) catch |err| {
                std.debug.panic("Couldn't load cuda module {s}: {}", .{ path, err });
            };
        },
    }
    return module;
}

/// Create a function with the correct signature for a cuda Kernel.
/// The kernel must come from the default .cu file
pub inline fn CudaKernel(comptime name: [:0]const u8) type {
    return FnStruct(name, @field(cu, name));
}

pub inline fn ZigKernel(comptime Module: anytype, comptime name: [:0]const u8) type {
    return FnStruct(name, @field(Module, name));
}

pub fn FnStruct(comptime name: []const u8, comptime func: anytype) type {
    return struct {
        const Self = @This();
        const CpuFn = *const @TypeOf(func);
        pub const Args = std.meta.ArgsTuple(@TypeOf(func));

        f: cu.CUfunction,

        pub fn init(module: cu.CUmodule) !Self {
            var f: cu.CUfunction = undefined;
            const code = cu.cuModuleGetFunction(&f, module, @ptrCast(name));
            if (code != cu.CUDA_SUCCESS) log.err("Couldn't load function {s}", .{name});
            try check(code);
            return .{ .f = f };
        }

        // TODO: deinit -> CUDestroy

        // Note: I'm not fond of having the primary launch be on the Function object,
        // but it works best with Zig type inference
        pub inline fn launch(self: *const Self, stream: *const Stream, grid: Grid, args: Args) !void {
            if (args.len != @typeInfo(Args).@"struct".fields.len) {
                @compileError("Expected more arguments");
            }
            try self.launchWithSharedMem(stream, grid, 0, args);
        }

        pub fn launchWithSharedMem(self: *const Self, stream: *const Stream, grid: Grid, shared_mem: usize, args: Args) !void {
            // TODO: this seems error prone, could we make the type of the shared buffer
            // part of the function signature ?
            try stream.launchWithSharedMem(self.f, grid, shared_mem, args);
        }

        // pub fn debugCpuCall(grid: Grid, point: Grid, args: Args) void {
        //     cu.threadIdx = point.threads.dim3();
        //     cu.blockDim = grid.threads.dim3();
        //     cu.blockIdx = point.blocks.dim3();
        //     cu.gridDim = grid.blocks.dim3();
        //     _ = @call(.{}, CpuFn, args);
        // }

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

test "we use only one context per GPU" {
    var default_ctx: cu.CUcontext = undefined;
    try check(cu.cuCtxGetCurrent(&default_ctx));
    std.log.warn("default_ctx: {any}", .{std.mem.asBytes(&default_ctx).*});

    const stream = try Stream.init(0);
    var stream_ctx: cu.CUcontext = undefined;
    try check(cu.cuStreamGetCtx(stream._stream, &stream_ctx));
    std.log.warn("stream_ctx: {any}", .{std.mem.asBytes(&stream_ctx).*});
    // try std.testing.expectEqual(default_ctx, stream_ctx);

    // Create a new stream
    const stream2 = try Stream.init(0);
    var stream2_ctx: cu.CUcontext = undefined;
    try check(cu.cuStreamGetCtx(stream2._stream, &stream2_ctx));
    std.log.warn("stream2_ctx: {any}", .{std.mem.asBytes(&stream2_ctx).*});
    try std.testing.expectEqual(stream_ctx, stream2_ctx);
}

test "cuda alloc" {
    var stream = try Stream.init(0);
    defer stream.deinit();

    const d_greyImage = try alloc(u8, 128);
    try memset(u8, d_greyImage, 0);
    defer free(d_greyImage);
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
            .field_type = FnStruct(decl.name, decl.data.Fn.fn_type),
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
