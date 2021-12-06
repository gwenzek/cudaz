const std = @import("std");
const builtin = @import("builtin");
const meta = std.meta;
const testing = std.testing;
const TypeInfo = std.builtin.TypeInfo;

const cudaz_options = @import("cudaz_options");
pub const cu = @import("cuda_cimports.zig").cu;
pub const cuda_errors = @import("cuda_errors.zig");
pub const check = cuda_errors.check;
pub const Error = cuda_errors.Error;
const attributes = @import("attributes.zig");
pub const Attributes = attributes.Attributes;
pub const getAttr = attributes.getAttr;

pub const kernel_ptx_content = if (cudaz_options.portable) @embedFile(cudaz_options.kernel_ptx_path) else [0:0]u8{};
const log = std.log.scoped(.Cuda);

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
        const cu_dev = try getDevice(device);
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
        var ptr = @intToPtr([*]DestType, int_ptr);
        return ptr[0..size];
    }

    pub fn free(self: *const Stream, device_ptr: anytype) void {
        var raw_ptr: *c_void = if (meta.trait.isSlice(@TypeOf(device_ptr)))
            @ptrCast(*c_void, device_ptr.ptr)
        else
            @ptrCast(*c_void, device_ptr);
        _ = cu.cuMemFreeAsync(@ptrToInt(raw_ptr), self._stream);
    }

    pub fn memcpyHtoD(self: *const Stream, comptime DestType: type, d_target: []DestType, h_source: []const DestType) !void {
        std.debug.assert(h_source.len == d_target.len);
        try check(cu.cuMemcpyHtoDAsync(
            @ptrToInt(d_target.ptr),
            @ptrCast(*const c_void, h_source.ptr),
            h_source.len * @sizeOf(DestType),
            self._stream,
        ));
    }

    pub fn memcpyDtoH(self: *const Stream, comptime DestType: type, h_target: []DestType, d_source: []const DestType) !void {
        std.debug.assert(d_source.len == h_target.len);
        try check(cu.cuMemcpyDtoHAsync(
            @ptrCast(*c_void, h_target.ptr),
            @ptrToInt(d_source.ptr),
            d_source.len * @sizeOf(DestType),
            self._stream,
        ));
    }

    pub fn allocAndCopyResult(
        self: *const Stream,
        comptime DestType: type,
        host_allocator: *std.mem.Allocator,
        d_source: []const DestType,
    ) ![]DestType {
        var h_tgt = try host_allocator.alloc(DestType, d_source.len);
        try self.memcpyDtoH(DestType, h_tgt, d_source);
        return h_tgt;
    }

    pub fn memset(self: *const Stream, comptime DestType: type, slice: []DestType, value: DestType) !void {
        var d_ptr = @ptrToInt(slice.ptr);
        var n = slice.len;
        var memset_res = switch (@sizeOf(DestType)) {
            1 => cu.cuMemsetD8Async(d_ptr, @bitCast(u8, value), n, self._stream),
            2 => cu.cuMemsetD16Async(d_ptr, @bitCast(u16, value), n, self._stream),
            4 => cu.cuMemsetD32Async(d_ptr, @bitCast(u32, value), n, self._stream),
            else => @compileError("memset doesn't support type: " ++ @typeName(DestType)),
        };
        try check(memset_res);
    }

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
        if (builtin.mode == .Debug) {
            // In CUDA operation are asynchronous.
            // The consequence is that an error in a kernel will only be
            // returned later on. In debug mode we want to know which
            // kernel is responsible for which error, so we have to wait
            // for this kernel to end before scheduling another.
            // TODO use callback API to keep the asynchronous scheduling
            self.synchronize();
        }
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

    pub fn asyncWait(self: *Stream) void {
        // We need to discard the const frame pointer to call the Cuda api.
        suspend {
            var frame = @frame();
            var frame_ptr = @intToPtr(*c_void, @ptrToInt(frame));
            check(cu.cuLaunchHostFunc(self._stream, resumeFrameAfterStream, frame_ptr)) catch unreachable;
        }
    }
};

fn resumeFrameAfterStream(frame_ptr: ?*c_void) callconv(.C) void {
    const frame = @ptrCast(anyframe, @alignCast(8, frame_ptr.?));
    resume frame;
}

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
    var ptr = @intToPtr([*]DestType, int_ptr);
    return ptr[0..size];
}

// TODO: move all this to stream using async variants
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

pub fn allocAndCopyResult(
    comptime DestType: type,
    host_allocator: *std.mem.Allocator,
    d_source: []const DestType,
) ![]DestType {
    var h_tgt = try host_allocator.alloc(DestType, d_source.len);
    try memcpyDtoH(DestType, h_tgt, d_source);
    return h_tgt;
}

pub fn readResult(comptime DestType: type, d_source: *const DestType) !DestType {
    var h_res: [1]DestType = undefined;
    try check(cu.cuMemcpyDtoH(
        @ptrCast(*c_void, &h_res),
        @ptrToInt(d_source),
        @sizeOf(DestType),
    ));
    return h_res[0];
}

pub fn memcpyHtoD(comptime DestType: type, d_target: []DestType, h_source: []const DestType) !void {
    std.debug.assert(h_source.len == d_target.len);
    try check(cu.cuMemcpyHtoD(
        @ptrToInt(d_target.ptr),
        @ptrCast(*const c_void, h_source.ptr),
        h_source.len * @sizeOf(DestType),
    ));
}
pub fn memcpyDtoH(comptime DestType: type, h_target: []DestType, d_source: []const DestType) !void {
    std.debug.assert(d_source.len == h_target.len);
    try check(cu.cuMemcpyDtoH(
        @ptrCast(*c_void, h_target.ptr),
        @ptrToInt(d_source.ptr),
        d_source.len * @sizeOf(DestType),
    ));
}

pub fn push(value: anytype) !*@TypeOf(value) {
    const DestType = @TypeOf(value);
    var d_ptr = try alloc(DestType, 1);
    try check(cu.cuMemcpyHtoD(
        @ptrToInt(d_ptr.ptr),
        @ptrCast(*const c_void, &value),
        @sizeOf(DestType),
    ));
    return @ptrCast(*DestType, d_ptr.ptr);
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
    try testing.expect(cu.CUDA_VERSION > 11000);
    try testing.expectEqual(cu.cuInit(0), cu.CUDA_SUCCESS);
}

var _device = [_]cu.CUdevice{-1} ** 8;
fn getDevice(device: u3) !cu.CUdevice {
    var cu_dev = &_device[device];
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
    var cu_ctx = &_ctx[device];
    if (cu_ctx.* == null) {
        try check(cu.cuCtxCreate(cu_ctx, 0, cu_dev));
    }
    return cu_ctx.*;
}

var _default_module: cu.CUmodule = null;

fn defaultModule() cu.CUmodule {
    if (_default_module != null) return _default_module;
    const file = cudaz_options.kernel_ptx_path;

    if (kernel_ptx_content.len == 0) {
        log.warn("Loading Cuda module from local file {s}", .{file});
        // Note: I tried to make this a path relative to the executable but failed because
        // the main executable and the test executable are in different folder
        // but refer to the same .ptx file.
        check(cu.cuModuleLoad(&_default_module, file)) catch |err| {
            std.debug.panic("Couldn't load cuda module: {s}: {}", .{ file, err });
        };
    } else {
        log.info("Loading Cuda module from embedded file.", .{});
        check(cu.cuModuleLoadData(&_default_module, kernel_ptx_content)) catch |err| {
            std.debug.panic("Couldn't load embedded cuda module. Originally file was at {s}: {}", .{ file, err });
        };
    }
    if (_default_module == null) {
        std.debug.panic("Couldn't find module.", .{});
    }
    return _default_module;
}

/// Create a function with the correct signature for a cuda Kernel.
/// The kernel must come from the default .cu file
pub inline fn Function(comptime name: [:0]const u8) type {
    return FnStruct(name, @field(cu, name));
}

pub fn FnStruct(comptime name: []const u8, comptime func: anytype) type {
    return struct {
        const Self = @This();
        const CpuFn = func;
        pub const Args = meta.ArgsTuple(@TypeOf(Self.CpuFn));

        f: cu.CUfunction,

        pub fn init() !Self {
            var f: cu.CUfunction = undefined;
            var code = cu.cuModuleGetFunction(&f, defaultModule(), @ptrCast([*c]const u8, name));
            if (code != cu.CUDA_SUCCESS) log.err("Couldn't load function {s}", .{name});
            try check(code);
            var res = Self{ .f = f };
            log.info("Loaded function {}@{}", .{ res, f });
            return res;
        }

        // TODO: deinit -> CUDestroy

        // Note: I'm not fond of having the primary launch be on the Function object,
        // but it works best with Zig type inference
        pub inline fn launch(self: *const Self, stream: *const Stream, grid: Grid, args: Args) !void {
            if (args.len != @typeInfo(Args).Struct.fields.len) {
                @compileError("Expected more arguments");
            }
            try self.launchWithSharedMem(stream, grid, 0, args);
        }

        pub fn launchWithSharedMem(self: *const Self, stream: *const Stream, grid: Grid, shared_mem: usize, args: Args) !void {
            // TODO: this seems error prone, could we make the type of the shared buffer
            // part of the function signature ?
            try stream.launchWithSharedMem(self.f, grid, shared_mem, args);
        }

        pub fn debugCpuCall(grid: Grid, point: Grid, args: Args) void {
            cu.threadIdx = point.threads.dim3();
            cu.blockDim = grid.threads.dim3();
            cu.blockIdx = point.blocks.dim3();
            cu.gridDim = grid.blocks.dim3();
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
    log.warn("My kernel: {s}", .{@TypeOf(cu.rgba_to_greyscale)});
}

test "we use only one context per GPU" {
    var default_ctx: cu.CUcontext = undefined;
    try check(cu.cuCtxGetCurrent(&default_ctx));
    std.log.warn("default_ctx: {any}", .{std.mem.asBytes(&default_ctx).*});

    var stream = try Stream.init(0);
    var stream_ctx: cu.CUcontext = undefined;
    try check(cu.cuStreamGetCtx(stream._stream, &stream_ctx));
    std.log.warn("stream_ctx: {any}", .{std.mem.asBytes(&stream_ctx).*});
    // try testing.expectEqual(default_ctx, stream_ctx);

    // Create a new stream
    var stream2 = try Stream.init(0);
    var stream2_ctx: cu.CUcontext = undefined;
    try check(cu.cuStreamGetCtx(stream2._stream, &stream2_ctx));
    std.log.warn("stream2_ctx: {any}", .{std.mem.asBytes(&stream2_ctx).*});
    try testing.expectEqual(stream_ctx, stream2_ctx);
}

test "rgba_to_greyscale" {
    var stream = try Stream.init(0);
    defer stream.deinit();
    log.warn("cuda: {}", .{stream});
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
    stream.synchronize();
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
    stream.synchronize();
    log.warn("stream: {}, fn: {}", .{ stream, rgba_to_greyscale.f });
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

    log.warn("stream: {}, fn: {}", .{ stream, rgba_to_greyscale.f });
    var timer = GpuTimer.start(&stream);
    try rgba_to_greyscale.launch(
        &stream,
        .{ .blocks = Dim3.init(numCols, numRows, 1) },
        .{ d_rgbaImage.ptr, d_greyImage.ptr, numRows, numCols },
    );
    timer.stop();
    log.warn("rgba_to_greyscale took: {}", .{timer.elapsed()});
    try testing.expect(timer.elapsed() > 0);
}

pub fn Kernels(comptime module: type) type {
    // @compileLog(@typeName(module));
    const decls = @typeInfo(module).Struct.decls;
    var kernels: [decls.len]TypeInfo.StructField = undefined;
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
    return @Type(TypeInfo{
        .Struct = TypeInfo.Struct{
            .is_tuple = false,
            .layout = .Auto,
            .decls = &[_]TypeInfo.Declaration{},
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
