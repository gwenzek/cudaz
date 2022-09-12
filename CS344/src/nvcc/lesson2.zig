const std = @import("std");
const log = std.log;
const cuda = @import("cudaz");
const cu = cuda.cu;

const ARRAY_SIZE = 100;

pub fn main() anyerror!void {
    var general_purpose_allocator = std.heap.GeneralPurposeAllocator(.{}){};
    const alloc = general_purpose_allocator.allocator();
    const args = try std.process.argsAlloc(alloc);
    defer std.process.argsFree(alloc, args);

    var stream = try cuda.Stream.init(0);
    defer stream.deinit();

    // Instructions: This program is needed for the next quiz
    // uncomment increment_naive to measure speed and accuracy
    // of non-atomic increments or uncomment increment_atomic to
    // measure speed and accuracy of  atomic icrements
    time_kernel(alloc, &stream, "increment_naive", 1e6, 1000, 1e6);
    time_kernel(alloc, &stream, "increment_atomic", 1e6, 1000, 1e6);
    time_kernel(alloc, &stream, "increment_naive", 1e6, 1000, 100);
    time_kernel(alloc, &stream, "increment_atomic", 1e6, 1000, 100);
    time_kernel(alloc, &stream, "increment_atomic", 1e7, 1000, 100);
}

fn time_kernel(
    alloc: std.mem.Allocator,
    stream: *cuda.Stream,
    comptime kernel_name: [:0]const u8,
    num_threads: u32,
    block_width: u32,
    comptime array_size: u32,
) void {
    _time_kernel(alloc, stream, kernel_name, num_threads, block_width, array_size) catch |err| {
        log.err("Failed to run kernel {s}: {}", .{ kernel_name, err });
    };
}

fn _time_kernel(
    alloc: std.mem.Allocator,
    stream: *cuda.Stream,
    comptime kernel_name: [:0]const u8,
    num_threads: u32,
    block_width: u32,
    comptime array_size: u32,
) !void {
    const kernel = try cuda.CudaKernel(kernel_name).init();

    // declare and allocate memory
    const h_array = try alloc.alloc(i32, array_size);
    defer alloc.free(h_array);
    var d_array = try cuda.alloc(i32, array_size);
    defer cuda.free(d_array);
    try cuda.memset(i32, d_array, 0);

    log.info("*** Will benchmark kernel {s} ***", .{kernel_name});
    log.info("{} total threads in {} blocks writing into {} array elements", .{ num_threads, num_threads / block_width, array_size });
    var timer = cuda.GpuTimer.start(stream);
    try kernel.launch(
        stream,
        cuda.Grid.init1D(num_threads, block_width),
        .{ d_array.ptr, array_size },
    );
    timer.stop();

    // copy back the array of sums from GPU and print
    try cuda.memcpyDtoH(i32, h_array, d_array);
    log.info("array: {any}", .{h_array[0..std.math.min(h_array.len, 100)]});
    log.info("time elapsed: {d:.4} ms", .{timer.elapsed()});
}
