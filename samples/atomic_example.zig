const std = @import("std");
const log = std.log;
const cuda_module = @import("cuda");
const Cuda = cuda_module.Cuda;
const cu = cuda_module.cu;
const Dim3 = cuda_module.Dim3;

const ARRAY_SIZE = 100;

pub fn main() anyerror!void {
    var general_purpose_allocator = std.heap.GeneralPurposeAllocator(.{}){};
    const alloc = &general_purpose_allocator.allocator;
    const args = try std.process.argsAlloc(alloc);
    defer std.process.argsFree(alloc, args);

    var cuda = try Cuda.init(0);
    defer cuda.deinit();

    // Instructions: This program is needed for the next quiz
    // uncomment increment_naive to measure speed and accuracy
    // of non-atomic increments or uncomment increment_atomic to
    // measure speed and accuracy of  atomic icrements
    time_kernel(alloc, &cuda, "increment_naive", 1e6, 1000, 1e6);
    time_kernel(alloc, &cuda, "increment_atomic", 1e6, 1000, 1e6);
    time_kernel(alloc, &cuda, "increment_naive", 1e6, 1000, 100);
    time_kernel(alloc, &cuda, "increment_atomic", 1e6, 1000, 100);
    time_kernel(alloc, &cuda, "increment_atomic", 1e7, 1000, 100);
}

fn time_kernel(
    alloc: *std.mem.Allocator,
    cuda: *Cuda,
    comptime kernel_name: [:0]const u8,
    num_threads: u32,
    block_width: u32,
    comptime array_size: u32,
) void {
    _time_kernel(alloc, cuda, kernel_name, num_threads, block_width, array_size) catch |err| {
        log.err("Failed to run kernel {s}: {}", .{ kernel_name, err });
    };
}

fn _time_kernel(
    alloc: *std.mem.Allocator,
    cuda: *Cuda,
    comptime kernel_name: [:0]const u8,
    num_threads: u32,
    block_width: u32,
    comptime array_size: u32,
) !void {
    const kernel = try cuda_module.KernelSignature("./cudaz/kernel.ptx", kernel_name).init(cuda);
    var timer = cuda_module.GpuTimer.init(cuda);

    // declare and allocate memory
    const h_array = try alloc.alloc(i32, array_size);
    defer alloc.free(h_array);
    var d_array = cuda.alloc(i32, array_size);
    defer cuda.free(d_array);
    cuda.memset(i32, d_array, 0);

    log.info("*** Will benchmark kernel {s} ***", .{kernel_name});
    log.info("{} total threads in {} blocks writing into {} array elements", .{ num_threads, num_threads / block_width, array_size });
    timer.start();
    try kernel.launch(
        .{ .x = num_threads / block_width },
        .{ .x = block_width },
        .{ .@"0" = d_array.ptr, .@"1" = array_size },
    );
    timer.stop();

    // copy back the array of sums from GPU and print
    cuda.memcpyDtoH(i32, h_array, d_array);
    log.info("array: {any}", .{h_array[0..std.math.min(h_array.len, 100)]});
    log.info("time elapsed: {d:.4} ms", .{timer.elapsed()});
}
