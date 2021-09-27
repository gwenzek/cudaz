const std = @import("std");
const log = std.log;
const cuda_module = @import("cuda");
const Cuda = cuda_module.Cuda;
const cu = cuda_module.cu;
const NUM_THREADS = 1000000;
const ARRAY_SIZE = 100;
const BLOCK_WIDTH = 1000;

pub fn main() anyerror!void {
    var general_purpose_allocator = std.heap.GeneralPurposeAllocator(.{}){};
    const gpa = &general_purpose_allocator.allocator;
    const args = try std.process.argsAlloc(gpa);
    defer std.process.argsFree(gpa, args);

    var cuda = try Cuda.init(0);
    defer cuda.deinit();

    var timer = cuda_module.GpuTimer.init(&cuda);
    log.info("{} total threads in {} blocks writing into {} array elements", .{ NUM_THREADS, NUM_THREADS / BLOCK_WIDTH, ARRAY_SIZE });

    // declare and allocate host memory
    var h_array = [_]i32{0} ** ARRAY_SIZE;

    // declare, allocate, and zero out GPU memory
    var d_array = cuda.malloc(i32, ARRAY_SIZE);
    cuda.memset(i32, d_array, 0);

    // launch the kernel - comment out one of these
    timer.start();

    // Instructions: This program is needed for the next quiz
    // uncomment increment_naive to measure speed and accuracy
    // of non-atomic increments or uncomment increment_atomic to
    // measure speed and accuracy of  atomic icrements
    // increment_naive<<<NUM_THREADS/BLOCK_WIDTH, BLOCK_WIDTH>>>(d_array);
    const increment_atomic = try cuda_module.KernelSignature("./cudaz/kernel.ptx", "increment_atomic").init(&cuda);
    // const increment_atomic = try cuda.kernel("./cudaz/kernel.ptx", "increment_atomic");
    log.info("loaded kernel", .{});
    try increment_atomic.launch(
        .{ .x = NUM_THREADS / BLOCK_WIDTH, .y = BLOCK_WIDTH },
        .{},
        .{ .@"0" = d_array.ptr },
    );
    timer.stop();

    log.info("ran kernel", .{});
    // copy back the array of sums from GPU and print
    cuda.memcpyDtoH(i32, &h_array, d_array);
    log.info("array: {any}", .{h_array});
    log.info("time elapsed: {} ms", .{timer.elapsed()});

    // free GPU memory allocation and exit
    // cudaFree(d_array);
}
