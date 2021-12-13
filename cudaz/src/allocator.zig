const std = @import("std");
const cuda = @import("cuda.zig");

const assert = std.debug.assert;
const mem = std.mem;
const Allocator = std.mem.Allocator;
const log = std.log.scoped(.Cuda);

/// This allocator takes the cuda allocator, wraps it, and provides an interface
/// where you can allocate without freeing, and then free it all together.
/// It also need a host allocator to store the node metadata.
/// It is an adapation from std lib arena allocator, where create_node uses the host allocator
/// Technically it works, but don't have all the convenient methods
/// from the std.mem.Allocator API. See below for more explanation
pub const ArenaAllocator = struct {
    nodes: std.ArrayList([]u8),
    end_index: usize = 0,

    const Node = []u8;

    pub fn init(host_allocator: Allocator) ArenaAllocator {
        return .{ .nodes = std.ArrayList([]u8).init(host_allocator) };
    }

    pub fn deinit(self: *ArenaAllocator) void {
        for (self.nodes.items) |node| {
            // this has to occur before the free because the free frees node
            cuda.free(node);
        }
        self.nodes.deinit();
    }

    fn createNode(self: *ArenaAllocator, prev_len: usize, minimum_size: usize) Allocator.Error!*Node {
        const big_enough_len = prev_len + minimum_size;
        const len = big_enough_len + big_enough_len / 2;
        const buf = cuda.alloc(u8, len) catch |err| switch (err) {
            error.OutOfMemory => cuda.alloc(u8, minimum_size) catch return error.OutOfMemory,
            else => unreachable,
        };
        const buf_node = try self.nodes.addOne();
        buf_node.* = buf;
        self.end_index = 0;
        return buf_node;
    }

    pub fn alloc(self: *ArenaAllocator, comptime T: type, n_items: usize) Allocator.Error![]T {
        const n = @sizeOf(T) * n_items;
        var num_nodes = self.nodes.items.len;
        var cur_node = if (num_nodes > 0) &self.nodes.items[num_nodes - 1] else try self.createNode(0, n);
        // this while loop should only execute twice
        var counter: u8 = 0;
        while (true) {
            const cur_buf = cur_node.*;
            const new_end_index = self.end_index + n;
            if (new_end_index <= cur_buf.len) {
                const result = cur_buf[self.end_index..new_end_index];
                self.end_index = new_end_index;
                return result;
            }

            // Allocate more memory
            cur_node = try self.createNode(cur_buf.len, n);
            counter += 1;
            std.debug.assert(counter < 2);
        }
    }

    pub fn resizeFn(allocator: Allocator, buf: []u8, buf_align: u29, new_len: usize, len_align: u29, ret_addr: usize) Allocator.Error!usize {
        _ = buf_align;
        _ = len_align;
        _ = ret_addr;
        const self = @fieldParentPtr(ArenaAllocator, "allocator", allocator);

        var num_nodes = self.nodes.items.len;
        var cur_node = if (num_nodes > 0) &self.nodes.items[num_nodes - 1] else return error.OutOfMemory;
        const cur_buf = cur_node.*[@sizeOf(Node)..];
        if (@ptrToInt(cur_buf.ptr) + self.end_index != @ptrToInt(buf.ptr) + buf.len) {
            if (new_len > buf.len)
                return error.OutOfMemory;
            return new_len;
        }

        if (buf.len >= new_len) {
            self.end_index -= buf.len - new_len;
            return new_len;
        } else if (cur_buf.len - self.end_index >= new_len - buf.len) {
            self.end_index += new_len - buf.len;
            return new_len;
        } else {
            return error.OutOfMemory;
        }
    }
};

// *** Tentative to create a standard allocator API using cuAlloc ***
// This doesn't work because Allocator.zig from std will call @memset(undefined)
// on the returned pointer which will segfault, because we're returning a device pointer.
// https://github.com/ziglang/zig/issues/4298 want to make the @memset optional
// But does it make sense to have an allocator that return GPU memory ?
// Most function that want an allocator want to read/write the returned data.
// I think we should only have this in GPU code.

// TODO: we could create an allocator that map the memory to the host
// this will likely make read/write much slower though
// https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__VA.html#group__CUDA__VA
// fn cudaAllocMappedFn(allocator: Allocator, n: usize, ptr_align: u29, len_align: u29, ra: usize) Allocator.Error![]u8 {
// }
//
pub const cuda_allocator = &cuda_allocator_state;
var cuda_allocator_state = Allocator{
    .allocFn = cudaAllocFn,
    .resizeFn = cudaResizeFn,
};

fn cudaAllocFn(allocator: Allocator, n: usize, ptr_align: u29, len_align: u29, ra: usize) Allocator.Error![]u8 {
    _ = allocator;
    _ = ra;
    _ = ptr_align;
    _ = len_align;

    const x = cuda.alloc(u8, n) catch |err| switch (err) {
        error.OutOfMemory => return error.OutOfMemory,
        else => {
            log.err("Cuda error while allocating memory: {}", .{err});
            return error.OutOfMemory;
        },
    };
    @memset(x.ptr, undefined, x.len);
    log.warn("allocated {}b at {*}", .{ x.len, x.ptr });
    return x;
}

fn cudaResizeFn(allocator: Allocator, buf: []u8, buf_align: u29, new_len: usize, len_align: u29, ra: usize) Allocator.Error!usize {
    _ = allocator;
    _ = ra;
    _ = buf_align;
    if (new_len == 0) {
        cuda.free(buf);
        return new_len;
    }
    if (new_len <= buf.len) {
        return std.mem.alignAllocLen(buf.len, new_len, len_align);
    }
    return error.OutOfMemory;
}

test "cuda_allocator" {
    _ = try cuda.Stream.init(0);
    const x = cuda.alloc(u8, 800) catch |err| switch (err) {
        error.OutOfMemory => return error.OutOfMemory,
        else => {
            log.err("Cuda error while allocating memory: {}", .{err});
            return error.OutOfMemory;
        },
    };
    log.warn("allocated {} bytes at {*}", .{ x.len, x.ptr });
    @memset(x.ptr, undefined, x.len);
    // try std.heap.testAllocator(cuda_allocator);
}

test "nice error when OOM" {
    var stream = try cuda.Stream.init(0);
    defer stream.deinit();
    var arena = ArenaAllocator.init(std.testing.allocator);

    var last_err: anyerror = blk: {
        while (true) {
            _ = arena.alloc(u8, 1024 * 1024) catch |err| break :blk err;
        }
    };
    try std.testing.expectEqual(last_err, error.OutOfMemory);
    log.warn("Cleaning up cuda memory and reallocating", .{});
    arena.deinit();
    var buff = try cuda.alloc(u8, 1024 * 1024);
    defer cuda.free(buff);
}
