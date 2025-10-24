
First cd into matmul root, the zig build run:

```
~/Documents/cudaz$ cd matmul
~/Documents/cudaz/matmul$ zig build run -freference-trace=20
warning(matmul): cuda: CuStream(device=0, stream=cuda.struct_CUstream_st@33a92810)
debug(cuda): Launching kernel 33acd970 with grid: .{ .blocks = .{ .x = 1, .y = 1, .z = 1 }, .threads = .{ .x = 16, .y = 16, .z = 1 } }
matmul .{ .m = 3, .n = 3, .k = 2 } took 2.72960e-2s
```

For optimized mode (only changes the frontend, ptx is always compiled in release mode)

```
zig build run -freference-trace=20 -Doptimize=ReleaseFast
```

Generated ptx can be found in zig-cache:

```
ls -lhSt .zig-cache/o/*/matmul.s
```

