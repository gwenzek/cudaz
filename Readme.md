# Cudaz

## Overview

The main motivation for this project 
was to complete the assignment of [Intro to Parallel Programming](https://classroom.udacity.com/courses/cs344)
using as little C++ as possible.

The project started in 2021 with Zig 0.7.0 and was in limbo
until 2025 where I upgraded it to Zig 0.15.2 leveraging all the improvement the compiler and toolchain got int the meantime.

Cuda is a superset of C++ with custom annotation to distinguish between
device (GPU) functions and host (CPU) functions.
They also have special variables for GPU thread IDs 
and special syntax to schedule a GPU function.

You're supposed to compile this Cuda code using `nvcc` NVidia proprietary compiler,
to Nvidia PTX "assembly" format.
NVCC creates complicated object containing the ptx and automotically load it
when the executable/library is loaded.

But Zig has a LLVM backend that can also output `ptx`.
With `build.zig` we can control the full build magic done by `nvcc`
and expose it in a more modular way to the user.

## Project structure

This repo is divided in several parts:
* `cuda.zig`: a wrapper around `cuda.h` the CPU part of Cuda.
* `nvptx.zig`: PTX intrinsics that help write device code in Zig.
* `build.zig`: contains code to compile the kernel into ptx, and expose it to CPU code.

* `CS344` contains code for all the lesson and homework. **Not ported yet**

