# Cudaz

## Overview

The main motivation for this project 
was to complete the assignment of [Intro to Parallel Programming](https://classroom.udacity.com/courses/cs344)
using as little C++ as possible.

The class is meant to use Cuda. 
Cuda is a superset of C++ with custom annotation to distinguish between
device (GPU) functions and host (CPU) functions.
They also have special variables for GPU thread IDs 
and special syntax to schedule a GPU function.

You're supposed to compile this Cuda code using `nvcc` NVidia proprietary compiler.
But Cuda also has a C api that you can call easily in Zig.
And you can also load device code using a the PTX "assembly" format.
This assembly can be produced by `nvptx` itself, 
allowing you to write only the GPU code in C,
compile it with `nvcc` and load it from your Zig code.
Since Zig can parse the C code for GPU,
it knows the signature of your device code and can properly call them.

The second, more experimental, way is to generate the PTX
using LLVM through Zig stage 2.
That way you can write both host and device code in Zig.

## Project structure

This repo is divided in several parts:
* `cudaz` folder contains the "library" code
* `CS344` contains code for all the lesson and homework.
    Typically code is divided in two files.
    Host code: `hw1.zig`
    Device code: `hw1.cu` or `hw1_kernel.zig`
* [lodepng](https://github.com/lvandeve/lodepng) is a dependency to read/write images.
    * Run `git submodule init; git submodule update` to fetch it.

## Using Zig to drive the GPU

A lot of the magic happens in `build.zig` and notably in `addCudaz` function.
Generally we assume one executable will only have one .ptx.
This is actually important because we need to `cImport` at the same time
`cuda.h` and your device code.
I don't think it's a huge constraint since you can include several files
in your main device code file.

The main gotchas is that the .cu code must be `C` not `C++`.
To help with this you can include [cuda_helpers.h](./cudaz/cuda_helpers.h)
that defines a few helpers like min/max.
You also need to disable name mangling by wrapping your full device code with:

```C
#ifdef __cplusplus
extern "C" {
#endif

...

#ifdef __cplusplus
}
#endif
```

The `#ifdef __cplusplus` is unfortunately needed because
the `extern "C"` will trip up the Zig C-parser.

I recommend looking at the examples to learn more about how the API work.
And also taking the full class :-)

To use block-shared memory (`__shared` keyword in Cuda)
you'll need to use the `SHARED` macro defined in [cuda_helpers.h](./cudaz/cuda_helpers.h).

The main issue with the Cuda API,
is that most operation will use the default context and default GPU. 
This make it a bit awkward if you need to write code to drive two GPUs,
because you'll need to call `cuContextPush`/`cuContextPop` every time you want 
to talk to the other GPU.
I haven't tried to fix this in the Zig wrapper which is just a wrapper
with some utility function (also my laptop has only one GPU).

## Using Zig to write device code

For this I'm using stage2 compiler.
Zig stage1 can theoretically  target the PTX platform too
but it's seems to be broken in 0.9 dev versions.
(nvptx-cuda platform is Tier4 of support which means "unsupported by Zig but LLVM has the flag, so maybe it will work")
I was able to use a [light fork](https://github.com/gwenzek/zig/pull/1) 
of Zig stage2 to generate a `.ptx` though
without having to do any crazy stuff.
More details and pointers can be found on this ["documentation issue"](https://github.com/ziglang/zig/issues/10064)

