// When compiling .cu files `nvcc` will automatically add some builtin types
// and thread variables.
// This headers aims to allow zig to also be able to compile .cu files.
// The goal isn't to call the zig compiled cuda, but to have proper type
// checking on the kernel function call.

#include "builtin_types.h"
// #define __global__ ""

dim3 blockDim;
dim3 blockIdx;

dim3 threadDim;
dim3 threadIdx;

int atomicAdd(int* a, int b) {a += b;}
