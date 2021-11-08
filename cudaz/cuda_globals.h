// When compiling .cu files `nvcc` will automatically add some builtin types
// and thread variables.
// This headers aims to allow zig to also be able to compile .cu files.
// The goal isn't to call the zig compiled cuda, but to have proper type
// checking on the kernel function call.

#include "builtin_types.h"
#define __global__

// In cuda shared memory buffers are declared as extern array
// But this confuses Zig, because it can't find the extern definition.
// Declare them as pointers so that Zig doesn't try to find the size.
#define SHARED(NAME, TYPE) extern TYPE *NAME;
#define extern

dim3 gridDim;
dim3 blockIdx;
dim3 blockDim;
dim3 threadIdx;

int atomicAdd(int* a, int b) {a += b;}
int atomicMin(int* a, int b) {
  if (b < *a) *a = b;
}
int atomicMax(int* a, int b) {
  if (b > *a) *a = b;
}

void __syncthreads() {}
