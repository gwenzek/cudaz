// When compiling .cu files `nvcc` will automatically add some builtin types
// and thread variables.
// This headers aims to allow zig to also be able to compile .cu files.
// The goal isn't to call the zig compiled cuda, but to have proper type
// checking on the kernel function call.

#include "builtin_types.h"

dim3 blockIdx;
dim3 gridIdx;
