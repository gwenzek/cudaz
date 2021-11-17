#include <stdbool.h>

#define uchar unsigned char
#define FLAT_ID_2D(threadIdx, blockIdx, blockDim) threadIdx + blockDim * blockIdx;
#define ID_X threadIdx.x + blockDim.x * blockIdx.x
#define ID_Y threadIdx.y + blockDim.y * blockIdx.y
#define ID_Z threadIdx.z + blockDim.z * blockIdx.z
#define MIN(X, Y) (((X) < (Y)) ? (X) : (Y))
#define MAX(X, Y) (((X) > (Y)) ? (X) : (Y))
#define CLAMP(x, n) MIN(n - 1, MAX(0, x))

// Get the last valid thread id in this block
// The last block often contains thread id that goes above the input size
#define LAST_TID(n) (blockIdx.x == gridDim.x - 1) ? (n-1) % blockDim.x : blockDim.x - 1;

#ifdef __cplusplus
// This is only seen by nvcc, not by Zig

// You must use this macro to declare shared buffers
#define SHARED(NAME, TYPE) extern __shared__ TYPE NAME[];

#endif
