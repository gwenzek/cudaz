#define MIN(X, Y) (((X) < (Y)) ? (X) : (Y))
#define MAX(X, Y) (((X) > (Y)) ? (X) : (Y))
#define CLAMP(x, n) MIN(n - 1, MAX(0, x))
#define uchar unsigned char

#ifdef __cplusplus
// This is only seen by nvcc, not by Zig

// You must use this macro to declare shared buffers
#define SHARED(NAME, TYPE) extern __shared__ TYPE NAME[];

#endif
