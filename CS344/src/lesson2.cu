// Disable C++ name mangling from nvcc
#ifdef __cplusplus
extern "C" {
#endif

__global__ void increment_naive(int *g, int array_size) {
  // which thread is this?
  int i = blockIdx.x * blockDim.x + threadIdx.x;

  // each thread to increment consecutive elements, wrapping at array_size
  i = i % array_size;
  g[i] = g[i] + 1;
}

__global__ void increment_atomic(int *g, int array_size) {
  // which thread is this?
  int i = blockIdx.x * blockDim.x + threadIdx.x;

  // each thread to increment consecutive elements, wrapping at array_size
  i = i % array_size;
  atomicAdd(& g[i], 1);

}

#ifdef __cplusplus
}
#endif
