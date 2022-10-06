#ifdef __cplusplus
// This is only seen by nvcc, not by Zig

extern "C" {
#endif

__global__ void rgbToGreyscale(
    const uchar3 *const rgba_image,
    unsigned char *const grey_image,
    int num_pixels
) {
    // Fill in the kernel to convert from color to greyscale
    // the mapping from components of a uchar3 to RGB is:
    // .x -> R ; .y -> G ; .z -> B
    // The output (grey_image) at each pixel should be the result of
    // applying the formula: output = .299f * R + .587f * G + .114f * B;
    // Calculate a 1D offset for this thread.
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i > num_pixels) return;
    uchar3 px = rgba_image[i];
    float R = px.x;
    float G = px.y;
    float B = px.z;
    float output = (0.299f * R + 0.587f * G + 0.114f * B);
    grey_image[i] = output;
}


__global__ void incrementNaive(int *g, int array_size)
{
  // id of current thread
  int i = blockIdx.x * blockDim.x + threadIdx.x;

  // each thread to increment consecutive elements, wrapping at array_size
  i = i % array_size;
  g[i] = g[i] + 1;
}

__global__ void incrementAtomic(int *g, int array_size)
{
  // id of current thread
  int i = blockIdx.x * blockDim.x + threadIdx.x;

  // each thread to increment consecutive elements, wrapping at array_size
  i = i % array_size;
  atomicAdd(& g[i], 1);
}


#ifdef __cplusplus
}
#endif
