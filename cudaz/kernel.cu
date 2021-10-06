#ifdef __cplusplus
// This is only seen by nvcc, not by Zig

// You must use this macro to declare shared buffers
#define SHARED(NAME, TYPE) extern __shared__ TYPE NAME[];

extern "C" {
#endif

__global__ void rgba_to_greyscale(const uchar3 *const rgbaImage,
                                  unsigned char *const greyImage, int numRows,
                                  int numCols) {
  // TODO
  // Fill in the kernel to convert from color to greyscale
  // the mapping from components of a uchar4 to RGBA is:
  // .x -> R ; .y -> G ; .z -> B ; .w -> A
  //
  // The output (greyImage) at each pixel should be the result of
  // applying the formula: output = .299f * R + .587f * G + .114f * B;
  // Note: We will be ignoring the alpha channel for this conversion

  // First create a mapping from the 2D block and grid locations
  // to an absolute 2D location in the image, then use that to
  // calculate a 1D offset
  uchar3 px = rgbaImage[blockIdx.x * numCols + blockIdx.y];
  float R = px.x;
  float G = px.y;
  float B = px.z;
  float output = (0.299f * R + 0.587f * G + 0.114f * B);
  greyImage[blockIdx.x * numCols + blockIdx.y] = output;
}


__global__ void increment_naive(int *g, int array_size)
{
  // which thread is this?
  int i = blockIdx.x * blockDim.x + threadIdx.x;

  // each thread to increment consecutive elements, wrapping at array_size
  i = i % array_size;
  g[i] = g[i] + 1;
}

__global__ void increment_atomic(int *g, int array_size)
{
  // which thread is this?
  int i = blockIdx.x * blockDim.x + threadIdx.x;

  // each thread to increment consecutive elements, wrapping at array_size
  i = i % array_size;
  atomicAdd(& g[i], 1);

}


// HW 2
__global__ void gaussian_blur(const unsigned char *const inputChannel,
                              unsigned char *const outputChannel, uint numRows,
                              uint numCols, const float *const filter,
                              const int filterWidth) {

    const uint2 thread_2D_pos = {blockIdx.x * blockDim.x + threadIdx.x,
                                       blockIdx.y * blockDim.y + threadIdx.y};
    const uint thread_1D_pos = thread_2D_pos.y * numCols + thread_2D_pos.x;

    if (thread_2D_pos.x >= numCols || thread_2D_pos.y >= numRows)
      return;
    outputChannel[thread_1D_pos] = inputChannel[thread_1D_pos];
    // NOTE: If a thread's absolute position 2D position is within the image, but
    // some of its neighbors are outside the image, then you will need to be extra
    // careful. Instead of trying to read such a neighbor value from GPU memory
    // (which won't work because the value is out of bounds), you should
    // explicitly clamp the neighbor values you read to be within the bounds of
    // the image. If this is not clear to you, then please refer to sequential
    // reference solution for the exact clamping semantics you should follow.

    int halfWidth = filterWidth / 2;
    float pixel = 0;
    for (int r = -halfWidth; r <= halfWidth; ++r) {
        int n_y = thread_2D_pos.y + r;
        if (n_y >= numRows) n_y = numRows - 1;
        if (n_y < 0) n_y = 0;
        for (int c = -halfWidth; c <= halfWidth; ++c) {
            int n_x = thread_2D_pos.x + c;
            if (n_x >= numCols) n_x = numCols - 1;
            if (n_x < 0) n_x = 0;
            pixel += filter[(r + halfWidth) * filterWidth + c + halfWidth] * inputChannel[n_y * numCols + n_x];
        }
    }

    outputChannel[thread_1D_pos] = pixel;
}

// This kernel takes in an image represented as a uchar4 and splits
// it into three images consisting of only one color channel each
__global__ void separateChannels(const uchar3 *const inputImage,
                                 int numRows, int numCols,
                                 unsigned char *const redChannel,
                                 unsigned char *const greenChannel,
                                 unsigned char *const blueChannel) {
    const uint2 thread_2D_pos = {blockIdx.x * blockDim.x + threadIdx.x,
                                       blockIdx.y * blockDim.y + threadIdx.y};
    const uint thread_1D_pos = thread_2D_pos.y * numCols + thread_2D_pos.x;

    if (thread_2D_pos.x >= numCols || thread_2D_pos.y >= numRows)
      return;

    uchar3 px = inputImage[thread_1D_pos];
    redChannel[thread_1D_pos] = px.x;
    greenChannel[thread_1D_pos] = px.y;
    blueChannel[thread_1D_pos] = px.z;
}

// This kernel takes in three color channels and recombines them
// into one image.  The alpha channel is set to 255 to represent
// that this image has no transparency.
__global__ void recombineChannels(const unsigned char *const redChannel,
                                  const unsigned char *const greenChannel,
                                  const unsigned char *const blueChannel,
                                  uchar3 *const outputImageRGBA, int numRows,
                                  int numCols) {
  const uint2 thread_2D_pos = {blockIdx.x * blockDim.x + threadIdx.x,
                                       blockIdx.y * blockDim.y + threadIdx.y};

  const uint thread_1D_pos = thread_2D_pos.y * numCols + thread_2D_pos.x;

  // make sure we don't try and access memory outside the image
  // by having any threads mapped there return early
  if (thread_2D_pos.x >= numCols || thread_2D_pos.y >= numRows)
    return;

  unsigned char red = redChannel[thread_1D_pos];
  unsigned char green = greenChannel[thread_1D_pos];
  unsigned char blue = blueChannel[thread_1D_pos];

  uchar3 outputPixel = {red, green, blue};
  outputImageRGBA[thread_1D_pos] = outputPixel ;
}

#ifdef __cplusplus
}
#endif
