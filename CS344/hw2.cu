#ifdef __cplusplus
extern "C" {
#endif

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
