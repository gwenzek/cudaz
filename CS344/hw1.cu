#ifdef __cplusplus
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
  uint myId = blockIdx.x * blockDim.x + threadIdx.x;
  if (myId >= numRows * numCols) return;
  uchar3 px = rgbaImage[myId];
  float R = px.x;
  float G = px.y;
  float B = px.z;
  float output = (0.299f * R + 0.587f * G + 0.114f * B);
  greyImage[myId] = (unsigned char)output;
}

#ifdef __cplusplus
}
#endif
