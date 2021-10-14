#include <math.h>
#define MIN(x, y) (((x) < (y)) ? (x) : (y))
#define MAX(x, y) (((x) > (y)) ? (x) : (y))
#define CLAMP(x, n) MIN(n - 1, MAX(0, x))
#define uchar unsigned char
#ifdef __cplusplus
// This is only seen by nvcc, not by Zig

// You must use this macro to declare shared buffers
#define SHARED(NAME, TYPE) extern __shared__ TYPE NAME[];

extern "C" {
#endif

__device__ uchar readChannel(uchar3* d_pixels, int offset, uint channel) {
    uchar* channel_ptr = (uchar*)d_pixels;
    return channel_ptr[3 * offset + channel];
}
__device__ void writeChannel(uchar3* d_pixels, int offset, uint channel, uchar val) {
    uchar* channel_ptr = (uchar*)d_pixels;
    channel_ptr[3 * offset + channel] = val;
}

__global__ void naive_normalized_cross_correlation(
    float *d_response, uchar3 *d_original,
    uchar3 *d_template,
    int num_pixels_y,
    int num_pixels_x,
    int template_half_height,
    int template_height,
    int template_half_width,
    int template_width,
    int template_size
) {
  int ny = num_pixels_y;
  int nx = num_pixels_x;
  int knx = template_width;
  int2 image_index_2d = {
    (int)((blockIdx.x * blockDim.x) + threadIdx.x),
    (int)((blockIdx.y * blockDim.y) + threadIdx.y)
  };
  int channel = threadIdx.z;
  int image_index_1d = (nx * image_index_2d.y) + image_index_2d.x;

  if (image_index_2d.x < nx && image_index_2d.y < ny) {
    // compute image mean
    float image_sum = 0.0f;
    float template_sum = 0.0f;

    for (int y = -template_half_height; y <= template_half_height; y++) {
      for (int x = -template_half_width; x <= template_half_width; x++) {
        int2 image_offset_index_2d = {
            CLAMP(image_index_2d.x + x, nx),
            CLAMP(image_index_2d.y + y, ny)
        };
        int image_offset_index_1d = (nx * image_offset_index_2d.y) + image_offset_index_2d.x;

        uchar original = readChannel(d_original, image_offset_index_1d, channel);
        image_sum += (float)original;
      }
    }

    float template_mean = template_sum / (float)template_size;
    float image_mean = image_sum / (float)template_size;

    // compute sums
    float sum_of_image_template_diff_products = 0.0f;
    float sum_of_squared_image_diffs = 0.0f;
    float sum_of_squared_template_diffs = 0.0f;

    for (int y = -template_half_height; y <= template_half_height; y++) {
      for (int x = -template_half_width; x <= template_half_width; x++) {
        int2 image_offset_index_2d = {
            CLAMP(image_index_2d.x + x, nx),
            CLAMP(image_index_2d.y + y, ny)
        };
        int image_offset_index_1d =
            (nx * image_offset_index_2d.y) +
            image_offset_index_2d.x;

        unsigned char image_offset_value = readChannel(d_original, image_offset_index_1d, channel);
        float image_diff = (float)image_offset_value - image_mean;

        int2 template_index_2d = {x + template_half_width, y + template_half_height};
        int template_index_1d = (knx * template_index_2d.y) + template_index_2d.x;

        unsigned char template_value = readChannel(d_template, template_index_1d, channel);
        float template_diff = template_value - template_mean;

        float image_template_diff_product = image_offset_value * template_diff;
        float squared_image_diff = image_diff * image_diff;
        float squared_template_diff = template_diff * template_diff;

        sum_of_image_template_diff_products += image_template_diff_product;
        sum_of_squared_image_diffs += squared_image_diff;
        sum_of_squared_template_diffs += squared_template_diff;
      }
    }

    //
    // compute final result
    //
    float result_value = 0.0f;

    if (sum_of_squared_image_diffs != 0 && sum_of_squared_template_diffs != 0) {
      result_value =
          sum_of_image_template_diff_products /
          sqrt(sum_of_squared_image_diffs * sum_of_squared_template_diffs);
    }

    d_response[image_index_1d] = result_value;
  }
}


__global__
void blellochCdf(float* d_cdf, uint* d_bins, int n) {
    // We need synchronization across all threads so only one block
    if (blockDim.x == 0) return;
    int tid = threadIdx.x;
    if (tid >= n) return;

    // Reduce
    uint step = 1;
    for (; step < n; step *= 2) {
        if (tid >= step && (n - 1 - tid) % (step * 2) == 0) {
            d_bins[tid] += d_bins[tid - step];
        }
        __syncthreads();
    }

    uint total = d_bins[n - 1];
    if (tid == n - 1) d_bins[tid] = 0;

    // Downsweep
    for (step /= 2; step > 0; step /= 2) {
        if (tid >= step && (n - 1 - tid) % (step * 2) == 0) {
            uint left = d_bins[tid - step];
            uint right = d_bins[tid];
            d_bins[tid] = left + right;
            d_bins[tid - step] = right;
        }
        __syncthreads();
    }

    // Normalization
    float normalized = 1.0 / total;
    d_cdf[tid] = d_bins[tid] * normalized;
}

__global__
void reduce_min(const float* d_in, float* d_out, int num_pixels)
{
    // sdata is allocated in the kernel call: 3rd arg to <<<b, t, shmem>>>
    SHARED(sdata, float);
    int myId = threadIdx.x + blockDim.x * blockIdx.x;
    if (myId >= num_pixels) {
      return;
    }
    int tid  = threadIdx.x;

    // load shared mem from global mem
    sdata[tid] = d_in[myId];
    __syncthreads();            // make sure entire block is loaded!

    // do reduction in shared mem
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] = MIN(sdata[tid], sdata[tid + s]);
        }
        __syncthreads();        // make sure all adds at one stage are done!
    }

    // only thread 0 writes result for this block back to global mem
    if (tid == 0) {
        d_out[blockIdx.x] = sdata[0];
    }
}

__global__
void range(uint* d_out, uint n) {
    int myId = threadIdx.x + blockDim.x * blockIdx.x;
    if (myId >= n) return;
    d_out[myId] = myId;
}

__global__ void remove_redness(
    const unsigned int *d_coordinates,
    uchar3 *d_rgb,
    int num_coordinates,
    int num_pixels_y,
    int num_pixels_x,
    int template_half_height,
    int template_half_width) {
  int ny = num_pixels_y;
  int nx = num_pixels_x;
  int global_index_1d = (blockIdx.x * blockDim.x) + threadIdx.x;

  int imgSize = num_pixels_x * num_pixels_y;

  if (global_index_1d < num_coordinates) {
    unsigned int image_index_1d = d_coordinates[imgSize - global_index_1d - 1];
    int2 image_index_2d = {
        (int)image_index_1d % num_pixels_x,
        (int)image_index_1d / num_pixels_x
    };

    for (int y = image_index_2d.y - template_half_height;
         y <= image_index_2d.y + template_half_height; y++) {
      for (int x = image_index_2d.x - template_half_width;
           x <= image_index_2d.x + template_half_width; x++) {
        int2 image_offset_index_2d = {CLAMP(x, nx), CLAMP(y, ny)};
        int image_offset_index_1d = (nx * image_offset_index_2d.y) + image_offset_index_2d.x;

        uchar3 px = d_rgb[image_offset_index_1d];
        uchar3 new_px = {
            (uchar)((px.y + px.z) << 1),
            px.y,
            px.z
        };

        d_rgb[image_offset_index_1d] = new_px;
      }
    }
  }
}

#ifdef __cplusplus
}
#endif
