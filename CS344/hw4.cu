#include <math.h>
#include "cuda_helpers.h"

#ifdef __cplusplus
extern "C" {
#endif

const uint _radix_mask = (2 << 4) - 1;

__device__ uchar readChannel(const uchar3* d_pixels, int offset, uint channel) {
    uchar* channel_ptr = (uchar*)d_pixels;
    return channel_ptr[3 * offset + channel];
}

__device__ void writeChannel(uchar3* d_pixels, int offset, uint channel, uchar val) {
    uchar* channel_ptr = (uchar*)d_pixels;
    channel_ptr[3 * offset + channel] = val;
}

__global__ void sort_network(const float* d_in) {
    uint i = threadIdx.x;
    uint n = threadDim.x;
    SHARED(d_group, uint);
    d_group[i] = d_in[ID_X];
    __syncthreads();

    // Using notation from https://en.wikipedia.org/wiki/Bitonic_sorter#How_the_algorithm_works
    for (uint k = 2; k <= n; k *= 2) {
        // Corresponds to green blocks
        bool upward = (i & k) == 0;
        for (uint j = k/2; j >=0; j/=2) {
            uint l = i ^ j;
            if (l > i) {
                bool sorted = arr[i] < arr[l];
                if ((upward & !sorted) || (!upward & sorted)) {
                    float arr_i = arr[i];
                    arr[i] = arr[l];
                    arr[l] = arr_i;
                }
            }
            __syncthreads();
        }
    }
    d_in[ID_X] = d_group[i];
}

__global__ void find_radix(
    const float* d_in,
    uchar* d_out,
    uchar shift,
    int n
) {
    uint id = ID_X;
    if (id >= n) return;
    uint x = ((uint*)d_in)[id];
    x = x >> shift & _radix_mask;
    d_out[id] = (uchar)x;
}

__global__
void radix_cdf(const uchar* d_radix, uint* h, int n) {
    int tid = ID_X;
    if (tid >= n) return;
    int group = blockIdx.y;
    SHARED(d_bins, uint);
    d_bins[tid] = d_radix[tid] == (uchar)group ? 1 : 0;
    __syncthreads();
    // Reduce
    uint step = 1;
    for (; step < n; step *= 2) {
        if (tid >= step && (n - 1 - tid) % (step * 2) == 0) {
            d_bins[tid] += d_bins[tid - step];
        }
        __syncthreads();
    }

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
    d_bins_out[blockIdx.y * n + ID_X] = d_bins[tid];
}

__global__
void cdf_vertical(uint* d_cdf, int n, int step) {
    int  tid = ID_X;
    if (tid >= n) return;
    // Here thread id correspond to a grid id in radix_cdf
    int group = tid;
    SHARED(d_bins, uint);
    d_bins[tid] = d_cdf[(group + 1) * step - 1];
    // Reduce
    uint step = 1;
    for (; step < n; step *= 2) {
        if (tid >= step && (n - 1 - tid) % (step * 2) == 0) {
            d_bins[tid] += d_bins[tid - step];
        }
        __syncthreads();
    }

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
    d_cdf[group * step] += d_bins[tid];
}

__global__
void cdf_increment_after_vertical(uint* d_cdf, int n, int step) {
    uint tid = ID_X;
    uint group_start = (tid % step) * step;
    if (tid == group_start || tid >= n) return;

    d_cdf[tid] += d_cdf[group_start];
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
void reduce_min(const float* d_in, float* d_out, int num_pixels)
{
    // sdata is allocated in the kernel call: 3rd arg to <<<b, t, shmem>>>
    SHARED(sdata, float);
    int id = ID_X;
    if (id >= num_pixels) {
      return;
    }
    int tid  = threadIdx.x;

    // load shared mem from global mem
    sdata[tid] = d_in[id];
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
    int id = ID_X;
    if (id >= n) return;
    d_out[id] = id;
}

__global__ void remove_redness(
    const unsigned int *d_coordinates, const uchar3 *d_rgb, uchar3 *d_out,
    int num_coordinates,
    int num_pixels_y, int num_pixels_x,
    int template_half_height, int template_half_width
) {
  int ny = num_pixels_y;
  int nx = num_pixels_x;
  int global_index_1d = ID_X;

  int imgSize = num_pixels_x * num_pixels_y;

  if (global_index_1d < num_coordinates) {
    uint image_index_1d = d_coordinates[imgSize - global_index_1d - 1];
    uint2 image_index_2d = {
        image_index_1d % num_pixels_x,
        image_index_1d / num_pixels_x
    };

    for (int y = image_index_2d.y - template_half_height;
         y <= image_index_2d.y + template_half_height; y++) {
      for (int x = image_index_2d.x - template_half_width;
           x <= image_index_2d.x + template_half_width; x++) {
        int clamped_index = (nx * CLAMP(y, ny)) + CLAMP(x, nx);

        uchar g_value = readChannel(d_rgb, clamped_index, 1);
        uchar b_value = readChannel(d_rgb, clamped_index, 2);
        uchar gb_average = ((uint)g_value + (uint)b_value) / 2;

        writeChannel(d_out, clamped_index, 0, gb_average);
      }
    }
  }
}

#ifdef __cplusplus
}
#endif
