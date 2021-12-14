#include <math.h>
#include "cuda_helpers.h"

#ifdef __cplusplus
extern "C" {
#endif

__global__
void rgb_to_xyY(float3* d_rgb,
                           float3* d_xyY, float delta,
                           int num_pixels_y, int num_pixels_x) {
  int ny = num_pixels_y;
  int nx = num_pixels_x;
  uint2 image_index_2d = {
    (blockIdx.x * blockDim.x) + threadIdx.x,
    (blockIdx.y * blockDim.y) + threadIdx.y
  };
  int image_index_1d = (nx * image_index_2d.y) + image_index_2d.x;

  if (image_index_2d.x < nx && image_index_2d.y < ny) {
    float3 rgb = d_rgb[image_index_1d];
    float r = rgb.x;
    float g = rgb.y;
    float b = rgb.z;

    float X = (r * 0.4124f) + (g * 0.3576f) + (b * 0.1805f);
    float Y = (r * 0.2126f) + (g * 0.7152f) + (b * 0.0722f);
    float Z = (r * 0.0193f) + (g * 0.1192f) + (b * 0.9505f);

    float L = X + Y + Z;

    float3 xyY = {X / L, Y / L, log10f(delta + Y)};
    d_xyY[image_index_1d] = xyY;
  }
}

__global__
void normalize_cdf(unsigned int *d_input_cdf, float *d_output_cdf, int n) {

  int myId = (blockIdx.x * blockDim.x) + threadIdx.x;
  if (myId >= n) return;

  const float normalized = 1.f / d_input_cdf[n - 1];
  d_output_cdf[myId] = d_input_cdf[myId] * normalized;
}

__global__
void tone_map(const float3* d_xyY,
                        const float *d_cdf_norm, float3* d_rgb_new, float min_log_Y,
                        float log_Y_range, int num_bins, int num_pixels_y,
                        int num_pixels_x) {
  int ny = num_pixels_y;
  int nx = num_pixels_x;
  uint2 image_index_2d = {
    (blockIdx.x * blockDim.x) + threadIdx.x,
    (blockIdx.y * blockDim.y) + threadIdx.y
  };
  int image_index_1d = (nx * image_index_2d.y) + image_index_2d.x;

  if (image_index_2d.x < nx && image_index_2d.y < ny) {
    float3 xyY = d_xyY[image_index_1d];
    float x = xyY.x;
    float y = xyY.y;
    float log_Y = xyY.z;

    int bin_index = (log_Y - min_log_Y) / log_Y_range * num_bins;
    bin_index = MIN(num_bins - 1, bin_index);
    float Y_new = d_cdf_norm[bin_index];

    float X_new = x * (Y_new / y);
    float Z_new = (1 - x - y) * (Y_new / y);

    float3 rgb_new = {
      (X_new * 3.2406f) + (Y_new * -1.5372f) + (Z_new * -0.4986f),
      (X_new * -0.9689f) + (Y_new * 1.8758f) + (Z_new * 0.0415f),
      (X_new * 0.0557f) + (Y_new * -0.2040f) + (Z_new * 1.0570f),
    };

    d_rgb_new[image_index_1d] = rgb_new;
  }
}


__global__
void reduce_minmax_lum(const float3* d_xyY, float2 * d_out, int num_pixels)
{
    // sdata is allocated in the kernel call: 3rd arg to <<<b, t, shmem>>>
    SHARED(sdata, float2);
    int myId = threadIdx.x + blockDim.x * blockIdx.x;
    if (myId >= num_pixels) {
      return;
    }
    int tid  = threadIdx.x;

    // load shared mem from global mem
    float2 minmax = {d_xyY[myId].z, d_xyY[myId].z};
    sdata[tid] = minmax;
    __syncthreads();            // make sure entire block is loaded!

    // do reduction in shared mem
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1)
    {
        if (tid < s)
        {
            minmax.x = MIN(sdata[tid].x, sdata[tid + s].x);
            minmax.y = MAX(sdata[tid].y, sdata[tid + s].y);
            sdata[tid] = minmax;
        }
        __syncthreads();        // make sure all adds at one stage are done!
    }

    // only thread 0 writes result for this block back to global mem
    if (tid == 0)
    {
        d_out[blockIdx.x] = sdata[0];
    }
}

__global__
void reduce_minmax(const float2* d_in, float2 * d_out, int num_pixels)
{
    // sdata is allocated in the kernel call: 3rd arg to <<<b, t, shmem>>>
    SHARED(sdata, float2);
    int myId = threadIdx.x + blockDim.x * blockIdx.x;
    if (myId >= num_pixels) {
      return;
    }
    int tid  = threadIdx.x;
    // load shared mem from global mem
    sdata[tid] = d_in[myId];
    __syncthreads();            // make sure entire block is loaded!

    float2 minmax;
    // do reduction in shared mem
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1)
    {
        if (tid < s)
        {
            minmax.x = MIN(sdata[tid].x, sdata[tid + s].x);
            minmax.y = MAX(sdata[tid].y, sdata[tid + s].y);
            sdata[tid] = minmax;
        }
        __syncthreads();        // make sure all adds at one stage are done!
    }

    // only thread 0 writes result for this block back to global mem
    if (tid == 0)
    {
        d_out[blockIdx.x] = sdata[0];
    }
}

__global__
void lum_histo(uint *d_bins, const float3 *d_xyY, float lum_min, float lum_range, int bin_count, int num_pixels)
{
    int myId = threadIdx.x + blockDim.x * blockIdx.x;
    if (myId >= num_pixels) return;

    float lum = d_xyY[myId].z;
    int bin = (lum - lum_min) / lum_range * (float)bin_count;
    bin = MIN(bin, bin_count - 1);
    bin = MAX(bin, 0);
    atomicAdd(&(d_bins[bin]), 1);
}


__global__
void computeCdf(float* d_cdf, const uint *d_bins, int bin_count)
{
    int myId = threadIdx.x + blockDim.x * blockIdx.x;
    if (myId >= bin_count) return;

    // That's probably not the best way of using a GPU :-)
    // TODO implement scan algorithm
    uint prefix_sum = 0;
    int i = 0;
    for (i = 0; i < myId; ++i) {
      prefix_sum += d_bins[i];
    }
    uint total = prefix_sum;
    for (; i < bin_count; ++i) {
      total += d_bins[i];
    }
    float normalized = 1.0 / total;
    d_cdf[myId] = prefix_sum * normalized;
}


__global__
void blellochCdf(float* d_cdf, uint* d_bins, int n) {
    // We need synchronization across all threads so only one block
    if (gridDim.x > 1) return;
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


#ifdef __cplusplus
}
#endif
