#include "cuda.h"
#include "cuda_helpers.h"

#ifdef __cplusplus
extern "C" {
#endif

__global__ void quicksort(float* d_out, int n) {
    int tid = threadIdx.x + blockDim.x * blockIdx.x;

    if (n == 1) return;

    // TODO actual run partitionning
    int pivot_id = n / 2;

    // CUstream sub_stream;
    if (tid == 0 && pivot_id > 0) {
        // Left stream
        // cudaStreamCreateWithFlags(&sub_stream, CU_STREAM_NON_BLOCKING);
        // quicksort<<<(n + 1023) / 1024, 1024, 0, sub_stream>>>(&d_out[pivot_id], n - pivot_id);
    }
    // TODO: right stream
    // Even better we can start the two streams at once.

}


#ifdef __cplusplus
}
#endif
