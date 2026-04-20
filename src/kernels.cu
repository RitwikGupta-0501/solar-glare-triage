#include <cuda_runtime.h>
#include <device_launch_parameters.h>

__global__ void findMaxKernel(unsigned char* d_img, int* d_block_maxes, int n) {
    // Static allocation is safer for debugging
    __shared__ int sdata[256];

    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    // Load and cast to int immediately
    sdata[tid] = (i < n) ? (int)d_img[i] : 0;
    __syncthreads();

    // Standard reduction loop
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            if (sdata[tid + s] > sdata[tid]) {
                sdata[tid] = sdata[tid + s];
            }
        }
        __syncthreads();
    }

    if (tid == 0) d_block_maxes[blockIdx.x] = sdata[0];
}

__global__ void findMinKernel(unsigned char* d_img, int* d_block_mins, int n) {
    __shared__ int sdata[256];

    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    // Load 255 for out-of-bounds so they don't win the "Min" contest
    sdata[tid] = (i < n) ? (int)d_img[i] : 255;
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            if (sdata[tid + s] < sdata[tid]) {
                sdata[tid] = sdata[tid + s];
            }
        }
        __syncthreads();
    }

    if (tid == 0) d_block_mins[blockIdx.x] = sdata[0];
}
