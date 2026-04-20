#include <cuda_runtime.h>
#include <device_launch_parameters.h>

// Tile dimensions matching our block size (16x16 = 256 threads)
#define TILE_W 16
#define TILE_H 16

__global__ void detectGlareTiles(unsigned char* d_img, unsigned char* d_grid_mask, int width, int height, int grid_w) {
    // Shared memory to hold the glare count for the 256 pixels in this specific tile
    __shared__ int s_glare_counts[TILE_W * TILE_H];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int bx = blockIdx.x;
    int by = blockIdx.y;

    // Global pixel coordinates
    int x = bx * blockDim.x + tx;
    int y = by * blockDim.y + ty;

    // Local 1D thread index for shared memory (0 to 255)
    int tid = ty * blockDim.x + tx;

    // 1. Evaluate the pixel
    int is_glare = 0;
    if (x < width && y < height) {
        // If the pixel is blown out (e.g., > 240 brightness), flag it
        if (d_img[y * width + x] > 240) {
            is_glare = 1;
        }
    }
    s_glare_counts[tid] = is_glare;
    __syncthreads();

    // 2. Parallel Reduction to count total glare pixels in this tile
    for (unsigned int s = (blockDim.x * blockDim.y) / 2; s > 0; s >>= 1) {
        if (tid < s) {
            s_glare_counts[tid] += s_glare_counts[tid + s];
        }
        __syncthreads();
    }

    // 3. Thread 0 handles the memory write-back to the global grid mask
    if (tid == 0) {
        // If more than 12 pixels in this 256-pixel tile are blown out (~5%), mark the tile as compromised
        if (s_glare_counts[0] > 12) {
            d_grid_mask[by * grid_w + bx] = 1;
        } else {
            d_grid_mask[by * grid_w + bx] = 0;
        }
    }
}
