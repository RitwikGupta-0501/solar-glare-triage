#include <iostream>
#include <algorithm>
#include <cuda_runtime.h>

#define STB_IMAGE_IMPLEMENTATION
#include "../include/stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "../include/stb_image_write.h"
#include "kernels.cu"

// Error checking macro
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cout << "Usage: ./glare_normalizer <image_path>" << std::endl;
        return -1;
    }

    int width, height, channels;
    // Load as grayscale
    unsigned char *h_img = stbi_load(argv[1], &width, &height, &channels, 1);
    if (!h_img) {
        std::cerr << "Error: Could not load image." << std::endl;
        return -1;
    }

    size_t img_size_bytes = width * height * sizeof(unsigned char);

    // Define the Tile and Grid metrics
    int TILE_SIZE = 16;
    int grid_w = (width + TILE_SIZE - 1) / TILE_SIZE;
    int grid_h = (height + TILE_SIZE - 1) / TILE_SIZE;
    size_t grid_size_bytes = grid_w * grid_h * sizeof(unsigned char);

    std::cout << "Loaded image: " << width << "x" << height << " (" << img_size_bytes << " bytes)" << std::endl;
    std::cout << "Tile Grid Topology: " << grid_w << "x" << grid_h << " tiles (" << grid_size_bytes << " bytes total)" << std::endl;

    // 1. Allocate Device Memory
    unsigned char *d_img;
    unsigned char *d_grid_mask;

    gpuErrchk(cudaMalloc(&d_img, img_size_bytes));
    gpuErrchk(cudaMalloc(&d_grid_mask, grid_size_bytes));

    // Zero out the grid mask to prevent garbage values
    gpuErrchk(cudaMemset(d_grid_mask, 0, grid_size_bytes));

    // 2. Copy the massive image to the device
    gpuErrchk(cudaMemcpy(d_img, h_img, img_size_bytes, cudaMemcpyHostToDevice));

    // 3. Launch the Triage Kernel
    dim3 threadsPerBlock(TILE_SIZE, TILE_SIZE);
    dim3 blocksPerGrid(grid_w, grid_h);

    std::cout << "Scanning 16MP matrix for sensor clipping..." << std::endl;
    detectGlareTiles<<<blocksPerGrid, threadsPerBlock>>>(d_img, d_grid_mask, width, height, grid_w);

    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());

    // 4. Copy ONLY the microscopic grid mask back to the CPU (Massive PCIe Optimization)
    unsigned char *h_grid_mask = (unsigned char*)malloc(grid_size_bytes);
    gpuErrchk(cudaMemcpy(h_grid_mask, d_grid_mask, grid_size_bytes, cudaMemcpyDeviceToHost));

    // 5. CPU Overlay: Draw bounding boxes on the compromised tiles
    std::cout << "Drawing telemetry bounds on compromised sectors..." << std::endl;

    // Thickness of the bounding box lines
    int thickness = 3;

    for (int gy = 0; gy < grid_h; gy++) {
        for (int gx = 0; gx < grid_w; gx++) {

            // If the GPU flagged this tile as a glare zone
            if (h_grid_mask[gy * grid_w + gx] == 1) {

                int start_x = gx * TILE_SIZE;
                int start_y = gy * TILE_SIZE;
                int end_x = std::min(start_x + TILE_SIZE, width);
                int end_y = std::min(start_y + TILE_SIZE, height);

                // Draw a box inside the tile boundaries
                for(int y = start_y; y < end_y; y++) {
                    for(int x = start_x; x < end_x; x++) {
                        // If the pixel is on the outer edge of the tile, paint it solid white (255)
                        if (x < start_x + thickness || x > end_x - thickness - 1 ||
                            y < start_y + thickness || y > end_y - thickness - 1) {

                            h_img[y * width + x] = 0;
                        }
                    }
                }
            }
        }
    }

    // 6. Save the final image directly from the CPU buffer
    stbi_write_jpg("data/processed/triage_overlay.jpg", width, height, 1, h_img, 100);
    std::cout << "Success! Check data/processed/triage_overlay.jpg" << std::endl;

    // 7. Cleanup
    free(h_grid_mask);
    stbi_image_free(h_img);
    cudaFree(d_img);
    cudaFree(d_grid_mask);

    return 0;
}
