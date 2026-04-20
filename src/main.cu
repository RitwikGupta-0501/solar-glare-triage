#include <iostream>
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

    size_t img_size = width * height * sizeof(unsigned char);
    std::cout << "Loaded image: " << width << "x" << height << " (" << img_size << " bytes)" << std::endl;

    unsigned char *d_img;
    gpuErrchk(cudaMalloc(&d_img, img_size));

    gpuErrchk(cudaMemcpy(d_img, h_img, img_size, cudaMemcpyHostToDevice));

    int threadsPerBlock = 256;
    int blocksPerGrid = (width * height + threadsPerBlock - 1) / threadsPerBlock;

    int *d_block_maxes;
    int *d_block_mins;

    gpuErrchk(cudaMalloc(&d_block_maxes, blocksPerGrid * sizeof(int)));
    gpuErrchk(cudaMalloc(&d_block_mins, blocksPerGrid * sizeof(int)));

    cudaMemset(d_block_maxes, 0, blocksPerGrid * sizeof(int));
    cudaMemset(d_block_mins, 255, blocksPerGrid * sizeof(int));

    findMaxKernel<<<blocksPerGrid, threadsPerBlock>>>(d_img, d_block_maxes, width * height);
    findMinKernel<<<blocksPerGrid, threadsPerBlock>>>(d_img, d_block_mins, width * height);

    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());

    int *h_block_maxes = (int*)malloc(blocksPerGrid * sizeof(int));
    cudaMemcpy(h_block_maxes, d_block_maxes, blocksPerGrid * sizeof(int), cudaMemcpyDeviceToHost);
    int *h_block_mins = (int*)malloc(blocksPerGrid * sizeof(int));
    cudaMemcpy(h_block_mins, d_block_mins, blocksPerGrid * sizeof(int), cudaMemcpyDeviceToHost);

    int globalMax = 0;
    for(int i = 0; i < blocksPerGrid; i++) {
        if(h_block_maxes[i] > globalMax) globalMax = h_block_maxes[i];
    }

    int globalMin = 255;
    for(int i = 0; i < blocksPerGrid; i++) {
        if(h_block_mins[i] < globalMin) globalMin = h_block_mins[i];
    }

    if (globalMax == globalMin) globalMax++;

    std::cout << "Glare Peak (Max): " << globalMax << " | Darkest Point (Min): " << globalMin << std::endl;

    unsigned char *h_out = (unsigned char*)malloc(img_size);
    gpuErrchk(cudaMemcpy(h_out, d_img, img_size, cudaMemcpyDeviceToHost));

    stbi_write_jpg("data/processed/test_output.jpg", width, height, 1, h_out, 100);

    std::cout << "Success! Check data/processed/test_output.jpg" << std::endl;

    // Cleanup
    free(h_out);
    stbi_image_free(h_img);
    cudaFree(d_img);

    return 0;
}
