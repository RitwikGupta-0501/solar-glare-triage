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
    unsigned char *h_img = stbi_load(argv[1], &width, &height, &channels, 1);
    if (!h_img) {
        std::cerr << "Error: Could not load image." << std::endl;
        return -1;
    }

    size_t img_size_bytes = width * height * sizeof(unsigned char);
    size_t float_size_bytes = width * height * sizeof(float);
    std::cout << "Loaded image: " << width << "x" << height << " (" << img_size_bytes << " bytes)" << std::endl;

    // 1. Allocate Device Memory
    unsigned char *d_original;
    float *d_blur_temp;
    float *d_blur_final;
    unsigned char *d_final_out;

    gpuErrchk(cudaMalloc(&d_original, img_size_bytes));
    gpuErrchk(cudaMalloc(&d_blur_temp, float_size_bytes));
    gpuErrchk(cudaMalloc(&d_blur_final, float_size_bytes));
    gpuErrchk(cudaMalloc(&d_final_out, img_size_bytes));

    // 2. Copy original image to GPU
    gpuErrchk(cudaMemcpy(d_original, h_img, img_size_bytes, cudaMemcpyHostToDevice));

    // 3. Setup 2D Grid and Blocks
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid(
        (width + threadsPerBlock.x - 1) / threadsPerBlock.x,
        (height + threadsPerBlock.y - 1) / threadsPerBlock.y
    );

    // 4. Execute the Pipeline
    // A radius of 35 gives a 71x71 pixel window, large enough to smooth out the grid lines
    int blurRadius = 35;

    std::cout << "Generating Illumination Map (Horizontal Pass)..." << std::endl;
    blurHorizontal<<<blocksPerGrid, threadsPerBlock>>>(d_original, d_blur_temp, width, height, blurRadius);
    gpuErrchk(cudaPeekAtLastError());

    std::cout << "Generating Illumination Map (Vertical Pass)..." << std::endl;
    blurVertical<<<blocksPerGrid, threadsPerBlock>>>(d_blur_temp, d_blur_final, width, height, blurRadius);
    gpuErrchk(cudaPeekAtLastError());

    std::cout << "Neutralizing Glare (Spatial Subtraction)..." << std::endl;
    subtractIllumination<<<blocksPerGrid, threadsPerBlock>>>(d_original, d_blur_final, d_final_out, width, height);
    gpuErrchk(cudaPeekAtLastError());

    gpuErrchk(cudaDeviceSynchronize());

    // 5. Copy Back and Save
    unsigned char *h_out = (unsigned char*)malloc(img_size_bytes);
    gpuErrchk(cudaMemcpy(h_out, d_final_out, img_size_bytes, cudaMemcpyDeviceToHost));

    stbi_write_jpg("data/processed/spatial_fixed.jpg", width, height, 1, h_out, 100);
    std::cout << "Success! Check data/processed/spatial_fixed.jpg" << std::endl;

    // 6. Cleanup
    free(h_out);
    stbi_image_free(h_img);
    cudaFree(d_original);
    cudaFree(d_blur_temp);
    cudaFree(d_blur_final);
    cudaFree(d_final_out);

    return 0;
}
