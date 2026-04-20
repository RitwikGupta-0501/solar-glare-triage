#include <iostream>
#include <cuda_runtime.h>

#define STB_IMAGE_IMPLEMENTATION
#include "../include/stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "../include/stb_image_write.h"

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

    // Allocate Device Memory
    unsigned char *d_img;
    gpuErrchk(cudaMalloc(&d_img, img_size));

    // Copy Host to Device
    gpuErrchk(cudaMemcpy(d_img, h_img, img_size, cudaMemcpyHostToDevice));

    // --- PHASE 2 & 3 WILL LIVE HERE ---
    // For Phase 1, we just copy it back to prove the bridge works
    // -----------------------------------

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
