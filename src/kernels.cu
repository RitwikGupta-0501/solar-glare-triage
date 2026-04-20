#include <cuda_runtime.h>
#include <device_launch_parameters.h>

// 1. Horizontal Pass
__global__ void blurHorizontal(unsigned char* d_in, float* d_out, int width, int height, int radius) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        float sum = 0.0f;
        int count = 0;

        // Slide window horizontally
        for (int dx = -radius; dx <= radius; dx++) {
            int currentX = x + dx;
            // Clamp to image boundaries
            if (currentX >= 0 && currentX < width) {
                sum += (float)d_in[y * width + currentX];
                count++;
            }
        }
        // Write out the average as a float to prevent precision loss
        d_out[y * width + x] = sum / count;
    }
}

// 2. Vertical Pass
__global__ void blurVertical(float* d_in, float* d_out, int width, int height, int radius) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        float sum = 0.0f;
        int count = 0;

        // Slide window vertically
        for (int dy = -radius; dy <= radius; dy++) {
            int currentY = y + dy;
            if (currentY >= 0 && currentY < height) {
                sum += d_in[currentY * width + x];
                count++;
            }
        }
        d_out[y * width + x] = sum / count;
    }
}

// 3. The Subtraction (High-Pass)
__global__ void subtractIllumination(unsigned char* d_original, float* d_blurred, unsigned char* d_final, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        int idx = y * width + x;

        float orig = (float)d_original[idx];
        float blur = d_blurred[idx];

        // P_new = (Original - Blurred) + 128 (Neutral Gray Offset)
        float result = (orig - blur) + 128.0f;

        // Clamp just in case
        if (result < 0.0f) result = 0.0f;
        if (result > 255.0f) result = 255.0f;

        d_final[idx] = (unsigned char)result;
    }
}
