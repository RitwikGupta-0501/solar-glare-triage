# Solar Glare Triage Pipeline (CUDA)

## Overview
This project is a high-performance GPU pipeline designed as a pre-filter for an autonomous solar panel cleaning robot. Operating outdoors introduces harsh, dynamic lighting. When direct sunlight causes "sensor clipping" (blown-out white regions), it destroys local texture data. If fed raw into a computer vision daemon, these dead zones can cause false positives or mask real debris.

Instead of relying on a slow embedded CPU or using unpredictable AI inpainting to "hallucinate" missing data, this pipeline uses a deterministic **Tile-Based Spatial Reduction** written in pure CUDA. It acts as a high-speed anomaly detector, scanning high-resolution matrices in milliseconds and flagging sensor-blind spots so the downstream AI state machine can safely ignore them.

## Systems Architecture & Optimizations
* **Spatial Partitioning:** High-resolution images (e.g., 16MP) are mathematically sliced into a grid of 16x16 pixel tiles.
* **Shared Memory Reduction:** Each CUDA block (256 threads) processes exactly one tile. Threads concurrently evaluate their pixels and use `__shared__` memory to execute a parallel reduction, tallying the compromised pixels without race conditions.
* **PCIe Bandwidth Mitigation:** Instead of returning a modified 16MB image over the PCIe bus, the GPU returns a microscopic boolean mask (under 4KB). This reduces return bandwidth overhead by 99.9%.
* **Zero-Bloat I/O:** Image loading and writing are handled via single-header `stb` libraries, entirely bypassing heavy dependencies like OpenCV.

## Dependencies
* **NVIDIA CUDA Toolkit** (`nvcc`)
* C++ Standard Library
* **stb_image.h / stb_image_write.h** (placed in the `include/` directory)

## Project Structure
```text
solar_glare_normalizer/
├── src/
│   ├── main.cu        # CPU Orchestration, Memory Bridging, Overlay Logic
│   └── kernels.cu     # Tile-based parallel reduction kernels
├── include/
│   ├── stb_image.h
│   └── stb_image_write.h
├── data/
│   ├── raw/           # Input images
│   └── processed/     # Output bounding-box images
└── Makefile           # Build configurations
```

## Build Instructions
The project is built using a standard `Makefile`. The compiler flags are currently optimized for the Pascal architecture (`sm_61`). 

To build the project, simply run:
```bash
make
```

To clean the build artifacts and clear out previously processed images:
```bash
make clean
```

## Usage
Execute the compiled binary and pass the path to the raw solar panel image as an argument:

```bash
./glare_normalizer data/raw/your_image.jpg
```

**Execution Flow:**
1. The CPU loads the JPEG into a flat byte array and allocates VRAM.
2. The GPU executes the triage scan across the tile grid.
3. The microscopic boolean grid is pulled back to the host.
4. The CPU draws tight bounding boxes around the compromised sectors.
5. The annotated image is saved to `data/processed/triage_overlay.jpg`.

## Expected Output
The resulting image will feature the original solar panel with thick bounding boxes drawn strictly over the areas where specular glare has caused sensor clipping. This output is ready to be passed to an object detection model, which can be programmed to ignore the boxed coordinates.
