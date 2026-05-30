# Computing for Data Science 2 — SNU GSDS

# Lab 3: CUDA Tiled Convolution with Streams and Pinned Memory

This lab is based on the "GPU Teaching Kit Labs", produced jointly by NVIDIA and the University of Illinois (UIUC).

## Overview

Implement a **tiled 2-D image convolution** kernel in CUDA that uses:
- **Shared memory tiling** to reduce global memory bandwidth
- **CUDA streams** to overlap H↔D data transfers with kernel execution
- **Pinned (page-locked) host memory** (`cudaMallocHost`) for efficient async transfers

The convolution uses a 5×5 mask applied to each channel of an RGB image.

## Build Instructions

### Step 1 — Build libgputk (login node is fine)

```bash
conda activate comp2
cd /path/to/Lab3_cuda
mkdir build && cd build
cmake .. -DBUILD_LIBGPUTK_LIBRARY=ON -DBUILD_LOGTIME=ON
make
```

### Step 2 — Build template and data generator (requires GPU via srun)

```bash
cd sources
srun -p class2 -N 1 --gres=gpu:1 make convolution
srun -p class2 -N 1 --gres=gpu:1 make dataset_generator_convolution
```

### Step 3 — Generate test datasets

```bash
srun -p class2 -N 1 --gres=gpu:1 ./dataset_generator_convolution
```

This creates `Convolution/Dataset/0/`, `1/`, … each containing:
- `input0.ppm` — input RGB image
- `input1.raw` — 5×5 convolution mask (25 floats)
- `output.ppm` — expected output image

### Step 4 — Run the template

```bash
srun -p class2 -N 1 --gres=gpu:1 \
    ./Convolution_template \
    -e "./Convolution/Dataset/0/output.ppm" \
    -i "./Convolution/Dataset/0/input0.ppm,./Convolution/Dataset/0/input1.raw" \
    -o "./test_result_0.ppm" \
    -t image
```

Or use the helper script:

```bash
bash test.sh -dataset 0
```

## What You Need to Implement

Edit `sources/template_convolution.cu`. All locations are marked with `//@@ INSERT CODE HERE`.

Key tasks:
1. **Kernel** (`convolution`): implement shared-memory tiling — load the `w×w` halo tile, sync, compute the convolution sum, clamp to `[0,1]`, write output.
2. **Memory allocation**: allocate pinned host buffers (`cudaMallocHost`) and device buffers; create streams.
3. **Async H→D transfers**: divide the image into `numStreams` horizontal slices; use `cudaMemcpyAsync` per stream.
4. **Kernel launch**: one kernel call per stream slice (with correct offsets and row counts).
5. **Async D→H transfers**: copy each output slice back asynchronously; synchronize all streams.
6. **Cleanup**: free device/pinned memory, destroy streams.

## Submission

Submit a `.tar.gz` named `<student-id>.tar.gz` containing:
- `report.pdf`
- `template_convolution.cu` (your implementation)

```bash
tar czf 20xx-xxxxx.tar.gz report.pdf template_convolution.cu
```
