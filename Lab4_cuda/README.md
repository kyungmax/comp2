# Computing for Data Science 2 — SNU GSDS

# Lab 4: CUDA Parallel Reduction

This lab is based on the "GPU Teaching Kit Labs", produced jointly by NVIDIA and the University of Illinois (UIUC).

## Overview

Given a list of `n` floating-point numbers, compute the sum

```
result = lst[0] + lst[1] + ... + lst[n-1]
```

on the GPU using **parallel reduction**. You will implement three variants:

| Template                   | Technique                                              |
| -------------------------- | ------------------------------------------------------ |
| `template_basic.cu`        | Shared-memory reduction tree (lecture 16 algorithm)    |
| `template_warp.cu`         | Warp-level reduction with `__shfl_down_sync` primitive |
| `template_unified.cu`      | Reduction using CUDA Unified Memory (`cudaMallocManaged`) |

Each block reduces its own segment of the input into a single partial sum. The final scalar sum is then accumulated on the host across the partial sums (the test harness compares only `hostOutput[0]`).

## Server Environment

This is a **login node** — never compile or run GPU code directly. All builds and runs must go through SLURM on a compute node, after activating the pre-configured conda environment:

```bash
conda activate comp2
```

The `comp2` env provides `cuda-nvcc=12.4`, `cuda-cudart-dev=12.4`, `cuda-cudart-static=12.4`, and `cmake`. CUDA 12.4 is compatible with system `gcc 13.3`.

## Build Instructions

### Step 1 — Build libgputk (login node is fine)

```bash
cd /path/to/Lab4_cuda
mkdir build && cd build
cmake .. -DBUILD_LIBGPUTK_LIBRARY=ON -DBUILD_LOGTIME=ON
make
```

This produces `build/libgputk.a`, the support library linked into your template.

### Step 2 — Build a template variant (requires GPU via srun)

```bash
cd ../sources
srun -p class2 -N 1 --gres=gpu:1 make template_basic
# or
srun -p class2 -N 1 --gres=gpu:1 make template_warp
# or
srun -p class2 -N 1 --gres=gpu:1 make template_unified
```

Each target produces the binary `Reduction_Template`.

### Step 3 — Run on a single dataset

```bash
srun -p class2 -N 1 --gres=gpu:1 \
    ./Reduction_Template \
    -e "./Reduction/Dataset/0/output.raw" \
    -i "./Reduction/Dataset/0/input.raw" \
    -o "./test_result_0.raw" \
    -t vector
```

### Step 4 — Run all 10 datasets via helper script

```bash
srun -p class2 -N 1 --gres=gpu:1 bash test_all.sh -type basic
# or -type warp, -type unified
```

For a single dataset:

```bash
srun -p class2 -N 1 --gres=gpu:1 bash test.sh -dataset 3 -type warp
```

## What You Need to Implement

In each `template_<variant>.cu`, all locations are marked with `//@@`. You must implement:

1. The `__global__ void total(float *input, float *output, int len)` kernel:
   - Load a segment of the input into shared memory (or via the variant-specific mechanism).
   - Traverse the reduction tree.
   - Write the block's partial sum to `output[blockIdx.x]`.
2. In `main`:
   - Allocate GPU memory (or unified memory) for input/output.
   - Copy input H→D (skip for the unified-memory variant).
   - Configure grid/block dimensions.
   - Launch the kernel.
   - Copy partial sums D→H and reduce on the host into `hostOutput[0]`.
   - Free device memory.

## Submission

Submit a `.tar.gz` named `<student-id>.tar.gz` containing:

- `report.pdf`
- The template file(s) you implemented (`template_basic.cu`, `template_warp.cu`, and/or `template_unified.cu`)

```bash
tar czf 20xx-xxxxx.tar.gz report.pdf template_basic.cu template_warp.cu template_unified.cu
```
