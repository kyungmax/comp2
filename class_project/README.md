# HNSW Week 2 Class Project Implementation

This folder contains the Week 2 implementation for the proposal:

`GPU-Parallel Multi-Entry Descent for HNSW: Entry-Point Quality as a Lever for Recall`.

The course project document asks for a CUDA/data-parallel implementation with functionality, performance evidence, code quality, and a final report. The proposal's Week 2 target is:

- CPU K-way multi-entry upper-layer descent from Week 1.
- GPU K-way upper-layer descent kernel.
- One block per query, one warp per descent.
- Verify GPU/CPU parity for `K in {1, 4, 16}`.

## What Is Implemented

- `include/hnsw_week2/upper_descent.hpp`
  - HNSW-like CSR graph representation for levels `0..max_level`.
  - CPU greedy upper-layer descent equivalent to hnswlib's top-layer search.
  - CPU K-way multi-entry descent.
  - Layer-0 beam search that accepts the deduplicated K entry points.

- `src/gpu_upper_descent.cu`
  - CUDA K-way upper-layer descent.
  - Launch mapping: `grid.x = query_count`, `blockDim.x = K * 32`.
  - Each warp executes one independent greedy descent.
  - Warp-shuffle reduction computes L2 distance cooperatively across dimensions.

- `tests/week2_parity.cpp`
  - Builds a deterministic synthetic HNSW-like graph.
  - Checks that best-of-K entry distance is monotonic for `K = 1, 4, 16`.
  - Runs layer-0 search from K entry points and prints recall@5.
  - If CUDA is available, compares GPU and CPU entry points, distances, and downstream layer-0 results.

## Build and Test

CPU-only on this machine:

```bash
cmake -S . -B build
cmake --build build
ctest --test-dir build --output-on-failure
```

CUDA build on a machine with `nvcc`:

```bash
cmake -S . -B build-cuda -DHNSW_WEEK2_ENABLE_CUDA=ON
cmake --build build-cuda
ctest --test-dir build-cuda --output-on-failure
```

For an NVIDIA RTX A6000 server, pass the Ampere architecture explicitly if the toolchain does not infer it:

```bash
cmake -S . -B build-cuda -DHNSW_WEEK2_ENABLE_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=86
cmake --build build-cuda -j
ctest --test-dir build-cuda --output-on-failure
```

The local macOS environment currently does not have `nvcc`, so GPU parity code is implemented but not locally executable here.
