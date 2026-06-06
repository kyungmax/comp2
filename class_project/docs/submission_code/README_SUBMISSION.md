# Submission Code Package

This directory contains the core implementation and report files for the HNSW GPU multi-entry upper-layer descent project.

## Contents

- `src/gpu_upper_descent.cu`
  - CUDA implementation of GPU-parallel K-way upper-layer descent.
  - One warp handles one `(query, seed)` descent.
  - Warp lanes split the vector-distance computation and use shuffle reduction.

- `src/bench_gpu_upper_descent.cpp`
  - Benchmark and artifact-generation binary for GPU upper descent.
  - Produces `gpu_best_k*_w*_nearest.i32`, distance files, and timing metadata.

- `src/upper_descent.cpp`
  - CPU reference implementation for upper-layer descent and level-0 graph search helpers.

- `include/hnsw_week2/*.hpp`
  - C++ public interfaces and graph/result structs.

- `scripts/export_faiss_upper_graph.py`
  - Exports a Faiss HNSW graph into the C++ binary CSR format.

- `scripts/faiss_hnsw_multi_entry.py`
  - Runs baseline Faiss HNSW, CPU multi-entry upper descent, and downstream evaluation from GPU-generated entry files.

- `tests/week2_parity.cpp`
  - CPU/GPU parity test. If CUDA is unavailable at runtime, GPU checks are skipped.

- `CMakeLists.txt`
  - Build configuration for the CPU library, CUDA library/stub, benchmark, and test binary.

- `report/final_report_ko.docx`
  - Final Korean report.

- `report/final_report.md`
  - Markdown report source.

## Not Included

Large generated data files are intentionally not included:

- `data/glove-200-angular.hdf5`
- exported Faiss index files
- exported vector/graph binary files
- generated GPU entry artifacts

Those files are too large for a compact code submission and should be regenerated or provided separately if required.

## Build

From the project root, with this directory copied back as the project root layout:

```bash
cmake -S . -B build
cmake --build build -j
ctest --test-dir build --output-on-failure
```

If CUDA is not available, the project builds a CUDA stub and the GPU parity portion of the test is skipped.

## Main Reproduction Commands

Export Faiss HNSW graph:

```bash
PYTHONPATH=.pydeps python3 scripts/export_faiss_upper_graph.py \
  --dataset data/glove-200-angular.hdf5 \
  --out-dir data/faiss_upper_full_m16_efc200_q10000 \
  --train-size 0 \
  --query-size 10000 \
  --M 16 \
  --ef-construction 200 \
  --ef-search 32 \
  --metric angular \
  --write-index
```

Generate GPU upper-descent entries:

```bash
./build/bench_gpu_upper_descent \
  data/faiss_upper_full_m16_efc200_q10000 \
  1,16,32,64,128,256 \
  3 \
  3 \
  data/faiss_upper_full_m16_efc200_q10000/gpu_entries
```

Evaluate GPU entries with unchanged Faiss level-0 search:

```bash
PYTHONPATH=.pydeps python3 scripts/faiss_hnsw_multi_entry.py \
  --dataset data/glove-200-angular.hdf5 \
  --train-size 0 \
  --query-size 10000 \
  --result-k 10 \
  --M 16 \
  --ef-construction 200 \
  --ef-search 16,32,64,128 \
  --faiss-index data/faiss_upper_full_m16_efc200_q10000/index.faiss \
  --gpu-entry-dir data/faiss_upper_full_m16_efc200_q10000/gpu_entries \
  --entry-count 1,16,32,64,128,256 \
  --seed-policy none \
  --upper-window 3 \
  --handoff best \
  --metric angular
```

