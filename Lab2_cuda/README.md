# Computing for Data Science 2 - Lab 2 (Spring 2026)

## CUDA Batched Tiled Matrix Multiplication

## Prerequisites

- Miniconda environment `comp2` activated: `conda activate comp2`
- All GPU compilation and execution must be done on **compute nodes** via SLURM:
  ```
  srun -p class2 -N 1 --gres=gpu:1 <command>
  ```

## Build Instructions

### 1. Build libgputk (on login node — no GPU needed)

```bash
mkdir -p build && cd build
cmake .. -DBUILD_LIBGPUTK_LIBRARY=ON -DBUILD_LOGTIME=ON
make -j4
cd ..
```

### 2. Build template, data generator, and generate datasets (on compute node)

```bash
srun -p class2 -N 1 --gres=gpu:1 bash -c \
  'cd sources && make dataset_generator && ./dataset_generator && make template'
```

## Running

### Single dataset

```bash
srun -p class2 -N 1 --gres=gpu:1 bash -c \
  'cd sources && ./BatchedMatMul_template \
    -e BatchedMatMul/Dataset/0/output.raw \
    -i BatchedMatMul/Dataset/0/input0.raw,BatchedMatMul/Dataset/0/input1.raw,BatchedMatMul/Dataset/0/input2.raw \
    -t matrix'
```

### Using test scripts (from inside sources/)

```bash
# Single dataset (default: 0)
srun -p class2 -N 1 --gres=gpu:1 bash test.sh -dataset 3

# All 15 datasets (0–14)
srun -p class2 -N 1 --gres=gpu:1 bash test_all.sh
```

## Datasets

The dataset generator produces two groups under `BatchedMatMul/Dataset/`:

- **Correctness datasets (0–8)** — varied `(batch, M, K, N)` shapes for verifying the
  kernel on different sizes (square, rectangular, non-tile-aligned). Use these for
  Report Q6 timing.
- **Batch-scaling datasets (9–14)** — fixed per-batch size `64 × 64 × 64` with
  `batch ∈ {1, 2, 4, 8, 16, 32}`. Use these for Report Q7 (batch-level parallelism
  experiment).

## Input / Output Format

Each dataset directory under `BatchedMatMul/Dataset/<n>/` contains:

| File | Description |
|------|-------------|
| `input0.raw` | A matrices flattened as `(batch * M)` rows × `K` cols |
| `input1.raw` | B matrices flattened as `(batch * K)` rows × `N` cols |
| `input2.raw` | **4×1 column vector** containing `[batch, M, K, N]` (header `4 1` to avoid gpuTKImport 1-row swap quirk) |
| `output.raw` | C matrices flattened as `(batch * M)` rows × `N` cols |

## Submission

Students submit:
- Modified `template.cu`
- `report.pdf`

Packaged as `20xx-xxxxx.tar.gz`.
