# Computing for Data Science 2 - Lab 1 (Spring 2026)

## CUDA Vector Addition & Tiled Matrix Multiplication

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

### 2. Build templates and data generators (on compute node)

```bash
cd sources
srun -p class2 -N 1 --gres=gpu:1 bash -c \
  'make all && make generators && \
   ./dataset_generator_vecadd && \
   ./dataset_generator_matmul'
```

## Running

### Part A: Vector Addition

```bash
srun -p class2 -N 1 --gres=gpu:1 ./VectorAdd_template \
  -e VectorAdd/Dataset/0/output.raw \
  -i VectorAdd/Dataset/0/input0.raw,VectorAdd/Dataset/0/input1.raw \
  -t vector
```

### Part B: Tiled Matrix Multiplication

```bash
srun -p class2 -N 1 --gres=gpu:1 ./TiledMatMul_template \
  -e MatMul/Dataset/0/output.raw \
  -i MatMul/Dataset/0/input0.raw,MatMul/Dataset/0/input1.raw \
  -t matrix
```

**Note:** Use `-t vector` for Part A and `-t matrix` for Part B. The `-e` flag specifies the expected output for correctness checking. The `-i` flag takes comma-separated input files (no spaces after comma).
