#include <gputk.h>

#define TILE_WIDTH 16

__global__ void matMul(float *A, float *B, float *C,
                       int numARows, int numACols, int numBCols) {
  //@@ Declare shared memory for tiles of A and B
  __shared__ float tileA[TILE_WIDTH][TILE_WIDTH];
  __shared__ float tileB[TILE_WIDTH][TILE_WIDTH];
                    
  int row = blockIdx.y * TILE_WIDTH + threadIdx.y;
  int col = blockIdx.x * TILE_WIDTH + threadIdx.x;
  float val = 0.0f;

  int numTiles = (numACols + TILE_WIDTH - 1) / TILE_WIDTH;

  for (int t = 0; t < numTiles; t++) {
    //@@ Load tile of A into shared memory (with boundary check, use 0.0f for out-of-bounds)
    //@@ Load tile of B into shared memory (with boundary check, use 0.0f for out-of-bounds)
    tileA[threadIdx.y][threadIdx.x] = (row < numARows && aCol < numACols)

    //@@ Synchronize to ensure tiles are loaded
    __syncthreads();

    //@@ Accumulate partial dot product from this tile

    //@@ Synchronize before loading next tile
  }

  //@@ Write result to C (with boundary check)
}

int main(int argc, char **argv) {
  gpuTKArg_t args;
  float *hostA, *hostB, *hostC;
  float *deviceA, *deviceB, *deviceC;
  int numARows, numACols;
  int numBRows, numBCols;
  int numCRows, numCCols;

  args = gpuTKArg_read(argc, argv);

  gpuTKTime_start(Generic, "Importing data and creating memory on host");
  hostA = (float *)gpuTKImport(gpuTKArg_getInputFile(args, 0), &numARows, &numACols);
  hostB = (float *)gpuTKImport(gpuTKArg_getInputFile(args, 1), &numBRows, &numBCols);
  numCRows = numARows;
  numCCols = numBCols;
  hostC = (float *)malloc(numCRows * numCCols * sizeof(float));
  gpuTKTime_stop(Generic, "Importing data and creating memory on host");

  gpuTKLog(TRACE, "A: ", numARows, " x ", numACols);
  gpuTKLog(TRACE, "B: ", numBRows, " x ", numBCols);
  gpuTKLog(TRACE, "C: ", numCRows, " x ", numCCols);

  gpuTKTime_start(GPU, "Allocating GPU memory.");
  //@@ Allocate GPU memory for A, B, C

  gpuTKTime_stop(GPU, "Allocating GPU memory.");

  gpuTKTime_start(GPU, "Copying input memory to the GPU.");
  //@@ Copy A and B to GPU

  gpuTKTime_stop(GPU, "Copying input memory to the GPU.");

  //@@ Initialize grid and block dimensions
  //   Block: (TILE_WIDTH, TILE_WIDTH)
  //   Grid: (ceil(numBCols/TILE_WIDTH), ceil(numARows/TILE_WIDTH))

  gpuTKTime_start(Compute, "Performing CUDA computation");
  //@@ Launch matMul kernel

  cudaDeviceSynchronize();
  gpuTKTime_stop(Compute, "Performing CUDA computation");

  gpuTKTime_start(Copy, "Copying output memory to the CPU");
  //@@ Copy C from GPU to host

  gpuTKTime_stop(Copy, "Copying output memory to the CPU");

  gpuTKTime_start(GPU, "Freeing GPU Memory");
  //@@ Free GPU memory

  gpuTKTime_stop(GPU, "Freeing GPU Memory");

  gpuTKSolution(args, hostC, numCRows, numCCols);

  free(hostA);
  free(hostB);
  free(hostC);

  return 0;
}
