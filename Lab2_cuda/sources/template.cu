#include <gputk.h>

#define gpuTKCheck(stmt)                                                     \
  do {                                                                    \
    cudaError_t err = stmt;                                               \
    if (err != cudaSuccess) {                                             \
      gpuTKLog(ERROR, "Failed to run stmt ", #stmt);                         \
      gpuTKLog(ERROR, "Got CUDA error ...  ", cudaGetErrorString(err));      \
      return -1;                                                          \
    }                                                                     \
  } while (0)

#define TILE_WIDTH 16

// Compute C[b] = A[b] * B[b] for all batches in parallel
__global__ void batchedMatMul(float *A, float *B, float *C,
                              int batch, int M, int K, int N) {
  //@@ Implement the batched tiled matrix multiplication kernel.
  //@@ Hint: use blockIdx.z for batch indexing.
  //@@ Hint: use shared memory tiling (TILE_WIDTH x TILE_WIDTH) on A and B.

  // sum += A[b * M * K + i * K + k] * B[b * K * N + k * N + j];
  //matrices A of shape (batch, M, K) and B of shape (batch, K, N)
  __shared__ float tileA[TILE_WIDTH][TILE_WIDTH];
  __shared__ float tileB[TILE_WIDTH][TILE_WIDTH];
  int batchAIdx = blockIdx.z * M * K;
  int batchBIdx = blockIdx.z * K * N;
  int batchCIdx = blockIdx.z * M * N;
  int row = blockIdx.y * TILE_WIDTH + threadIdx.y;
  int col = blockIdx.x * TILE_WIDTH + threadIdx.x;
  float val = 0.0f;
  int numTiles = (K + TILE_WIDTH - 1) / TILE_WIDTH;
  for (int t = 0; t < numTiles; t++) {
    int aCol = t * TILE_WIDTH + threadIdx.x;
    int bRow = t * TILE_WIDTH + threadIdx.y;
    tileA[threadIdx.y][threadIdx.x] =
      (row < M && aCol < K) ? A[batchAIdx + row * K + aCol] : 0.0f;
    tileB[threadIdx.y][threadIdx.x] =
      (bRow < K && col < N) ? B[batchBIdx + bRow * N + col] : 0.0f;
    __syncthreads();
    for (int k = 0; k < TILE_WIDTH; k++)
      val += tileA[threadIdx.y][k] * tileB[k][threadIdx.x];
    __syncthreads();
  }
  if (row < M && col < N)
    C[batchCIdx + row * N + col] = val;
}

int main(int argc, char **argv) {
  gpuTKArg_t args;
  float *hostA, *hostB, *hostC;
  float *hostDims;
  float *deviceA, *deviceB, *deviceC;
  int dimsRows, dimsCols;
  int numARows, numACols;
  int numBRows, numBCols;
  int batch, M, K, N;

  args = gpuTKArg_read(argc, argv);

  gpuTKTime_start(Generic, "Importing data and creating memory on host");
  hostA = (float *)gpuTKImport(gpuTKArg_getInputFile(args, 0), &numARows, &numACols);
  hostB = (float *)gpuTKImport(gpuTKArg_getInputFile(args, 1), &numBRows, &numBCols);
  // input2.raw is a 4x1 column vector. gpuTKImport returns rows=4, cols=1.
  // data[0]=batch, data[1]=M, data[2]=K, data[3]=N. Cast floats to int.
  hostDims = (float *)gpuTKImport(gpuTKArg_getInputFile(args, 2), &dimsRows, &dimsCols);
  //@@ Extract batch, M, K, N from hostDims (cast each float element to int)
  batch = 0; M = 0; K = 0; N = 0;
  batch = (int)hostDims[0];
  M = (int)hostDims[1];
  K = (int)hostDims[2];
  N = (int)hostDims[3];
  //@@ Allocate hostC (batch * M * N floats)
  hostC = (float *)calloc(batch * M * N, sizeof(float));
  gpuTKTime_stop(Generic, "Importing data and creating memory on host");

  gpuTKLog(TRACE, "batch=", batch, " M=", M, " K=", K, " N=", N);

  gpuTKTime_start(GPU, "Allocating GPU memory.");
  //@@ Allocate device memory for deviceA (batch*M*K), deviceB (batch*K*N), deviceC (batch*M*N)
  int sizeA = batch*M*K*sizeof(float);
  int sizeB = batch*K*N*sizeof(float);
  int sizeC = batch*M*N*sizeof(float);

  cudaMalloc((void**)&deviceA, sizeA);
  cudaMalloc((void**)&deviceB, sizeB);
  cudaMalloc((void**)&deviceC, sizeC);

  gpuTKTime_stop(GPU, "Allocating GPU memory.");

  gpuTKTime_start(GPU, "Copying input memory to the GPU.");
  //@@ Copy hostA and hostB to deviceA and deviceB
  cudaMemcpy(deviceA, hostA, sizeA, cudaMemcpyHostToDevice);
  cudaMemcpy(deviceB, hostB, sizeB, cudaMemcpyHostToDevice);

  gpuTKTime_stop(GPU, "Copying input memory to the GPU.");

  //@@ Set grid and block dimensions.
  //@@ Block: (TILE_WIDTH, TILE_WIDTH). Grid: (ceil(N/TILE_WIDTH), ceil(M/TILE_WIDTH), batch).
  dim3 dimBlock(TILE_WIDTH, TILE_WIDTH);
  dim3 dimGrid(ceil(N/TILE_WIDTH), ceil(M/TILE_WIDTH), batch);

  gpuTKTime_start(Compute, "Performing CUDA computation");
  //@@ Launch batchedMatMul kernel
  batchedMatMul<<<dimGrid, dimBlock>>>(
    deviceA,deviceB,deviceC,
    batch,M,K,N
  );

  cudaDeviceSynchronize();
  gpuTKTime_stop(Compute, "Performing CUDA computation");

  gpuTKTime_start(Copy, "Copying output memory to the CPU");
  //@@ Copy deviceC back to hostC
  cudaMemcpy(hostC, deviceC, sizeC, cudaMemcpyDeviceToHost);

  gpuTKTime_stop(Copy, "Copying output memory to the CPU");

  gpuTKTime_start(GPU, "Freeing GPU Memory");
  //@@ Free deviceA, deviceB, deviceC
  cudaFree(deviceA);
  cudaFree(deviceB);
  cudaFree(deviceC);

  gpuTKTime_stop(GPU, "Freeing GPU Memory");

  gpuTKSolution(args, hostC, batch * M, N);

  free(hostA);
  free(hostB);
  free(hostC);
  free(hostDims);

  return 0;
}
