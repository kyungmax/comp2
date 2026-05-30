/********************************************************************
* NOTE: Use warp-level synchronization primitive.
********************************************************************/

// Given a list (lst) of length n
// Output its sum = lst[0] + lst[1] + ... + lst[n-1];

#include <gputk.h>

#define BLOCK_SIZE 512 //@@ You can change this

#define gpuTKCheck(stmt)                                                     \
  do {                                                                    \
    cudaError_t err = stmt;                                               \
    if (err != cudaSuccess) {                                             \
      gpuTKLog(ERROR, "Failed to run stmt ", #stmt);                         \
      return -1;                                                          \
    }                                                                     \
  } while (0)

__global__ void total(float *input, float *output, int len) {
  //@@ Load a segment of the input vector into shared memory

  //@@ Traverse the reduction tree

  //@@ Write the computed sum of the block to the output vector at the
  //@@ correct index


  __shared__ float partialSum[2 * BLOCK_SIZE];
  unsigned int t = threadIdx.x;
  unsigned int i = 2* blockIdx.x * blockDim.x + threadIdx.x;
  // load one element
  partialSum[t] = (i < len) ? input[i] : 0.0f;
  if(i+BLOCK_SIZE < len){
    partialSum[t+BLOCK_SIZE] = input[i + BLOCK_SIZE];
  }else{
    partialSum[t+BLOCK_SIZE] = 0.0f;
  }
  
  for (unsigned int stride = BLOCK_SIZE; stride >= 32; stride /= 2) {
    __syncthreads();
    if (t < stride)
     partialSum[t] += partialSum[t + stride];
  }

  __syncthreads();

  unsigned mask = 0xffffffff;
  float totalSum = (t < 32) ? partialSum[t] : 0.0f;  
  totalSum += __shfl_down_sync(mask, totalSum, 16);
  totalSum += __shfl_down_sync(mask, totalSum, 8);
  totalSum += __shfl_down_sync(mask, totalSum, 4);
  totalSum += __shfl_down_sync(mask, totalSum, 2);
  totalSum += __shfl_down_sync(mask, totalSum, 1);

  if(t == 0)
    output[blockIdx.x] = totalSum;
}

int main(int argc, char **argv) {
  int ii;
  gpuTKArg_t args;
  float *hostInput;  // The input 1D list
  float *hostOutput; // The output list
  float *deviceInput;
  float *deviceOutput;
  int numInputElements;  // number of elements in the input list
  int numOutputElements; // number of elements in the output list

  args = gpuTKArg_read(argc, argv);

  gpuTKTime_start(Generic, "Importing data and creating memory on host");
  hostInput =
      (float *)gpuTKImport(gpuTKArg_getInputFile(args, 0), &numInputElements);

  numOutputElements = numInputElements / (BLOCK_SIZE << 1);
  if (numInputElements % (BLOCK_SIZE << 1)) {
    numOutputElements++;
  }
  hostOutput = (float *)malloc(numOutputElements * sizeof(float));

  gpuTKTime_stop(Generic, "Importing data and creating memory on host");

  gpuTKLog(TRACE, "The number of input elements in the input is ",
        numInputElements);
  gpuTKLog(TRACE, "The number of output elements in the input is ",
        numOutputElements);

  gpuTKTime_start(GPU, "Allocating GPU memory.");
  //@@ Allocate GPU memory here
  
  int inputSize = numInputElements * sizeof(float);
  int outputSize = numOutputElements * sizeof(float);
  cudaMalloc((void**)&deviceInput, inputSize);
  cudaMalloc((void**)&deviceOutput, outputSize);

  gpuTKTime_stop(GPU, "Allocating GPU memory.");

  gpuTKTime_start(GPU, "Copying input memory to the GPU.");
  //@@ Copy memory to the GPU here

  cudaMemcpy(deviceInput, hostInput, inputSize, cudaMemcpyHostToDevice);

  gpuTKTime_stop(GPU, "Copying input memory to the GPU.");
  //@@ Initialize the grid and block dimensions here

  dim3 dimBlock(BLOCK_SIZE,1 , 1);
  dim3 dimGrid(numOutputElements, 1, 1);

  gpuTKTime_start(Compute, "Performing CUDA computation");
  //@@ Launch the GPU Kernel here

  total<<<dimGrid, dimBlock>>>(
    deviceInput, deviceOutput, numInputElements
  );

  cudaDeviceSynchronize();
  gpuTKTime_stop(Compute, "Performing CUDA computation");

  gpuTKTime_start(Copy, "Copying output memory to the CPU");
  //@@ Copy the GPU memory back to the CPU here

  cudaMemcpy(hostOutput, deviceOutput, outputSize, cudaMemcpyDeviceToHost);

  gpuTKTime_stop(Copy, "Copying output memory to the CPU");

  /********************************************************************
   * Reduce output vector on the host to hostOutput[0].
   * NOTE: One could also perform the reduction of the output vector
   * recursively and support any size input. For simplicity, we do not
   * require that for this lab.
   ********************************************************************/
  for(int i=1;i<numOutputElements;i++){
      hostOutput[0] += hostOutput[i];
  }

  gpuTKTime_start(GPU, "Freeing GPU Memory");
  //@@ Free the GPU memory here
  
  cudaFree(deviceInput);
  cudaFree(deviceOutput);

  gpuTKTime_stop(GPU, "Freeing GPU Memory");

  gpuTKSolution(args, hostOutput, 1);

  free(hostInput);
  free(hostOutput);

  return 0;
}
