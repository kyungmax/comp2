#include <gputk.h>

#define gpuTKCheck(stmt)                                                     \
  do {                                                                       \
    cudaError_t err = stmt;                                                  \
    if (err != cudaSuccess) {                                                \
      gpuTKLog(ERROR, "Failed to run stmt ", #stmt);                         \
      return -1;                                                             \
    }                                                                        \
  } while (0)

#define Mask_width   5
#define Mask_radius  (Mask_width / 2)
#define TILE_WIDTH   32
#define w            (TILE_WIDTH + Mask_width - 1)
#define clamp(x)     (min(max((x), 0.0f), 1.0f))

__global__ void convolution(const float *inputImage,
                            const float *mask,
                            float *outputImage,
                            int channels,
                            int width,
                            int height,
                            int rowOffset,
                            int currentStreamRows) {
  //@@ INSERT CODE HERE
  //@@ NOTE: Use shared memory for operating convolution.
  __shared__ float tile[w][w];

  //@@ TIP: Basic algorithm for convolution is as below.
  //@@ 1. Compute coordinates by using thread and block index.
  //@@ 2. Optimize code to use shared memory.

  //@@ Check if current output pixel coords is inside image bounds
  for (int c= 0 ; c < channels ; c++){
    // load to tile
    int tid = threadIdx.y * TILE_WIDTH + threadIdx.x;
    for (int i = tid; i < w * w; i += (TILE_WIDTH * TILE_WIDTH)) {
        int tile_row = i / w;
        int tile_col = i % w;

        int global_row = (blockIdx.y * TILE_WIDTH) + tile_row - Mask_radius + rowOffset;
        int global_col = (blockIdx.x * TILE_WIDTH) + tile_col - Mask_radius;

        if (global_row >= 0 && global_row < height && global_col >= 0 && global_col < width) {
            int inIdx = (global_row * width + global_col) * channels + c;
            tile[tile_row][tile_col] = inputImage[inIdx];
        } else {
            tile[tile_row][tile_col] = 0.0f;
        }
    }
    __syncthreads();

    int row_rel = (blockIdx.y * TILE_WIDTH) + threadIdx.y;
    int global_row_o = (blockIdx.y * TILE_WIDTH) + threadIdx.y + rowOffset;
    int global_col_o = (blockIdx.x * TILE_WIDTH) + threadIdx.x;
    if (global_row_o < height && global_col_o < width && row_rel < currentStreamRows) {      float sum = 0.0f;
      for (int i = 0; i < Mask_width; i++) {
        for (int j = 0; j < Mask_width; j++) {
          int maskIdx = i * Mask_width + j; 
          sum += mask[maskIdx] * tile[threadIdx.y + i][threadIdx.x + j];
        }
      }
      //@@ Write clamped result back to the output image
      int outIdx = (global_row_o * width + global_col_o) * channels + c;
      outputImage[outIdx] = clamp(sum);
    }
    __syncthreads();
  }
}

int main(int argc, char *argv[]) {
  gpuTKArg_t arg;
  int maskRows;
  int maskColumns;
  int imageChannels;
  int imageWidth;
  int imageHeight;
  char *inputImageFile;
  char *inputMaskFile;
  gpuTKImage_t inputImage;
  gpuTKImage_t outputImage;
  float *hostInputImageData;
  float *hostOutputImageData;
  float *hostMaskData;
  float *deviceInputImageData;
  float *deviceOutputImageData;
  float *deviceMaskData;

  arg           = gpuTKArg_read(argc, argv);

  inputImageFile = gpuTKArg_getInputFile(arg, 0);
  inputMaskFile  = gpuTKArg_getInputFile(arg, 1);

  inputImage     = gpuTKImport(inputImageFile);
  hostMaskData   = (float *)gpuTKImport(inputMaskFile, &maskRows, &maskColumns);

  assert(maskRows == Mask_width);
  assert(maskColumns == Mask_width);

  imageWidth      = gpuTKImage_getWidth(inputImage);
  imageHeight     = gpuTKImage_getHeight(inputImage);
  imageChannels   = gpuTKImage_getChannels(inputImage);

  outputImage     = gpuTKImage_new(imageWidth, imageHeight, imageChannels);

  hostInputImageData  = gpuTKImage_getData(inputImage);
  hostOutputImageData = gpuTKImage_getData(outputImage);

  gpuTKTime_start(GPU, "Doing GPU Computation (memory + compute)");

  gpuTKTime_start(GPU, "Doing GPU memory allocation");
  //@@ INSERT CODE HERE
  //@@ NOTE: use pinned memory and streams.
  int imageSize = imageChannels * imageWidth * imageHeight * sizeof(float);
  int maskSize = maskRows * maskColumns * sizeof(float);
  cudaMalloc((void**)&deviceInputImageData, imageSize);
  cudaMalloc((void**)&deviceOutputImageData, imageSize);
  cudaMalloc((void**)&deviceMaskData, maskSize);

  int numStreams = 8;
  cudaStream_t streams[numStreams];
  float *pinnedInput;
  float *pinnedOutput;
  for(int i=0;i<numStreams;i++){
    cudaStreamCreate(&streams[i]);
  }
  cudaMallocHost((void**)&pinnedInput, imageSize);
  cudaMallocHost((void**)&pinnedOutput, imageSize);

  gpuTKTime_stop(GPU, "Doing GPU memory allocation");

  gpuTKTime_start(Copy, "Copying data to the GPU");
  //@@ INSERT CODE HERE
  memcpy(pinnedInput, hostInputImageData, imageSize);
  cudaMemcpy(deviceMaskData, hostMaskData, maskSize, cudaMemcpyHostToDevice);

  gpuTKTime_stop(Copy, "Copying data to the GPU");

  gpuTKTime_start(Compute, "Doing the computation on the GPU");
  //@@ INSERT CODE HERE
  // @@ NOTE: use below code to call convolution kernel.

  int rowsPerStream = imageHeight / numStreams;
  for (int i = 0; i < numStreams; i++) {
    int rowOffset = i * rowsPerStream;
    int currentStreamRows = (i == numStreams - 1) ? (imageHeight - rowOffset) : rowsPerStream;

    int currentRowNum = i * rowsPerStream;
    int endRowNum = min(imageHeight, (i+1) * rowsPerStream);
    
    size_t streamSize = (size_t)currentStreamRows * imageWidth * imageChannels * sizeof(float);
    size_t offsetByte = (size_t)rowOffset * imageWidth * imageChannels * sizeof(float);

    size_t streamSizeForH2D = streamSize;
    size_t offsetByteForH2D = offsetByte;

    if(currentRowNum >= Mask_radius){
      offsetByteForH2D -= Mask_radius * imageWidth * imageChannels * sizeof(float);
      streamSizeForH2D += Mask_radius * imageWidth * imageChannels * sizeof(float);
    }
    if(endRowNum + Mask_radius  < imageHeight){
      streamSizeForH2D += Mask_radius * imageWidth * imageChannels * sizeof(float);
    }
    cudaMemcpyAsync(
      deviceInputImageData + (offsetByteForH2D/sizeof(float)), 
      pinnedInput + (offsetByteForH2D/sizeof(float)), streamSizeForH2D, cudaMemcpyHostToDevice, streams[i]);

    dim3 dimBlock(TILE_WIDTH, TILE_WIDTH, 1);
    dim3 dimGrid((imageWidth + TILE_WIDTH - 1) / TILE_WIDTH, 
                 (currentStreamRows + TILE_WIDTH - 1) / TILE_WIDTH, 1);

    convolution<<<dimGrid, dimBlock, 0, streams[i]>>>(
        deviceInputImageData, deviceMaskData, deviceOutputImageData, 
        imageChannels, imageWidth, imageHeight, rowOffset, currentStreamRows);

    cudaMemcpyAsync(pinnedOutput + (offsetByte/sizeof(float)), 
                    deviceOutputImageData + (offsetByte/sizeof(float)), 
                    streamSize, cudaMemcpyDeviceToHost, streams[i]);
  }

  for (int i = 0; i < numStreams; i++) {
    cudaStreamSynchronize(streams[i]);
  }

  gpuTKTime_stop(Compute, "Doing the computation on the GPU");

  gpuTKTime_start(Copy, "Copying data from the GPU");
  //@@ INSERT CODE HERE
  memcpy(hostOutputImageData, pinnedOutput, imageSize);

  gpuTKTime_stop(Copy, "Copying data from the GPU");

  gpuTKTime_stop(GPU, "Doing GPU Computation (memory + compute)");

  gpuTKSolution(arg, outputImage);

  //@@ Insert code here
  for (int i = 0; i < numStreams; i++) {
    cudaStreamDestroy(streams[i]);
  }
  cudaFree(deviceInputImageData);
  cudaFree(deviceOutputImageData);
  cudaFree(deviceMaskData);
  cudaFreeHost(pinnedInput);
  cudaFreeHost(pinnedOutput);
  free(hostMaskData);
  gpuTKImage_delete(outputImage);
  gpuTKImage_delete(inputImage);

  return 0;
}