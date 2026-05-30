#include "gputk.h"

static char *base_dir;

static void compute(float *C, float *A, float *B, int M, int K, int N) {
  for (int i = 0; i < M; i++) {
    for (int j = 0; j < N; j++) {
      float sum = 0.0f;
      for (int k = 0; k < K; k++) {
        sum += A[i * K + k] * B[k * N + j];
      }
      C[i * N + j] = sum;
    }
  }
}

static float *generate_data(int rows, int cols) {
  int n = rows * cols;
  float *data = (float *)malloc(sizeof(float) * n);
  for (int i = 0; i < n; i++) {
    data[i] = ((float)(rand() % 20) - 5) / 5.0f;
  }
  return data;
}

static void write_data(char *file_name, float *data, int rows, int cols) {
  FILE *handle = fopen(file_name, "w");
  fprintf(handle, "%d %d", rows, cols);
  for (int i = 0; i < rows; i++) {
    fprintf(handle, "\n");
    for (int j = 0; j < cols; j++) {
      if (j > 0) fprintf(handle, " ");
      fprintf(handle, "%.2f", data[i * cols + j]);
    }
  }
  fflush(handle);
  fclose(handle);
}

static void create_dataset(int datasetNum, int M, int K, int N) {
  const char *dir_name =
      gpuTKDirectory_create(gpuTKPath_join(base_dir, datasetNum));

  char *inputA_file_name = gpuTKPath_join(dir_name, "input0.raw");
  char *inputB_file_name = gpuTKPath_join(dir_name, "input1.raw");
  char *output_file_name = gpuTKPath_join(dir_name, "output.raw");

  float *A = generate_data(M, K);
  float *B = generate_data(K, N);
  float *C = (float *)calloc(sizeof(float), M * N);

  compute(C, A, B, M, K, N);

  write_data(inputA_file_name, A, M, K);
  write_data(inputB_file_name, B, K, N);
  write_data(output_file_name, C, M, N);

  free(A);
  free(B);
  free(C);
}

int main() {
  base_dir = gpuTKPath_join(gpuTKDirectory_current(), "MatMul", "Dataset");

  create_dataset(0, 16, 16, 16);
  create_dataset(1, 32, 32, 32);
  create_dataset(2, 64, 48, 64);
  create_dataset(3, 100, 100, 100);
  create_dataset(4, 128, 64, 128);
  create_dataset(5, 200, 150, 200);
  create_dataset(6, 256, 256, 256);
  create_dataset(7, 512, 256, 512);
  create_dataset(8, 1024, 512, 1024);
  return 0;
}
