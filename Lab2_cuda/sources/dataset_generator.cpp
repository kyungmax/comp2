#include "gputk.h"

static char *base_dir;

static float *generate_data(int n) {
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

static void create_dataset(int datasetNum, int batch, int M, int K, int N) {
  const char *dir_name =
      gpuTKDirectory_create(gpuTKPath_join(base_dir, datasetNum));

  char *input0_file_name = gpuTKPath_join(dir_name, "input0.raw");
  char *input1_file_name = gpuTKPath_join(dir_name, "input1.raw");
  char *input2_file_name = gpuTKPath_join(dir_name, "input2.raw");
  char *output_file_name = gpuTKPath_join(dir_name, "output.raw");

  float *A = generate_data(batch * M * K);
  float *B = generate_data(batch * K * N);
  float *C = (float *)calloc(batch * M * N, sizeof(float));

  for (int b = 0; b < batch; b++) {
    for (int i = 0; i < M; i++) {
      for (int j = 0; j < N; j++) {
        float sum = 0.0f;
        for (int k = 0; k < K; k++) {
          sum += A[b * M * K + i * K + k] * B[b * K * N + k * N + j];
        }
        C[b * M * N + i * N + j] = sum;
      }
    }
  }

  // input0.raw: (batch*M) x K — stacked A matrices
  write_data(input0_file_name, A, batch * M, K);
  // input1.raw: (batch*K) x N — stacked B matrices
  write_data(input1_file_name, B, batch * K, N);
  // input2.raw: 4x1 column vector [batch, M, K, N]
  // Header "4 1" avoids gpuTKImport's 1-row swap (rows==1 triggers rows=cols, cols=1).
  float dims[4] = {(float)batch, (float)M, (float)K, (float)N};
  write_data(input2_file_name, dims, 4, 1);
  // output.raw: (batch*M) x N — stacked C matrices
  write_data(output_file_name, C, batch * M, N);

  free(A);
  free(B);
  free(C);
}

int main() {
  base_dir =
      gpuTKPath_join(gpuTKDirectory_current(), "BatchedMatMul", "Dataset");

  // Correctness & general timing datasets (varied shapes)
  create_dataset(0, 2,  16,  16,  16);
  create_dataset(1, 2,  32,  32,  32);
  create_dataset(2, 4,  64,  64,  64);
  create_dataset(3, 4,  48,  32,  16);
  create_dataset(4, 4,  64, 128,  64);
  create_dataset(5, 8,  32,  64,  32);
  create_dataset(6, 2, 112,  48,  96);
  create_dataset(7, 4,  80,  99, 128);
  create_dataset(8, 8, 256, 128, 256);
  // Batch-scaling datasets: fixed per-batch (64 x 64) x (64 x 64), varying batch.
  // Use these for the batch-level parallelism experiment in the report (Q7).
  create_dataset(9,   1, 64, 64, 64);
  create_dataset(10,  2, 64, 64, 64);
  create_dataset(11,  4, 64, 64, 64);
  create_dataset(12,  8, 64, 64, 64);
  create_dataset(13, 16, 64, 64, 64);
  create_dataset(14, 32, 64, 64, 64);
  return 0;
}
