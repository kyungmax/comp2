#include "gputk.h"

#define Mask_width  5
#define Mask_radius (Mask_width / 2)
#define Channels    3

static char *base_dir;

static inline float clamp01(float x) {
  if (x < 0.0f) return 0.0f;
  if (x > 1.0f) return 1.0f;
  return x;
}

static void cpu_convolution(const float *in, const float *mask, float *out,
                            int width, int height, int channels) {
  for (int row = 0; row < height; row++) {
    for (int col = 0; col < width; col++) {
      for (int c = 0; c < channels; c++) {
        float sum = 0.0f;
        for (int i = 0; i < Mask_width; i++) {
          for (int j = 0; j < Mask_width; j++) {
            int in_r = row + i - Mask_radius;
            int in_c = col + j - Mask_radius;
            if (in_r >= 0 && in_r < height && in_c >= 0 && in_c < width) {
              int inIdx   = (in_r * width + in_c) * channels + c;
              int maskIdx = i * Mask_width + j;
              sum += mask[maskIdx] * in[inIdx];
            }
          }
        }
        int outIdx  = (row * width + col) * channels + c;
        out[outIdx] = clamp01(sum);
      }
    }
  }
}

static gpuTKImage_t generate_image(int width, int height) {
  gpuTKImage_t img = gpuTKImage_new(width, height, Channels);
  float *data      = gpuTKImage_getData(img);
  int n            = width * height * Channels;
  for (int i = 0; i < n; i++) {
    data[i] = (float)(rand() % 256) / 255.0f;
  }
  return img;
}

static float *generate_mask() {
  float *mask = (float *)malloc(sizeof(float) * Mask_width * Mask_width);
  float total = 0.0f;
  for (int i = 0; i < Mask_width * Mask_width; i++) {
    mask[i] = (float)(rand() % 100) / 100.0f;
    total += mask[i];
  }
  // Normalize so the sum is < 1 to avoid saturation/clamp.
  float scale = 0.9f / (total > 1e-6f ? total : 1.0f);
  for (int i = 0; i < Mask_width * Mask_width; i++) {
    mask[i] *= scale;
  }
  return mask;
}

static void create_dataset(int datasetNum, int width, int height) {
  srand(datasetNum + 1);

  const char *dir_name =
      gpuTKDirectory_create(gpuTKPath_join(base_dir, datasetNum));

  char *inputImage_file_name = gpuTKPath_join(dir_name, "input0.ppm");
  char *inputMask_file_name  = gpuTKPath_join(dir_name, "input1.raw");
  char *output_file_name     = gpuTKPath_join(dir_name, "output.ppm");

  gpuTKImage_t inputImage  = generate_image(width, height);
  float *mask              = generate_mask();
  gpuTKImage_t outputImage = gpuTKImage_new(width, height, Channels);

  cpu_convolution(gpuTKImage_getData(inputImage), mask,
                  gpuTKImage_getData(outputImage), width, height, Channels);

  gpuTKExport(inputImage_file_name, inputImage);
  gpuTKExport(inputMask_file_name, mask, Mask_width, Mask_width);
  gpuTKExport(output_file_name, outputImage);

  gpuTKImage_delete(inputImage);
  gpuTKImage_delete(outputImage);
  free(mask);
}

int main() {
  base_dir = gpuTKPath_join(gpuTKDirectory_current(), "Convolution", "Dataset");

  create_dataset(0, 64, 64);
  create_dataset(1, 128, 128);
  create_dataset(2, 256, 256);
  create_dataset(3, 512, 512);
  create_dataset(4, 640, 480);
  create_dataset(5, 800, 600);
  create_dataset(6, 1024, 768);
  return 0;
}
