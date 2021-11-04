#include <ScalarMulRunner.cuh>
#include <stdio.h>
#include <assert.h>

int main() {
  const int vec_size = 1 << 20;
  float *vec1 = new float[vec_size];
  float *vec2 = new float[vec_size];

  for(int i = 0; i < vec_size; ++i) {
    vec1[i] = 1.0f;
    vec2[i] = 2.0f;
  }

  cudaEvent_t start;
  cudaEvent_t stop;

  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  cudaEventRecord(start);
  float result = ScalarMulSumPlusReduction(vec_size, vec1, vec2, 1024);
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);

  float millis = 0;
  cudaEventElapsedTime(&millis, start, stop);
  printf("Elpased: %f\n", millis);

  printf("%f\n", result);
  fflush(stdout);
  assert(result == 2.0f * (1 << 20));

  result = 0;
  cudaEventRecord(start);
  result = ScalarMulTwoReductions(vec_size, vec1, vec2, 1024);
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&millis, start, stop);
  printf("Elpased: %f\n", millis);
  printf("%f\n", result);
  fflush(stdout);
  assert(result == 2.0f * (1 << 20));

  delete[] vec1;
  delete[] vec2;
  return 0;
}

