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

  float result = ScalarMulSumPlusReduction(vec_size, vec1, vec2, 1024);
  printf("%f\n", result);
  assert(result == 2.0f * (1 << 20));

  delete[] vec1;
  delete[] vec2;
  return 0;
}

