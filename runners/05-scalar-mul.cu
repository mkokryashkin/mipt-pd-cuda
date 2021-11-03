#include <ScalarMulRunner.cuh>
#include <stdio.h>

int main() {
  const int vec_size = 1 << 20;
  float *vec1 = new float[vec_size];
  float *vec2 = new float[vec_size];

  float result = ScalarMulSumPlusReduction(vec_size, vec1, vec2, 1024);
  printf("%f\n", result);

  delete[] vec1;
  delete[] vec2;
  return 0;
}

