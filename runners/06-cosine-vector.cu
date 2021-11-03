#include <CosineVector.cuh>
#include <stdio.h>
#include <assert.h>

int main() {
  const int vec_size = 1 << 20;
  float *vec1 = new float[vec_size]();
  float *vec2 = new float[vec_size]();

  vec1[0] = 1.0f;
  vec2[0] = 1.0f;
  vec2[1] = 1.0f;

  float result = CosineVector(vec_size, vec1, vec2);
  printf("%f\n", result);
  fflush(stdout);
  assert(result - 0.707f < 0.001f);

  delete[] vec1;
  delete[] vec2;
  return 0;
}

