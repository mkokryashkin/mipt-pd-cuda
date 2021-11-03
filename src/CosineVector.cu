#include <CosineVector.cuh>
#include <ScalarMulRunner.cuh>
#include <cmath>

float CosineVector(int numElements, float* vector1, float* vector2) {
  float module1 = sqrt(ScalarMulTwoReductions(numElements, vector1, vector1, 1024));
  float module2 = sqrt(ScalarMulTwoReductions(numElements, vector2, vector2, 1024));
  float scalar = ScalarMulTwoReductions(numElements, vector1, vector2, 1024);
  return scalar / (module1 * module2);
}

