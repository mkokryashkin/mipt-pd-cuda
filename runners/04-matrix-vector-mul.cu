#include <assert.h>
#include <stdio.h>
#include <MatrixVectorMul.cuh>

const float TOLERANCE = 0.001f;

void FillMatrix(float* mat, int width, int height, float value) {
  for(int row = 0; row < height; ++row) {
    for(int col = 0; col < width; ++col) {
      mat[row * width + col] = value;
    }
  }
}

int main() {
  int width = 1000;
  int height = 1000;

  float *h_A = new float[width * height];
  float *h_vec = new float[width];
  float *h_res = new float[width];

  FillMatrix(h_A, width, height, 2.0f);

  for (int row = 0; row < width; ++row) {
    h_vec[row] = row + 1;
  }

  float *A = NULL;
  float *vec = NULL;
  float *res = NULL;

  cudaMalloc(&A, width * height * sizeof(float));
  cudaMalloc(&vec, width * sizeof(float));
  cudaMalloc(&res, width * sizeof(float));

  cudaMemcpy(A, h_A, width * height * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(vec, h_vec, width * sizeof(float), cudaMemcpyHostToDevice);

  dim3 blockSize(32, 32);
  dim3 numBlocks((height + blockSize.x - 1) / blockSize.x, (width + blockSize.y - 1) / blockSize.y);

  MatrixVectorMul<<<numBlocks, blockSize>>>(height, width, A, vec, res);
	cudaDeviceSynchronize();

  cudaMemcpy(h_res, res, width * sizeof(float), cudaMemcpyDeviceToHost);

  for (int row = 0; row < width; ++row) {
    assert(h_res[row] - 1001000.0f < TOLERANCE);
  }

  cudaFree(A);
  cudaFree(vec);

  free(h_A);
  free(h_vec);

  return 0;
}

