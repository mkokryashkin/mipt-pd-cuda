#include <assert.h>
#include <MatrixVectorMul.cuh>

void FillMatrix(float* mat, int width, int height, float value) {
  for(int row = 0; row < height; ++row) {
    for(int col = 0; col < width; ++col) {
      mat[row * width + col] = value;
    }
  }
}

int main() {
  int width = 10000;
  int height = 10000;

  float *h_A = new float[width * height];
  float *h_vec = new float[width];
  float *h_res = new float[width];

  FillMatrix(h_A, width, height, 1.0f);
  FillMatrix(h_vec, width, 0, 2.0f);

  float *A = NULL;
  float *vec = NULL;
  float *res = NULL;

  cudaMalloc(&A, width * height);
  cudaMalloc(&vec, width);
  cudaMalloc(&res, width);

  cudaMemcpy(A, h_A, width * height, cudaMemcpyHostToDevice);
  cudaMemcpy(vec, h_vec, width, cudaMemcpyHostToDevice);

  dim3 blockSize(256, 256);
  dim3 numBlocks((height + blockSize.x - 1) / blockSize.x, (width + blockSize.y - 1) / blockSize.y);

  MatrixVectorMul<<<numBlocks, blockSize>>>(height, width, A, vec, res);
	cudaDeviceSynchronize();

  cudaMemcpy(h_res, res, width, cudaMemcpyDeviceToHost);

  for (int row = 0; row < width; ++row) {
    assert(h_res[row] = 2.0f * width);
  }

  cudaFree(A);
  cudaFree(vec);

  free(h_A);
  free(h_vec);

  return 0;
}

