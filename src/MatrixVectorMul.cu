#include <MatrixVectorMul.cuh>

__global__
void MatrixVectorMul(int height, int width, float* matrix, float* vector, float* result) {
  int row = blockIdx.x * blockDim.x + threadIdx.x;
  int col = blockIdx.y * blockDim.y + threadIdx.y;
	int stride_row = blockDim.x * gridDim.x;
	int stride_col = blockDim.y * gridDim.y;

  for (int i = row; i < height; i += stride_row) {
    float* rowA = matrix + row * width;
    for(int j = col; j < width; j += stride_col) {
       atomicAdd(&result[j], rowA[j] * vector[j]);
    }
	}
}

