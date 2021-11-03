#include <ScalarMulRunner.cuh>
#include <ScalarMul.cuh>
#include <CommonKernels.cuh>

__global__ void Reduce(float* in_data, float* out_data) {
    extern __shared__ float shared_data[];

    unsigned int tid = threadIdx.x;
    unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;

    shared_data[tid] = in_data[index];
    __syncthreads();

    for (unsigned int s = 1; s < blockDim.x; s *= 2) {
        if (tid % (2 * s) == 0) {
            shared_data[tid] += shared_data[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        out_data[blockIdx.x] = shared_data[0];
    }
}



float ScalarMulTwoReductions(int numElements, float* vector1, float* vector2, int blockSize) {
  return 0.0f;
}

float ScalarMulSumPlusReduction(int numElements, float* vector1, float* vector2, int blockSize) {
  const int numBlocks = (numElements + blockSize - 1) / blockSize;
  //__shared__ float shared_data[];

  float *vec1_d = NULL;
  float *vec2_d = NULL;
  float *result_d = NULL;
  float *out_d = NULL;

  cudaMalloc(&vec1_d, numElements * sizeof(float));
  cudaMalloc(&vec2_d, numElements * sizeof(float));
  cudaMalloc(&result_d, numBlocks * sizeof(float));
  cudaMalloc(&out_d, sizeof(float));
  cudaMemcpy(vec1_d, vector1, numElements * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(vec2_d, vector2, numElements * sizeof(float), cudaMemcpyHostToDevice);

  const int blockSizeReduce = (numBlocks + blockSize - 1) / blockSize;

  ScalarMulBlock<<<numBlocks, blockSize>>>(numElements, vec1_d, vec2_d, result_d);
  Reduce<<numBlocks, blockSizeReduce, numBlocks * sizeof(float)>>>(result_d, out_d);
  float result = 0;
  cudaMemcpy(&result, out_d, sizeof(float), cudaMemcpyDeviceToHost);

  cudaFree(vec1_d);
  cudaFree(vec2_d);
  cudaFree(result_d);
  cudaFree(out_d);
  return result;
}

