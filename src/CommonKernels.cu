#include <CommonKernels.cuh>

__global__ void Reduce(int* in_data, int* out_data) {
    extern __shared__ int shared_data[];

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

__global__ void KernelMul(int numElements, float* x, float* y, float* result) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;

	for (int i = index; i < numElements; i += stride) {
  result[i] = x[i] * y[i];
}
}

