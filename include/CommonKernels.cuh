#pragma once

__global__ void Reduce(int* in_data, int* out_data);
__global__ void KernelMul(int numElements, float* x, float* y, float* result);
// your can write kernels here for your operations
