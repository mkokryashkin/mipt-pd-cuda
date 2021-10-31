#pragma once


__global__ void KernelMatrixAdd(int height, int width, size_t pA, size_t pB, size_t pC, float* A, float* B, float* result);
