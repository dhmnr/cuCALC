#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>

#include <iostream>

#include "cucalc/cucalc.h"

__global__ void cucalc_integration_trapez_kernel(cucalc_func func, double h, double *d_fx, double a,
                                                 size_t n) {
  size_t mult, tid = threadIdx.x + blockIdx.x * blockDim.x;

  double x = tid * h + a;
  double res = (*func)(x);

  if (tid == 0 || tid == (n - 1))
    mult = 1;
  else
    mult = 2;
  d_fx[tid] = mult * res;
  printf("kernel!");

  //   size_t stride = (n - 1) / 2 + 1;
}

double cucalc_integration_trapez(cucalc_func func, double a, double b, size_t steps) {
  int BLOCK_SIZE = 512;
  size_t thread_count = steps + 2;
  dim3 blockSize(BLOCK_SIZE, 1, 1);
  dim3 gridSize((thread_count - 1) / BLOCK_SIZE + 1, 1, 1);
  double integral = 0, h = (b - a) / steps;

  double *d_fx, *h_fx;
  cudaMalloc((void **)&d_fx, sizeof(double) * thread_count);  // TODO error handling
  cudaMallocHost((void **)&h_fx, sizeof(double) * thread_count);
  std::cout << gridSize.x << std::endl;
  cucalc_integration_trapez_kernel<<<blockSize, gridSize>>>(func, h, d_fx, a, thread_count);
  cudaError_t cuda_ret = cudaDeviceSynchronize();
  if (cuda_ret != cudaSuccess) {
    printf("Unable to launch/execute kernel\n");
    printf(cudaGetErrorName(cuda_ret));
    printf("\n");
  }
  cudaMemcpy(h_fx, d_fx, sizeof(double) * thread_count, cudaMemcpyDeviceToHost);
  for (size_t i = 0; i < thread_count; i++) {
    integral += h_fx[i];
  }
  return integral;
}