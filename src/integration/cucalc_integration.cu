#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>

#include <iostream>

#include "cucalc/cucalc.h"
#include "cucalc/cucalc_common.h"
#include "cucalc/cucalc_integration.h"

__global__ void cucalc_integration_generic_kernel(void *func, double h, double *d_fx, double a,
                                                  int method) {
  size_t tid = threadIdx.x + blockIdx.x * blockDim.x;

  double res, x = tid * h + a;
  switch (method) {
    case TRAPEZOIDAL:
      res = 0.5 * h * ((cucalc_func(func))(x) + (cucalc_func(func))(x + h));
      break;
    case SIMPSON_1_3:
      h /= 2.0;
      res = (1.0 / 3) * h
            * ((cucalc_func(func))(x) + 4 * (cucalc_func(func))(x + h)
               + (cucalc_func(func))(x + 2 * h));
      break;
    case SIMPSON_3_8:
      h /= 3.0;
      res = (3.0 / 8) * h
            * ((cucalc_func(func))(x) + 3 * (cucalc_func(func))(x + h)
               + 3 * (cucalc_func(func))(x + 2 * h) + (cucalc_func(func))(x + 3 * h));
      break;
    case BOOLE:
      h /= 4.0;
      res = (2.0 / 45) * h
            * (7 * (cucalc_func(func))(x) + 32 * (cucalc_func(func))(x + h)
               + 12 * (cucalc_func(func))(x + 2 * h) + 32 * (cucalc_func(func))(x + 3 * h)
               + 7 * (cucalc_func(func))(x + 4 * h));
    default:
      break;
  }

  d_fx[tid] = res;
}

double cucalc_integration_generic(void *func, double a, double b, size_t steps, int method) {
  cudaError_t cuda_ret;
  int BLOCK_SIZE = 512;
  size_t thread_count = steps + 1;
  dim3 blockSize(BLOCK_SIZE, 1, 1);
  dim3 gridSize((thread_count - 1) / BLOCK_SIZE + 1, 1, 1);
  double h = (b - a) / steps;

  double *d_fx, *h_fx;
  cuda_ret = cudaMalloc((void **)&d_fx, sizeof(double) * thread_count);
  cudaErrorCheck(cuda_ret, "Unable to allocate memory on GPU", 1);

  cuda_ret = cudaMallocHost((void **)&h_fx, sizeof(double) * thread_count);
  cudaErrorCheck(cuda_ret, "Unable to allocate memory on host", 1);

  cucalc_integration_generic_kernel<<<gridSize, blockSize>>>(func, h, d_fx, a, method);
  cuda_ret = cudaDeviceSynchronize();
  cudaErrorCheck(cuda_ret, "Unable to launch kernel", 1);

  return cucalc_reduction_sum(d_fx, thread_count);
}

double cucalc_integration_trapez(void *func, double a, double b, size_t steps) {
  return cucalc_integration_generic(func, a, b, steps, TRAPEZOIDAL);
}

double cucalc_integration_simpson_1_3(void *func, double a, double b, size_t steps) {
  return cucalc_integration_generic(func, a, b, steps, SIMPSON_1_3);
}

double cucalc_integration_simpson_3_8(void *func, double a, double b, size_t steps) {
  return cucalc_integration_generic(func, a, b, steps, SIMPSON_3_8);
}

double cucalc_integration_boole(void *func, double a, double b, size_t steps) {
  return cucalc_integration_generic(func, a, b, steps, BOOLE);
}