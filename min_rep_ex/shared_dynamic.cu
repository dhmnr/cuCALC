#include <stdio.h>

#include <iostream>

#include "shared_dynamic.h"

__global__ void kernel(func_t op, double *d_x, double *result) {
  *result = (*op)(*d_x);
  printf("Res %f", *result);
}

void test(func_t h_func, double x) {
  double *d_x;
  cudaMalloc(&d_x, sizeof(double));
  cudaMemcpy(d_x, &x, sizeof(double), cudaMemcpyHostToDevice);

  double result;
  double *d_result, *h_result;
  cudaMalloc(&d_result, sizeof(double));
  h_result = &result;

  kernel<<<1, 1>>>(h_func, d_x, d_result);
  cudaDeviceSynchronize();
  cudaMemcpy(h_result, d_result, sizeof(double), cudaMemcpyDeviceToHost);
  std::cout << x << " squared is " << result << std::endl;
}
