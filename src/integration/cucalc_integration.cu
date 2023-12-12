#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>
#include "cucalc/cucalc_common.h"
#include <iostream>

#include "cucalc/cucalc.h"
#define gpuErrchk(val) cudaErrorCheck(val, __FILE__, __LINE__, true)
void cudaErrorCheck(cudaError_t err, char *file, int line, bool abort) {
  if (err != cudaSuccess) {
    printf("%s %s %d\n", cudaGetErrorString(err), file, line);
    if (abort) exit(-1);
  }
}

__device__ double cubed(double x) { return x * x * x; }

__device__ cucalc_func d_func = cubed;

__global__ void cucalc_integration_trapez_kernel(cucalc_func func, double h, double *d_fx, double a,
                                                 size_t n) {
  size_t tid = threadIdx.x + blockIdx.x * blockDim.x;

  double mult, x = tid * h + a;
  double res = (func)(x);

  if (tid == 0 || tid == (n - 1))
    mult = 1 / 2;
  else
    mult = 1;
  d_fx[tid] = mult * res;
}

double cucalc_integration_trapez(double a, double b, size_t steps) {
  int BLOCK_SIZE = 512;
  size_t thread_count = steps + 2;
  dim3 blockSize(BLOCK_SIZE, 1, 1);
  dim3 gridSize((thread_count - 1) / BLOCK_SIZE + 1, 1, 1);
  double integral = 0, h = (b - a) / steps;

  double *d_fx, *h_fx;
  cudaMalloc((void **)&d_fx, sizeof(double) * thread_count);  // TODO error handling
  cudaMallocHost((void **)&h_fx, sizeof(double) * thread_count);
  cucalc_func h_func;

  gpuErrchk(cudaMemcpyFromSymbol(&h_func, d_func, sizeof(cucalc_func), 0, cudaMemcpyDeviceToHost));
  printf("h_func Addr : %x\n", &h_func);
  cucalc_func h_func2 = h_func;
  printf("f_func2 Addr : %x\n", &h_func2);

  cucalc_integration_trapez_kernel<<<gridSize, blockSize>>>(h_func2, h, d_fx, a, thread_count);
  cudaError_t cuda_ret = cudaDeviceSynchronize();
  if (cuda_ret != cudaSuccess) {
    printf("Unable to launch/execute kernel\n");
    printf(cudaGetErrorString(cuda_ret));
    printf("\n");
  }
  return h * cucalc_reduction_sum(d_fx, thread_count);
}