#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>

#include <iostream>

#include "cucalc/cucalc.h"
#include "cucalc/cucalc_common.h"
#include "cucalc/cucalc_integration.h"

__global__ void cucalc_derivative_backward(void *func, double h, double *d_fx,double *h_fx, double a, size_t thread_count) {

  size_t tid = threadIdx.x + blockIdx.x * blockDim.x;
  double x = tid*h + a; 
  double result = (cucalc_func(func))(x);
  //__syncthreads();
  d_fx[tid] = result;
  if(tid != 0){
    h_fx[tid-1] = (d_fx[tid-1] - d_fx[tid])/h;
  } 
}
__global__ void cucalc_derivative_forward(void *func, double h, double *d_fx, double a,
                                                 size_t n) {
  size_t tid = threadIdx.x + blockIdx.x * blockDim.x;

  double mult, x = tid * h + a;
  double res = (cucalc_func(func))(x);

  if (tid == 0 || tid == (n - 1))
    mult = 1 / 2;
  else
    mult = 1;
  d_fx[tid] = mult * res;
}
__global__ void cucalc_derivative_central(void *func, double h, double *d_fx, double a,
                                                 size_t n) {
  size_t tid = threadIdx.x + blockIdx.x * blockDim.x;

  double mult, x = tid * h + a;
  double res = (cucalc_func(func))(x);

  if (tid == 0 || tid == (n - 1))
    mult = 1 / 2;
  else
    mult = 1;
  d_fx[tid] = mult * res;
}

double* cucalc_derivative_backward(void *func, double a, double b, size_t steps) {
  cudaError_t cuda_ret;
  int BLOCK_SIZE = 512;
  size_t thread_count = steps + 2;
  dim3 blockSize(BLOCK_SIZE, 1, 1);
  dim3 gridSize((thread_count - 1) / BLOCK_SIZE + 1, 1, 1);
  double h = (b - a) / steps;
  a = a - h; //for 1 backward

  double *d_fx, *h_fx;
  cuda_ret = cudaMalloc((void **)&d_fx, sizeof(double) * thread_count);
  cudaErrorCheck(cuda_ret, "Unable to allocate memory on GPU", 1);

  cuda_ret = cudaMallocHost((void **)&h_fx, sizeof(double) * thread_count);
  cudaErrorCheck(cuda_ret, "Unable to allocate memory on host", 1);

  cucalc_derivative_backward<<<gridSize, blockSize>>>(func, h, d_fx, h_fx, a, thread_count + 1);

  cuda_ret = cudaMemcpy(h_fx, d_fx, thread_count * sizeof(double), cudaMemcpyDeviceToHost);
  cudaErrorCheck(cuda_ret, "Unable to memory to host", 1);

  cuda_ret = cudaDeviceSynchronize();
  cudaErrorCheck(cuda_ret, "Unable to launch kernel", 1);

  return h_fx;
}

double cucalc_derivative_forward(void *func, double a, double b, size_t steps) {
  cudaError_t cuda_ret;
  int BLOCK_SIZE = 512;
  size_t thread_count = steps + 2;
  dim3 blockSize(BLOCK_SIZE, 1, 1);
  dim3 gridSize((thread_count - 1) / BLOCK_SIZE + 1, 1, 1);
  double h = (b - a) / steps;

  double *d_fx, *h_fx;
  cuda_ret = cudaMalloc((void **)&d_fx, sizeof(double) * thread_count);  // TODO error handling
  cudaErrorCheck(cuda_ret, "Unable to allocate memory on GPU", 1);

  cuda_ret = cudaMallocHost((void **)&h_fx, sizeof(double) * thread_count);
  cudaErrorCheck(cuda_ret, "Unable to allocate memory on host", 1);

  cucalc_derivative_forward<<<gridSize, blockSize>>>(func, h, d_fx, a, thread_count);
  cuda_ret = cudaDeviceSynchronize();
  cudaErrorCheck(cuda_ret, "Unable to launch kernel", 1);

  return h * cucalc_reduction_sum(d_fx, thread_count);
}

double cucalc_derivative_central(void *func, double a, double b, size_t steps) {
  cudaError_t cuda_ret;
  int BLOCK_SIZE = 512;
  size_t thread_count = steps + 2;
  dim3 blockSize(BLOCK_SIZE, 1, 1);
  dim3 gridSize((thread_count - 1) / BLOCK_SIZE + 1, 1, 1);
  double h = (b - a) / steps;

  double *d_fx, *h_fx;
  cuda_ret = cudaMalloc((void **)&d_fx, sizeof(double) * thread_count);  // TODO error handling
  cudaErrorCheck(cuda_ret, "Unable to allocate memory on GPU", 1);

  cuda_ret = cudaMallocHost((void **)&h_fx, sizeof(double) * thread_count);
  cudaErrorCheck(cuda_ret, "Unable to allocate memory on host", 1);

  cucalc_derivative_backward<<<gridSize, blockSize>>>(func, h, d_fx, a, thread_count);
  cuda_ret = cudaDeviceSynchronize();
  cudaErrorCheck(cuda_ret, "Unable to launch kernel", 1);

  return h * cucalc_reduction_sum(d_fx, thread_count);
}