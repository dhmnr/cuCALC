#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>

#include <iostream>

#include "cucalc/cucalc.h"
#define BLOCK_SIZE 512
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

__global__ void reduction_sum(double *out, double *d_fx, int size){
  __shared__ double partialSum[2*BLOCK_SIZE];
    unsigned int t = threadIdx.x;
    unsigned int start = 2*blockIdx.x*blockDim.x;
    partialSum[t] = d_fx[start + t];
    partialSum[blockDim.x+t] = d_fx[start + blockDim.x+t];
    unsigned int offset;
       if (blockIdx.x == size / (BLOCK_SIZE<<1)) {
        offset = (size % (BLOCK_SIZE<<1));
        if (blockDim.x+t < offset) {
            partialSum[t] = d_fx[start + t];
            partialSum[blockDim.x+t] = d_fx[start + blockDim.x+t];
        }
        else if (t < offset) {
            partialSum[t] = d_fx[start + t];
            partialSum[blockDim.x+t] = 0;
        }
        else {
            partialSum[t] = 0;
            partialSum[blockDim.x+t] = 0;
        }
    }
    else {
        partialSum[t] = d_fx[start + t];
        partialSum[blockDim.x+t] = d_fx[start + blockDim.x+t];
    }
    for (unsigned int stride = BLOCK_SIZE;stride >=1; stride =stride/2)
    {
        __syncthreads();
        if (t <stride)
        partialSum[t]+= partialSum[t+stride];
    }
    if (threadIdx.x == 0){

        out[blockIdx.x] = partialSum[0];
    }

}
double cucalc_integration_trapez(double a, double b, size_t steps) {
  //int BLOCK_SIZE = 512;
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

  cucalc_integration_trapez_kernel<<<blockSize, gridSize>>>(h_func2, h, d_fx, a, thread_count);
  cudaError_t cuda_ret = cudaDeviceSynchronize();
  if (cuda_ret != cudaSuccess) {
    printf("Unable to launch/execute kernel\n");
    printf(cudaGetErrorString(cuda_ret));
    printf("\n");
  }

  //Reduction Sum
  unsigned int out_elements;
  double *out_h, *out_d;
  // out_elements = in_elements / (BLOCK_SIZE<<1);
  // if(in_elements % (BLOCK_SIZE<<1)) out_elements++;

  out_elements = (thread_count - 1) / BLOCK_SIZE*2 + 1;
  out_h = (double*)malloc(out_elements * sizeof(double));

  cudaMalloc((void**)&out_d, out_elements * sizeof(double));
  //if(cuda_ret != cudaSuccess) FATAL("Unable to allocate device memory");

  //if(out_h == NULL) FATAL("Unable to allocate host");
  dim3 gridSizeReduction((thread_count - 1) / BLOCK_SIZE*2 + 1, 1, 1);
  reduction_sum<<<blockSize, gridSizeReduction>>>(out_d, d_fx, thread_count);
  cudaMemcpy(out_h, out_d, out_elements * sizeof(double),cudaMemcpyDeviceToHost);

  cudaMemcpy(h_fx, out_d, sizeof(double) * out_elements, cudaMemcpyDeviceToHost);
  for (size_t i = 0; i < out_elements; i++) {
    integral += h_fx[i];
  }
  return h * integral;
}