#define BLOCK_SIZE 512
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>
#include <iostream>

#include "cucalc/cucalc.h"

__global__ void reduction_sum(double *array, size_t array_length) {
  __shared__ double partialSum[2 * BLOCK_SIZE];
  unsigned int t = threadIdx.x;
  unsigned int start = 2 * blockIdx.x * blockDim.x;
  partialSum[t] = array[start + t];
  partialSum[blockDim.x + t] = array[start + blockDim.x + t];
  unsigned int offset;
  if (blockIdx.x == array_length / (BLOCK_SIZE << 1)) {
    offset = (array_length % (BLOCK_SIZE << 1));
    if (blockDim.x + t < offset) {
      partialSum[t] = array[start + t];
      partialSum[blockDim.x + t] = array[start + blockDim.x + t];
    } else if (t < offset) {
      partialSum[t] = array[start + t];
      partialSum[blockDim.x + t] = 0;
    } else {
      partialSum[t] = 0;
      partialSum[blockDim.x + t] = 0;
    }
  } else {
    partialSum[t] = array[start + t];
    partialSum[blockDim.x + t] = array[start + blockDim.x + t];
  }
  for (unsigned int stride = BLOCK_SIZE; stride >= 1; stride = stride / 2) {
    __syncthreads();
    if (t < stride) partialSum[t] += partialSum[t + stride];
  }
  if (threadIdx.x == 0) {
    array[blockIdx.x] = partialSum[0];
  }
}
/*
The following Reduction Sum function is used to calculate the sum provided the array and size of array.
The reduction kernel is launched repeatedly until it reduces to single block where the final sum is calculated. 
*/
double cucalc_reduction_sum(double *array, size_t array_length)
{
// Reduction Sum:-
  size_t block_count;
  double reduction_sum_output;

  for(size_t element_count = array_length; element_count>2; element_count = block_count)
  {
    block_count = (element_count - 1) / (BLOCK_SIZE * 2) + 1;
    dim3 blockSize(BLOCK_SIZE, 1, 1);
    dim3 gridSize(block_count, 1, 1);
    reduction_sum<<<gridSize, blockSize>>>(array, element_count);
    cudaError_t cuda_ret = cudaDeviceSynchronize();
    if (cuda_ret != cudaSuccess) {
      printf("Unable to launch/execute kernel\n");
      printf(cudaGetErrorString(cuda_ret));
      printf("\n");
  }
  
  }
  cudaMemcpy(&reduction_sum_output, array, sizeof(double), cudaMemcpyDeviceToHost);

  return reduction_sum_output;
}