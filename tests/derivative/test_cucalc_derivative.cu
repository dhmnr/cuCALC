#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>

#include "cucalc/cucalc.h"
#include "cucalc/cucalc_derivative.h"

__device__ double squared(double x) { return x * x; }

__device__ cucalc_func d_func = squared;

int main(int argc, char const* argv[]) {
  cudaSetDevice(3);
  void* h_func;
  cudaError_t cuda_ret
      = cudaMemcpyFromSymbol(&h_func, d_func, sizeof(cucalc_func), 0, cudaMemcpyDeviceToHost);
  if (cuda_ret != cudaSuccess) {
    printf("Unable to copy device function\n");
    printf(cudaGetErrorString(cuda_ret));
    printf("\n");
  }

  double *result_backward = cucalc_derivative_backward(h_func, 0, 2, 1000);
  printf("Backward %f\n",result_backward[500]);
  
  double *result_forward = cucalc_derivative_forward(h_func, 0, 2, 1000);
  printf("Forward %f\n",result_forward[500]);
  
  double *result_central = cucalc_derivative_central(h_func, 0, 2, 1000);
  printf("Central %f\n",result_central[500]);
  // if (1024 == (int)result)
  //   printf("Test passed!\n");
  // else
  //   printf("Test failed! expected : %d, actual : %d\n", 1024, (int)result);
  // return 0;
}