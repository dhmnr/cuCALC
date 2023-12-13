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

  double *result = cucalc_derivative_backward(h_func, 0, 2, 1000);
  double actual_value = 1.997999999999999776179038235568441450595855712890625000000000000;
  if (actual_value == result[500])
    printf("Backward Difference Test passed!\n");
  else
    printf("Backward Difference Test failed! expected : %.64f, actual : %.64f\n", 1.997999999999999776179038235568441450595855712890625000000000000, result[500]);
  
  actual_value = 2.0019999999999482653834093071054667234420776367187500000000000000;
  result = cucalc_derivative_forward(h_func, 0, 2, 1000);
  if (actual_value == result[500])
    printf("Forward Difference Test passed!\n");
  else
    printf("Forward Difference Test failed! expected : %.64f, actual : %.64f\n", 2.002000, result[500]);
  
  actual_value = 1.997999999999999776179038235568441450595855712890625000000000000;
  result = cucalc_derivative_central(h_func, 0, 2, 1000);

  if (actual_value == result[500])
    printf("Central Difference Test passed!\n");
  else
    printf("Central Difference Test failed! expected : %.64f, actual : %.64f\n", 1.998000, result[500]);
  return 0;
}