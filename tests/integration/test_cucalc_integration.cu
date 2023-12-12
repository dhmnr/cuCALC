#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>

#include "cucalc/cucalc.h"
#include "cucalc/cucalc_integration.h"

__device__ double cubed(double x) { return x * x * x; }

__device__ cucalc_func d_func = cubed;

int main(int argc, char const* argv[]) {
  void* h_func;
  cudaError_t cuda_ret
      = cudaMemcpyFromSymbol(&h_func, d_func, sizeof(cucalc_func), 0, cudaMemcpyDeviceToHost);
  if (cuda_ret != cudaSuccess) {
    printf("Unable to copy device function\n");
    printf(cudaGetErrorString(cuda_ret));
    printf("\n");
  }

  double result = cucalc_integration_trapez(h_func, 0, 8, 1 << 18);
  printf("Integral  = %f\n", result);

  return 0;
}
