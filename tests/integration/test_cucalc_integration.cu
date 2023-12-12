#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>
#include <stdlib.h>

#include "cucalc/cucalc.h"
#include "cucalc/cucalc_integration.h"

__device__ double cubed(double x) { return x * x * x; }

__device__ cucalc_func d_func = cubed;

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

  double a = 0, b = 8;
  double expected = pow(b, 4) / 4 - pow(a, 4) / 4;
  double result = cucalc_integration_trapez(h_func, a, b, 1 << 18);

  if (expected == result)
    printf("\x1B[32mcucalc_integration_trapez: test passed!\033[0m\n");
  else
    printf(
        "\x1B[31mcucalc_integration_trapez: test failed! \n\texpected : %f, actual : %f\033[0m\n",
        expected, result);
  return 0;
}
