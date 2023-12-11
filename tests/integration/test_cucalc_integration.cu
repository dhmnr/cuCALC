#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <iostream>

#include "cucalc/cucalc.h"
#include "cucalc/cucalc_integration.h"

#define gpuErrchk(val) cudaErrorCheck(val, __FILE__, __LINE__, true)
void cudaErrorCheck(cudaError_t err, char* file, int line, bool abort) {
  if (err != cudaSuccess) {
    printf("%s %s %d\n", cudaGetErrorString(err), file, line);
    if (abort) exit(-1);
  }
}

__device__ double cubed(double x) { return x * x * x; }

__device__ cucalc_func d_func = cubed;

int main(int argc, char const* argv[]) {
  // cucalc_func h_func;

  // gpuErrchk(cudaMemcpyFromSymbol(&h_func, d_func, sizeof(cucalc_func), 0,
  //                                cudaMemcpyDeviceToHost));
  // printf("Func Addr : %x\n", &d_func);
  double result = cucalc_integration_trapez(0, 8, 1 << 18);
  printf("Integral  = %f\n", result);

  return 0;
}
