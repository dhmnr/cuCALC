#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <iostream>

#include "cucalc/cucalc.h"
#include "cucalc/cucalc_integration.h"

__device__ double squared(double x) { return x*x*x + 1; }

__device__ cucalc_func p_add_func = squared;

#define gpuErrchk(val) \
    cudaErrorCheck(val, __FILE__, __LINE__, true)
void cudaErrorCheck(cudaError_t err, char* file, int line, bool abort)
{
    
    printf("ERRORCHECK");
    if(err != cudaSuccess)
    {
      printf("%s %s %d\n", cudaGetErrorString(err), file, line);
      if(abort) exit(-1);
    }
}

int main(int argc, char const *argv[]) {
  
  cucalc_func h_add_func;
  gpuErrchk(cudaMemcpyFromSymbol(&h_add_func, p_add_func, sizeof(cucalc_func)));
  cucalc_func d_func = h_add_func;
  double result = cucalc_integration_trapez(&h_add_func, 0, 2, 510);
  printf("Integral %f\n", result);

  return 0;
}
