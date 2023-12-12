#include <stdio.h>

// #define gpuErrchk(val) cudaErrorCheck(val, __FILE__, __LINE__, true)
void cudaErrorCheck(cudaError_t err, const char* message, bool abort) {
  if (err != cudaSuccess) {
    printf("%s:%s\n%s\n", cudaGetErrorName(err), cudaGetErrorString(err), message);
    if (abort) exit(-1);
  }
}