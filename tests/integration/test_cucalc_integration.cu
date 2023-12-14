#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

#include "cucalc/cucalc.h"
#include "cucalc/cucalc_integration.h"

typedef struct {
  struct timeval startTime;
  struct timeval endTime;
} Timer;

void startTime(Timer* timer) { gettimeofday(&(timer->startTime), NULL); }

void stopTime(Timer* timer) { gettimeofday(&(timer->endTime), NULL); }

float elapsedTime(Timer timer) {
  return ((float)((timer.endTime.tv_sec - timer.startTime.tv_sec)
                  + (timer.endTime.tv_usec - timer.startTime.tv_usec) / 1.0e6));
}

__device__ double cubed(double x) { return x * x * x; }

__device__ cucalc_func d_func = cubed;

double hostCubed(double x) { return x * x * x; }

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
  Timer timer;
  double n = 1 << 28;
  double res = 0, a = 0, b = 8;
  double expected = pow(b, 4) / 4 - pow(a, 4) / 4;
  startTime(&timer);
  double result = cucalc_integration_trapez(h_func, a, b, n);
  printf("GPU output : %f\n", result);
  stopTime(&timer);
  printf("GPU time : %f s\n", elapsedTime(timer));

  startTime(&timer);

  double h = (b - a) / n;
  for (size_t i = 0; i < n; i++) {
    double x = (double)i * h + a;
    res += 0.5 * h * (hostCubed(x) + hostCubed(x + h));
  }
  printf("CPU output : %f\n", res);
  stopTime(&timer);
  printf("CPU time : %f s\n", elapsedTime(timer));
  if (expected == (int)result)
    printf("\x1B[32mcucalc_integration_trapez: test passed!\033[0m\n");
  else
    printf(
        "\x1B[31mcucalc_integration_trapez: test failed! \n\texpected : %f, actual : %f\033[0m\n",
        expected, result);
  return 0;
}
