#include <iostream>

#include "shared_dynamic.h"

__device__ double square_func(double x) { return x * x; }

__device__ func_t d_square_func = square_func;

int main() {
  func_t h_square_func;

  cudaMemcpyFromSymbol(&h_square_func, d_square_func, sizeof(func_t));

  test(h_square_func, 10.00);
}
