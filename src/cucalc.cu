#include <cuda_runtime.h>
#include <cuda.h>
#include <iostream>


__global__ void device_cuda_hello(){
    printf("Hello World from GPU!\n");
}

int cuda_hello() {
    device_cuda_hello<<<1,1>>>(); 
    return 0;
}
