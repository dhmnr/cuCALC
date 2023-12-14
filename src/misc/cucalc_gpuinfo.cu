#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>

#include <iostream>

inline int _ConvertSMVer2Cores(int major, int minor) {
  // Defines for GPU Architecture types (using the SM version to determine
  // the # of cores per SM
  typedef struct {
    int SM;  // 0xMm (hexidecimal notation), M = SM Major version,
    // and m = SM minor version
    int Cores;
  } sSMtoCores;

  sSMtoCores nGpuArchCoresPerSM[]
      = {{0x30, 192}, {0x32, 192}, {0x35, 192}, {0x37, 192}, {0x50, 128}, {0x52, 128}, {0x53, 128},
         {0x60, 64},  {0x61, 128}, {0x62, 128}, {0x70, 64},  {0x72, 64},  {0x75, 64},  {0x80, 64},
         {0x86, 128}, {0x87, 128}, {0x89, 128}, {0x90, 128}, {-1, -1}};

  int index = 0;

  while (nGpuArchCoresPerSM[index].SM != -1) {
    if (nGpuArchCoresPerSM[index].SM == ((major << 4) + minor)) {
      return nGpuArchCoresPerSM[index].Cores;
    }

    index++;
  }

  // If we don't find the values, we default use the previous one
  // to run properly
  printf(
      "MapSMtoCores for SM %d.%d is undefined."
      "  Default to use %d Cores/SM\n",
      major, minor, nGpuArchCoresPerSM[index - 1].Cores);
  return nGpuArchCoresPerSM[index - 1].Cores;
}

int cucalc_gpuinfo() {
  // Check for the presence of CUDA-enabled devices
  int deviceCount;
  cudaGetDeviceCount(&deviceCount);

  if (deviceCount == 0) {
    std::cerr << "No CUDA-enabled devices found." << std::endl;
    return 1;
  }

  std::cout << "Found " << deviceCount << " CUDA-enabled device(s)." << std::endl;

  // Print information about each CUDA device
  for (int i = 0; i < deviceCount; ++i) {
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, i);

    std::cout << "Device " << i << ": " << deviceProp.name << std::endl;
    std::cout << "  Compute Capability: " << deviceProp.major << "." << deviceProp.minor
              << std::endl;
    std::cout << "  Total Global Memory: " << deviceProp.totalGlobalMem / (1024 * 1024) << " MB"
              << std::endl;
    std::cout << "  Multiprocessors: " << deviceProp.multiProcessorCount << std::endl;
    std::cout << "  CUDA Cores: "
              << deviceProp.multiProcessorCount
                     * _ConvertSMVer2Cores(deviceProp.major, deviceProp.minor)
              << std::endl;
    std::cout << "  Warp Size: " << deviceProp.warpSize << std::endl;
    std::cout << "  Shared Memory per Block: " << deviceProp.sharedMemPerBlock / 1024 << " KB"
              << std::endl;
    std::cout << "  Max Threads per Block: " << deviceProp.maxThreadsPerBlock << std::endl;
    std::cout << "  Max Blocks per Multiprocessor: " << deviceProp.maxBlocksPerMultiProcessor
              << std::endl;
    std::cout << "  Max Threads per Multiprocessor: " << deviceProp.maxThreadsPerMultiProcessor
              << std::endl;
    std::cout << "  Max Grid Size: (" << deviceProp.maxGridSize[0] << ", "
              << deviceProp.maxGridSize[1] << ", " << deviceProp.maxGridSize[2] << ")" << std::endl;
    std::cout << std::endl;
  }

  return 0;
}
