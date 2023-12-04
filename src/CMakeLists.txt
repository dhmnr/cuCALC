set(SOURCE_FILES 
    misc/cucalc_gpuinfo.cu
    integration/cucalc_integration.cu)

# Create a CUDA library
add_library(cucalc SHARED ${SOURCE_FILES})

# Link against CUDA libraries
target_link_libraries(cucalc ${CUDA_LIBRARIES})
