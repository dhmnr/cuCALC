set(SOURCE_FILES 
    misc/cucalc_gpuinfo.cu
    integration/cucalc_integration.cu)

# Create a CUDA library
add_library(cucalc SHARED ${SOURCE_FILES})

set_target_properties(cucalc PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# Link against CUDA libraries
target_link_libraries(cucalc ${CUDA_LIBRARIES})


