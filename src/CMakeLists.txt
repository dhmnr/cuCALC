set(SOURCE_FILES misc/gpuinfo.cu)

# Create a CUDA library
add_library(cucalc SHARED ${SOURCE_FILES})

# Link against CUDA libraries
target_link_libraries(cucalc ${CUDA_LIBRARIES})
