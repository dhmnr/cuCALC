
add_executable(gpuinfo misc/test_gpuinfo.cpp)
target_link_libraries(gpuinfo cucalc)
target_include_directories(gpuinfo PUBLIC ${PROJECT_SOURCE_DIR}/include)


add_executable(cucalc_integration integration/test_cucalc_integration.cu)
target_link_libraries(cucalc_integration cucalc ${CUDA_LIBRARIES})
target_include_directories(cucalc_integration PUBLIC ${PROJECT_SOURCE_DIR}/include)


