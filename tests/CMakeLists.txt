

add_executable(gpuinfo misc/test_gpuinfo.cpp)
target_link_libraries(gpuinfo cucalc)
target_include_directories(gpuinfo PUBLIC ${PROJECT_SOURCE_DIR}/include)


add_executable(test_cucalc_integration integration/test_cucalc_integration.cu)
set_target_properties(test_cucalc_integration PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_link_libraries(test_cucalc_integration cucalc ${CUDA_LIBRARIES})
target_include_directories(test_cucalc_integration PUBLIC ${PROJECT_SOURCE_DIR}/include)

add_executable(cucalc_derivative derivative/test_cucalc_derivative.cu)
set_target_properties(cucalc_derivative PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

target_link_libraries(cucalc_derivative cucalc ${CUDA_LIBRARIES})
target_include_directories(cucalc_derivative PUBLIC ${PROJECT_SOURCE_DIR}/include)