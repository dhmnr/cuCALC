add_executable(gpuinfo misc/gpuinfo.cpp)
target_link_libraries(gpuinfo cucalc)
target_include_directories(gpuinfo PUBLIC ${PROJECT_SOURCE_DIR}/include)

