add_executable(basic basic/main.cpp basic/sanity_check.cpp)
target_link_libraries(basic cucalc)
target_include_directories(basic PUBLIC ${PROJECT_SOURCE_DIR}/include)

