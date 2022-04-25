cmake_minimum_required(VERSION 3.15)

project(simd_avx 
	VERSION 1.0
	DESCRIPTION "Learn CMake")

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=broadwell")

add_executable(simd_avx src/simd_avx.cpp)



