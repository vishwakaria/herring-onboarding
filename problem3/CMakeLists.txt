cmake_minimum_required(VERSION 3.1)

project(cpu_gpu
        VERSION 1.0
        DESCRIPTION "Learn CMake")

find_package(CUDA REQUIRED)
include_directories(${CUDA_INCLUDE_DIRS})

add_executable(cpu_gpu src/cpu_gpu.cpp)
target_link_libraries(cpu_gpu ${CUDA_LIBRARIES})

