cmake_minimum_required(VERSION 3.1)

set(CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc)

project(cuda_streams
        VERSION 1.0
        DESCRIPTION "Learn CMake"
	LANGUAGES CUDA CXX)


enable_language(CUDA)

find_package(CUDA REQUIRED)
include_directories(${CUDA_INCLUDE_DIRS})

add_library(buffer_ops SHARED src/buffer_ops.cu)
set_target_properties(buffer_ops PROPERTIES LINKER_LANGUAGE CXX)
add_executable(cuda_streams src/cuda_streams.cpp)
target_link_libraries(cuda_streams buffer_ops)
target_link_libraries(cuda_streams ${CUDA_LIBRARIES})
