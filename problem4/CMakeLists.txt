execute_process (
    COMMAND bash -c "ln -s /usr/local/cuda/lib64/libnvrtc.so.*.*.* /usr/local/cuda/lib64/libnvrtc.so"
)

cmake_minimum_required(VERSION 3.10)
     
project(mpi_gpu
        VERSION 1.0
        DESCRIPTION "Learn CMake")
set(Torch_DIR /opt/conda/lib/python3.8/site-packages/torch/share/cmake/Torch/)
set(CUDA_TOOLKIT_ROOT_DIR /usr/local/cuda-11/)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

find_package(MPI REQUIRED)
find_package(CUDA REQUIRED)
find_package(Torch REQUIRED)

set(CMAKE_CXX_STANDARD 14)

include_directories(${CUDA_INCLUDE_DIRS})
include_directories(${MPI_INCLUDE_PATH})
include_directories(${TORCH_INCLUDE_DIRS})

add_executable(mpi_gpu src/mpi_gpu.cpp)

target_link_libraries(mpi_gpu PUBLIC ${CUDA_LIBRARIES})
target_link_libraries(mpi_gpu PUBLIC ${MPI_LIBRARIES})
target_link_libraries(mpi_gpu PUBLIC ${TORCH_LIBRARIES})
