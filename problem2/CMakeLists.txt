cmake_minimum_required(VERSION 3.15)

project(mpi_hello_world 
	VERSION 1.0
	DESCRIPTION "Learn CMake")

find_package(MPI REQUIRED)
include_directories(${MPI_INCLUDE_PATH})

add_executable(mpi_hello_world src/mpi_hello_world.c)
target_link_libraries(mpi_hello_world ${MPI_LIBRARIES})

