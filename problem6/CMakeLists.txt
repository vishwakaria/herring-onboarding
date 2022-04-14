cmake_minimum_required(VERSION 3.15)

project(hello_world 
	VERSION 1.0
	DESCRIPTION "Learn CMake")

set(CMAKE_CXX_FLAGS -pthread)

add_executable(hello_world src/hello_world.cpp)
