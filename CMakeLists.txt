cmake_minimum_required(VERSION 3.10)

# set project
project(PINNTest VERSION 0.1.0)
set(CMAKE_CXX_COMPLIER "g++")
set(CMAKE_CXX_STANDARD 17)

# set output directory
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# aOpenGL
set(aOpenGL_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../aOpenGL)
find_package(aOpenGL REQUIRED)

# aLibTorch
set(aLibTorch_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../aLibTorch)
find_package(aLibTorch REQUIRED)

add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/code) # ! the order matters, after all the external libraries!
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/main)

