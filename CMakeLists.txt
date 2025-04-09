cmake_minimum_required(VERSION 3.10)
project(SurfaceReader)

# Set the C++ standard
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Add the executable
add_executable(SurfaceReader src/main.cpp)
