#cmake_minimum_required(VERSION 3.10)
#project(SurfaceReader)
#
## Set the C++ standard
#set(CMAKE_CXX_STANDARD 14)
#set(CMAKE_CXX_STANDARD_REQUIRED True)
#
## Find Boost headers
#find_package(Boost 1.65 REQUIRED)  # version >= 1.65
#
## Add the executable
#add_executable(SurfaceReader src/main.cpp)
#
## Include Boost headers
#if(Boost_FOUND)
#    target_include_directories(SurfaceReader PRIVATE ${Boost_INCLUDE_DIRS})
#endif()
#
cmake_minimum_required(VERSION 3.24)
#
project(SurfaceReader LANGUAGES CXX)

# Set C++ standard
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# -------------------------
# Modern Boost config-mode
# -------------------------
# Use Config-mode to find Boost headers (math functions are header-only)
# CMake will search for BoostConfig.cmake
#find_package(Boost 1.65 REQUIRED)
find_package(Boost 1.65 CONFIG REQUIRED)

# Add executable
add_executable(SurfaceReader src/main.cpp)

# Link/include Boost
if(Boost_FOUND)
    target_include_directories(SurfaceReader PRIVATE ${Boost_INCLUDE_DIRS})
endif()

