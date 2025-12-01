cmake_minimum_required(VERSION 3.24)

project(SurfaceReader LANGUAGES CXX)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

cmake_policy(SET CMP0144 NEW)
cmake_policy(SET CMP0167 NEW)

# -------------------------
# Modern Boost config-mode
# -------------------------
# Use Config-mode to find Boost headers (math functions are header-only)
# CMake will search for BoostConfig.cmake
#find_package(Boost 1.65 REQUIRED)

# Now find Boost
# Tell CMake exactly where Boost config files are

set(Boost_NO_BOOST_CMAKE ON)  # force FindBoost mode
set(BOOST_ROOT /opt/homebrew/opt/boost)
find_package(Boost 1.89.0 REQUIRED)

# Find GSL
find_package(GSL REQUIRED)

#find_package(Boost 1.89 CONFIG REQUIRED HINTS /opt/homebrew/opt/boost)
# Include all .cpp files in src/
file(GLOB SOURCES "src/*.cpp")

# Add executable
add_executable(SurfaceReader ${SOURCES})

# Link/include Boost
if(Boost_FOUND)
    target_include_directories(SurfaceReader PRIVATE ${Boost_INCLUDE_DIRS})
endif()

if(GSL_FOUND)
    message(STATUS "Found GSL: ${GSL_INCLUDE_DIRS}, ${GSL_LIBRARIES}")
    target_include_directories(SurfaceReader PRIVATE ${GSL_INCLUDE_DIRS})
    target_link_libraries(SurfaceReader PRIVATE ${GSL_LIBRARIES} m)
endif()


# -----------------------------
# OpenMP configuration (macOS)
# -----------------------------
message(STATUS "Configuring OpenMP for macOS (AppleClang)...")
# Include and link manually because AppleClang doesn’t have OpenMP built-in
target_compile_options(SurfaceReader PRIVATE -Xpreprocessor -fopenmp)
target_link_libraries(SurfaceReader PRIVATE /opt/homebrew/opt/libomp/lib/libomp.dylib)
target_include_directories(SurfaceReader PRIVATE /opt/homebrew/opt/libomp/include)
target_compile_definitions(SurfaceReader PRIVATE USE_OPENMP)
