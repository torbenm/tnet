# Set the minimum version of CMake that can be used
# To find the cmake version run
# $ cmake --version
cmake_minimum_required(VERSION 3.31)

# Set the project name
project (tnet VERSION 1.0 LANGUAGES C)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin)

# Collect all source files recursively
file(GLOB_RECURSE SOURCES ${CMAKE_SOURCE_DIR}/src/*.c)

# Add the executable
add_executable(${PROJECT_NAME} ${SOURCES})


# Set build type to Release by default
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()