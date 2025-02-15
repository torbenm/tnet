# Set the minimum version of CMake that can be used
# To find the cmake version run
# $ cmake --version
cmake_minimum_required(VERSION 3.31)

# Set the project name
project (tnet VERSION 1.0 LANGUAGES C)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin)

# Find argp-standalone
find_path(ARGP_INCLUDE_DIRS argp.h)
find_library(ARGP_LIBRARIES NAMES argp)

if (ARGP_INCLUDE_DIRS AND ARGP_LIBRARIES)
    message(STATUS "Found argp: ${ARGP_LIBRARIES}")
else()
    message(FATAL_ERROR "argp not found. Please install argp-standalone.")
endif()

# Collect all source files recursively
file(GLOB_RECURSE SOURCES ${CMAKE_SOURCE_DIR}/src/*.c)

# Collect all header files (optional, useful for IDEs)
file(GLOB_RECURSE HEADERS ${CMAKE_SOURCE_DIR}/include/*.h ${CMAKE_SOURCE_DIR}/include/*.hpp)

# Add the executable
add_executable(${PROJECT_NAME} ${SOURCES})

# Include directories
target_include_directories(${PROJECT_NAME} PRIVATE ${CMAKE_SOURCE_DIR}/include ${ARGP_INCLUDE_DIRS})
target_link_libraries(${PROJECT_NAME}  PRIVATE ${ARGP_LIBRARIES})


# Set build type to Release by default
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()