cmake_minimum_required(VERSION 3.15)
project(HeterogeneousMatrixMultiplication)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include_directories(${CMAKE_CURRENT_SOURCE_DIR})
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/metal-cpp)

set(SOURCE_FILES main.cpp)

add_executable(${PROJECT_NAME} ${SOURCE_FILES})

find_library(METAL_FRAMEWORK Metal REQUIRED)
find_library(FOUNDATION_FRAMEWORK Foundation REQUIRED)
find_library(QUARTZCORE_FRAMEWORK QuartzCore REQUIRED)
find_library(ACCELERATE_FRAMEWORK Accelerate REQUIRED)

target_link_libraries(${PROJECT_NAME} PRIVATE
    ${METAL_FRAMEWORK}
    ${FOUNDATION_FRAMEWORK}
    ${QUARTZCORE_FRAMEWORK}
    ${ACCELERATE_FRAMEWORK}
    "-framework CoreML"
)

target_compile_options(${PROJECT_NAME} PRIVATE
    -Wall
    -Wextra
    -Wno-deprecated-declarations
)

message(STATUS "Metal framework: ${METAL_FRAMEWORK}")
message(STATUS "Foundation framework: ${FOUNDATION_FRAMEWORK}")
message(STATUS "QuartzCore framework: ${QUARTZCORE_FRAMEWORK}")
message(STATUS "Accelerate framework: ${ACCELERATE_FRAMEWORK}")
