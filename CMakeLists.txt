cmake_minimum_required(VERSION 3.30)
project(C__MPS)
find_library(FOUNDATION_FRAMEWORK Foundation)
find_library(METAL_FRAMEWORK Metal)
find_library(OBJC_RUNTIME objc)
include_directories(metal-cpp)
set(CMAKE_CXX_STANDARD 23)

add_executable(C__MPS main.cpp)
target_link_libraries(C__MPS
    PRIVATE
    ${FOUNDATION_FRAMEWORK}
    ${METAL_FRAMEWORK}
    ${OBJC_RUNTIME}
)

