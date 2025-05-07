cmake_minimum_required(VERSION 3.20)

if(NOT CMAKE_OSX_SYSROOT)
    execute_process(COMMAND xcrun --show-sdk-path OUTPUT_VARIABLE CMAKE_OSX_SYSROOT OUTPUT_STRIP_TRAILING_WHITESPACE)
endif()

project(HeteroCompute LANGUAGES C CXX OBJCXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT APPLE)
    message(FATAL_ERROR "This project is designed for Apple M1 platforms only")
endif()

find_package(nlohmann_json REQUIRED)

add_subdirectory(common)
add_subdirectory(programs/matrix_mult)
add_subdirectory(compiler)
add_subdirectory(runtime)

add_custom_target(test_all
        COMMAND ${CMAKE_SOURCE_DIR}/scripts/run_example.sh
        DEPENDS matrix_mult compiler runtime
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
)