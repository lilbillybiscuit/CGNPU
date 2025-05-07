# Find LLVM
set(CMAKE_PREFIX_PATH "/opt/homebrew/opt/llvm/lib/cmake/llvm")
find_package(LLVM REQUIRED CONFIG)
message(STATUS "Found LLVM ${LLVM_PACKAGE_VERSION}")
message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")

include_directories(${LLVM_INCLUDE_DIRS})
add_definitions(${LLVM_DEFINITIONS})

# Find nlohmann/json
find_package(nlohmann_json REQUIRED)

add_executable(compiler
        src/main.cpp
        src/parser.cpp
        src/ir_generator.cpp
        src/bytecode_generator.cpp
)

target_include_directories(compiler PRIVATE
        ${CMAKE_SOURCE_DIR}/common/src
        ${CMAKE_CURRENT_SOURCE_DIR}/src
)

# Link LLVM libraries
llvm_map_components_to_libnames(llvm_libs support core irreader)
target_link_libraries(compiler PRIVATE
        common
        nlohmann_json::nlohmann_json
        ${llvm_libs}
)