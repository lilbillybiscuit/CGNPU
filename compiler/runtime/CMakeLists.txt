project(runtime LANGUAGES CXX OBJCXX)

set(CMAKE_OBJCXX_FLAGS "${CMAKE_OBJCXX_FLAGS} -framework Metal -framework CoreML -framework Foundation")

add_custom_command(
        OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/matrix_mult.metallib
        COMMAND xcrun -sdk macosx metal -c ${CMAKE_CURRENT_SOURCE_DIR}/metal/matrix_mult.metal -o ${CMAKE_CURRENT_BINARY_DIR}/matrix_mult.air
        COMMAND xcrun -sdk macosx metallib ${CMAKE_CURRENT_BINARY_DIR}/matrix_mult.air -o ${CMAKE_CURRENT_BINARY_DIR}/matrix_mult.metallib
        DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/metal/matrix_mult.metal
)

add_custom_target(metal_shader DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/matrix_mult.metallib)

add_executable(runtime
        src/main.cpp
        src/runtime.cpp
        src/device_manager.cpp
        src/work_stealing.cpp
        src/cpu_executor.cpp
        src/gpu_executor.mm
        src/ane_executor.cpp
        src/profiler.cpp
        coreml/coreml_model_builder.mm
)

add_dependencies(runtime metal_shader)

target_include_directories(runtime PRIVATE
        ${CMAKE_SOURCE_DIR}/common/src
        ${CMAKE_CURRENT_SOURCE_DIR}/src
)

target_link_libraries(runtime PRIVATE
        common
        nlohmann_json::nlohmann_json
)

add_custom_command(TARGET runtime POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy
        ${CMAKE_CURRENT_BINARY_DIR}/matrix_mult.metallib
        ${CMAKE_BINARY_DIR}/matrix_mult.metallib
)