add_library(common STATIC
        src/instruction_set.cpp
        src/bytecode_format.cpp
        src/matrix_utils.cpp
        src/metal_buffer_wrapper.mm
)

target_include_directories(common PUBLIC src)
target_link_libraries(common PUBLIC nlohmann_json::nlohmann_json)

find_library(METAL_FRAMEWORK Metal REQUIRED)
find_library(FOUNDATION_FRAMEWORK Foundation REQUIRED)
find_library(COREML_FRAMEWORK CoreML REQUIRED)

set_source_files_properties(src/matrix_utils.cpp PROPERTIES
        COMPILE_FLAGS "-x objective-c++"
)

set_source_files_properties(src/metal_buffer_wrapper.mm PROPERTIES
        COMPILE_FLAGS "-x objective-c++"
)

target_link_libraries(common PUBLIC
        ${METAL_FRAMEWORK}
        ${FOUNDATION_FRAMEWORK}
        ${COREML_FRAMEWORK}
)