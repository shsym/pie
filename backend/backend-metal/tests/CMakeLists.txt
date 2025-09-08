cmake_minimum_required(VERSION 3.23)

# Metal backend tests
project(metal_backend_tests CXX OBJCXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Enable Objective-C++ for Metal integration
enable_language(OBJCXX)

# Use libc++ explicitly on macOS
if(APPLE)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -stdlib=libc++")
    set(CMAKE_OBJCXX_FLAGS "${CMAKE_OBJCXX_FLAGS} -stdlib=libc++")
endif()

# Find required frameworks on macOS
if(APPLE)
    find_library(METAL_FRAMEWORK Metal REQUIRED)
    find_library(FOUNDATION_FRAMEWORK Foundation REQUIRED)
else()
    message(FATAL_ERROR "Metal backend tests require macOS")
endif()

# Find nlohmann/json for JSON parsing
find_package(nlohmann_json QUIET)
if(NOT nlohmann_json_FOUND)
    # Fallback to header-only include if package not found
    message(STATUS "nlohmann_json package not found, using header-only include")
endif()

# Include directories
include_directories(src)
include_directories(../src)

# Detect C++ standard library include path automatically
if(APPLE)
    execute_process(
        COMMAND xcrun --show-sdk-path
        OUTPUT_VARIABLE SDK_ROOT
        OUTPUT_STRIP_TRAILING_WHITESPACE
    )

    set(CXX_STDLIB_PATH "${SDK_ROOT}/usr/include/c++/v1")
    if(EXISTS "${CXX_STDLIB_PATH}/cstdint")
        include_directories(SYSTEM ${CXX_STDLIB_PATH})
        message(STATUS "Using C++ stdlib from: ${CXX_STDLIB_PATH}")
    endif()
endif()

# Test sources
set(TEST_SOURCES
    src/artifact_reader.mm
    src/tensor_comparator.mm
    src/dtype_validator.mm
    src/validation_reporter.mm
    src/test_forward_pass_validation.mm
)

# Create test executable
add_executable(test_cuda_artifact_validation ${TEST_SOURCES})

# Link libraries
target_link_libraries(test_cuda_artifact_validation
    ${METAL_FRAMEWORK}
    ${FOUNDATION_FRAMEWORK}
    metal_dtype_conversion
    metal_common
)

# Link nlohmann_json if available as package
if(nlohmann_json_FOUND)
    target_link_libraries(test_cuda_artifact_validation nlohmann_json::nlohmann_json)
endif()

# Set Objective-C++ compiler flags
set_target_properties(test_cuda_artifact_validation PROPERTIES
    COMPILE_FLAGS "-fobjc-arc"
)

# Set output directory
set_target_properties(test_cuda_artifact_validation PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)

# Add install target
install(TARGETS test_cuda_artifact_validation
        DESTINATION bin)