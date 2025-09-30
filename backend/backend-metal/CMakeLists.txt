# Set minimum CMake version for Metal support
cmake_minimum_required(VERSION 3.23)

# Detect Metal availability (macOS only)
if(NOT APPLE)
    message(FATAL_ERROR "Metal backend requires macOS")
endif()

project(PIE_METAL_BACKEND CXX)

set(CMAKE_BUILD_TYPE Release)

# Set C++ Standard to match other backends
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Define output directories for binaries, matching backend-cuda structure
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

# Find Metal compiler
find_program(METAL_COMPILER xcrun REQUIRED)

# Include CPM.cmake for package management
include(cmake/CPM.cmake)

# Add pybind11 for Python bindings
CPMAddPackage(
  URI "gh:pybind/pybind11@2.11.1"
  OPTIONS "PYBIND11_FINDPYTHON ON"
)

# Find all Metal source files
file(GLOB METAL_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/src/*.metal")

# Create Metal library compilation target
set(METAL_AIR_FILES "")
foreach(METAL_SOURCE ${METAL_SOURCES})
    get_filename_component(KERNEL_NAME ${METAL_SOURCE} NAME_WE)
    set(AIR_FILE "${CMAKE_BINARY_DIR}/${KERNEL_NAME}.air")

    # Compile .metal to .air
    add_custom_command(
        OUTPUT ${AIR_FILE}
        COMMAND ${METAL_COMPILER} -sdk macosx metal -c ${METAL_SOURCE} -o ${AIR_FILE}
        DEPENDS ${METAL_SOURCE}
        COMMENT "Compiling Metal kernel: ${KERNEL_NAME}"
    )

    list(APPEND METAL_AIR_FILES ${AIR_FILE})
endforeach()

# Link all .air files into single .metallib
set(METALLIB_OUTPUT "${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/pie_metal_kernels.metallib")
add_custom_command(
    OUTPUT ${METALLIB_OUTPUT}
    COMMAND ${METAL_COMPILER} -sdk macosx metallib ${METAL_AIR_FILES} -o ${METALLIB_OUTPUT}
    DEPENDS ${METAL_AIR_FILES}
    COMMENT "Creating Metal library: pie_metal_kernels.metallib"
)

# Create custom target for Metal library
add_custom_target(pie_metal_kernels ALL DEPENDS ${METALLIB_OUTPUT})

# Python bindings module (main bindings)
pybind11_add_module(metal_bindings
    src/bindings.mm
)

# Python attention bindings module (dedicated attention bindings)
pybind11_add_module(metal_attention_bindings
    src/attention_bindings.mm
)

# Link Metal framework to both modules
target_link_libraries(metal_bindings PRIVATE
    "-framework Metal"
    "-framework Foundation"
)

target_link_libraries(metal_attention_bindings PRIVATE
    "-framework Metal"
    "-framework Foundation"
)

# Set target properties for Python modules
set_target_properties(metal_bindings PROPERTIES
    CXX_VISIBILITY_PRESET "hidden"
    VISIBILITY_INLINES_HIDDEN YES
)

set_target_properties(metal_attention_bindings PROPERTIES
    CXX_VISIBILITY_PRESET "hidden"
    VISIBILITY_INLINES_HIDDEN YES
)

# Ensure Metal library is built before bindings
add_dependencies(metal_bindings pie_metal_kernels)
add_dependencies(metal_attention_bindings pie_metal_kernels)

# Copy metallib to bindings output directory for runtime access
add_custom_command(TARGET metal_bindings POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy ${METALLIB_OUTPUT} $<TARGET_FILE_DIR:metal_bindings>/
    COMMENT "Copying metallib to bindings directory"
)

add_custom_command(TARGET metal_attention_bindings POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy ${METALLIB_OUTPUT} $<TARGET_FILE_DIR:metal_attention_bindings>/
    COMMENT "Copying metallib to attention bindings directory"
)

# Custom run target (optional)
add_custom_target(
    test_metal
    COMMAND python3 -c "import metal_bindings; print('Metal bindings loaded successfully')"
    DEPENDS metal_bindings
    WORKING_DIRECTORY ${CMAKE_LIBRARY_OUTPUT_DIRECTORY}
    COMMENT "Testing Metal bindings..."
)

# --- End of CMakeLists.txt ---