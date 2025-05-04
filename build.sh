#!/bin/bash

# Create build directory if it doesn't exist
mkdir -p build

# Navigate to build directory
cd build

# Generate build files with CMake
cmake ..

# Build the project
cmake --build .

echo "Build complete. Executable is at ./build/C__MPS"