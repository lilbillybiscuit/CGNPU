#!/bin/bash

set -e

echo "Building Heterogeneous Compute Compiler..."

if [[ "$(uname)" != "Darwin" ]]; then
    echo "Error: This project is designed for macOS only"
    exit 1
fi

if ! xcrun --show-sdk-path &> /dev/null; then
    echo "Error: Xcode SDK not found. Please run:"
    echo "xcode-select --install"
    echo "sudo xcode-select -s /Applications/Xcode.app/Contents/Developer"
    exit 1
fi

export SDKROOT=$(xcrun --show-sdk-path)
export CMAKE_OSX_SYSROOT=$SDKROOT

if ! command -v llvm-config &> /dev/null; then
    echo "Warning: LLVM not found in PATH. Please install with:"
    echo "brew install llvm"
    echo "export PATH=\"/opt/homebrew/opt/llvm/bin:\$PATH\""
    exit 1
fi

if ! command -v cmake &> /dev/null; then
    echo "Error: CMake not found. Please install with:"
    echo "brew install cmake"
    exit 1
fi

if [ ! -d "/opt/homebrew/include/nlohmann" ]; then
    echo "Warning: nlohmann/json not found. Please install with:"
    echo "brew install nlohmann-json"
    exit 1
fi

rm -rf build
mkdir -p build
cd build

echo "Configuring project..."
cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_MAKE_PROGRAM=make

echo "Building project..."
make -j$(sysctl -n hw.ncpu)

echo "Build complete!"
echo "Executables are in build/ directory:"
echo "- build/programs/matrix_mult/matrix_mult"
echo "- build/compiler/compiler"
echo "- build/runtime/runtime"

cd ..
chmod +x scripts/*.sh
