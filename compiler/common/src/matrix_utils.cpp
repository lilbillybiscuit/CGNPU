#include "matrix_utils.h"
#include "metal_buffer_wrapper.h"
#include <Metal/Metal.h>
#include <algorithm>
#include <iostream>

MatrixBuffer::MatrixBuffer(int size) 
    : size(size), 
      unifiedBuffer(nullptr), 
      metalBuffer(nullptr), 
      aneModel(nullptr) {
    state.store(MemoryAccessState::SHARED);
    size_t bufferSize = size * size * sizeof(int);
    metalBuffer = new MTLBufferWrapper();
    unifiedBuffer = metalBuffer->createBuffer(bufferSize, true);
    if (unifiedBuffer) {
        memset(unifiedBuffer, 0, bufferSize);
    } else {
        std::cerr << "ERROR: Failed to allocate unified memory for matrix of size " 
                  << size << "x" << size << std::endl;
        throw std::runtime_error("Failed to allocate unified memory for matrix");
    }
}

MatrixBuffer::~MatrixBuffer() {
    releaseResources();
}

void* MatrixBuffer::getUnifiedBufferPtr() {
    return unifiedBuffer;
}

int* MatrixBuffer::getCPUReadPtr() {
    std::lock_guard lock(accessMutex);
    auto currentState = state.load();
    if (currentState == MemoryAccessState::GPU_WRITING || 
        currentState == MemoryAccessState::ANE_WRITING) {
        syncFromDevice();
    }
    state.store(MemoryAccessState::CPU_READING);
    return static_cast<int*>(unifiedBuffer);
}
int* MatrixBuffer::getCPUWritePtr() {
    std::lock_guard lock(accessMutex);
    auto currentState = state.load();
    if (currentState == MemoryAccessState::GPU_WRITING || 
        currentState == MemoryAccessState::ANE_WRITING) {
        syncFromDevice();
    }
    state.store(MemoryAccessState::CPU_WRITING);
    return static_cast<int*>(unifiedBuffer);
}

void MatrixBuffer::releaseCPUAccess() {
    std::lock_guard lock(accessMutex);
    auto currentState = state.load();
    if (currentState == MemoryAccessState::CPU_READING || 
        currentState == MemoryAccessState::CPU_WRITING) {
        state.store(MemoryAccessState::SHARED);
    }
}

void MatrixBuffer::prepareForGPUAccess(bool readOnly) {
    std::lock_guard lock(accessMutex);
    auto currentState = state.load();
    if (currentState == MemoryAccessState::CPU_WRITING) {
        syncToDevice();
    }
    if (readOnly) {
        state.store(MemoryAccessState::GPU_READING);
    } else {
        state.store(MemoryAccessState::GPU_WRITING);
    }
}

void MatrixBuffer::releaseGPUAccess() {
    std::lock_guard lock(accessMutex);
    auto currentState = state.load();
    if (currentState == MemoryAccessState::GPU_READING || 
        currentState == MemoryAccessState::GPU_WRITING) {
        if (currentState == MemoryAccessState::GPU_WRITING) {
            metalBuffer->syncContents();
        }
        state.store(MemoryAccessState::SHARED);
    }
}

void MatrixBuffer::prepareForANEAccess(bool readOnly) {
    std::lock_guard<std::mutex> lock(accessMutex);
    auto currentState = state.load();
    if (currentState == MemoryAccessState::CPU_WRITING) {
        syncToDevice();
    }
    if (readOnly) {
        state.store(MemoryAccessState::ANE_READING);
    } else {
        state.store(MemoryAccessState::ANE_WRITING);
    }
}

void MatrixBuffer::releaseANEAccess() {
    std::lock_guard<std::mutex> lock(accessMutex);
    auto currentState = state.load();
    if (currentState == MemoryAccessState::ANE_READING || 
        currentState == MemoryAccessState::ANE_WRITING) {
        state.store(MemoryAccessState::SHARED);
    }
}

void MatrixBuffer::syncToDevice() const {
    if (metalBuffer) {
        metalBuffer->syncContents();
    }
}

void MatrixBuffer::syncFromDevice() {
}

void MatrixBuffer::releaseResources() {
    if (metalBuffer) {
        delete metalBuffer;
        metalBuffer = nullptr;
    }
    if (aneModel) {
        CFBridgingRelease(aneModel);
        aneModel = nullptr;
    }
    unifiedBuffer = nullptr;
}

int MatrixBuffer::get(int row, int col) const {
    if (row < 0 || row >= size || col < 0 || col >= size) {
        throw std::out_of_range("Matrix index out of bounds");
    }
    return static_cast<int*>(unifiedBuffer)[row * size + col];
}

void MatrixBuffer::set(int row, int col, int value) {
    if (row < 0 || row >= size || col < 0 || col >= size) {
        throw std::out_of_range("Matrix index out of bounds");
    }
    static_cast<int*>(unifiedBuffer)[row * size + col] = value;
}

int& MatrixBuffer::operator[](size_t index) {
    if (index >= static_cast<size_t>(size * size)) {
        throw std::out_of_range("Matrix index out of bounds");
    }
    return static_cast<int*>(unifiedBuffer)[index];
}

const int& MatrixBuffer::operator[](size_t index) const {
    if (index >= static_cast<size_t>(size * size)) {
        throw std::out_of_range("Matrix index out of bounds");
    }
    return static_cast<int*>(unifiedBuffer)[index];
}

int* MatrixBuffer::getRawData() {
    return static_cast<int*>(unifiedBuffer);
}

std::vector<WorkChunk> createWorkChunks(int matrixSize, int numChunks) {
    std::vector<WorkChunk> chunks;
    if (matrixSize <= 128) {
        int blockSize = std::min(32, matrixSize / 4);
        if (matrixSize % blockSize != 0) {
            while (matrixSize % blockSize != 0 && blockSize > 4) {
                blockSize -= 4; 
            }
        }
        for (int i = 0; i < matrixSize; i += blockSize) {
            for (int j = 0; j < matrixSize; j += blockSize) {
                int endRow = std::min(i + blockSize, matrixSize);
                int endCol = std::min(j + blockSize, matrixSize);
                chunks.emplace_back(i, endRow, j, endCol);
            }
        }
    } else {
        int blockSize = std::max(4, matrixSize / (int)sqrt(numChunks));
        for (int i = 0; i < matrixSize; i += blockSize) {
            for (int j = 0; j < matrixSize; j += blockSize) {
                int endRow = std::min(i + blockSize, matrixSize);
                int endCol = std::min(j + blockSize, matrixSize);
                chunks.emplace_back(i, endRow, j, endCol);
            }
        }
    }
    return chunks;
}

void partitionChunks(std::vector<WorkChunk>& chunks,
                    std::vector<WorkChunk>& cpu,
                    std::vector<WorkChunk>& gpu,
                    std::vector<WorkChunk>& ane) {
    int totalChunks = chunks.size();
    int cpuCount = totalChunks * 0.3;
    int gpuCount = totalChunks * 0.5;
    // int aneCount = totalChunks - cpuCount - gpuCount;
    for (int i = 0; i < chunks.size(); i++) {
        if (i < cpuCount) cpu.push_back(chunks[i]);
        else if (i < cpuCount + gpuCount) gpu.push_back(chunks[i]);
        else ane.push_back(chunks[i]);
    }
}