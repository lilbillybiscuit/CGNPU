#include <iostream>
#include <thread>
#include <chrono>
#include <vector>
#include <random>
#include <chrono>             

#define NS_PRIVATE_IMPLEMENTATION
#define CA_PRIVATE_IMPLEMENTATION
#define MTL_PRIVATE_IMPLEMENTATION

#include "metal-cpp/Foundation/Foundation.hpp"
#include "metal-cpp/Metal/Metal.hpp"
#include "metal-cpp/QuartzCore/QuartzCore.hpp"

#include <Accelerate/Accelerate.h>

const char* matrixMultiplyShader = R"(
#include <metal_stdlib>
using namespace metal;

kernel void matrixMultiply(
    device const float* matrixA [[buffer(0)]],
    device const float* matrixB [[buffer(1)]],
    device float* result [[buffer(2)]],
    device const uint& width [[buffer(3)]],
    uint2 position [[thread_position_in_grid]]
) {
    if (position.x >= width || position.y >= width) {
        return;
    }
    
    float sum = 0.0f;
    for (uint i = 0; i < width; i++) {
        sum += matrixA[position.y * width + i] * matrixB[i * width + position.x];
    }
    
    result[position.y * width + position.x] = sum;
}

kernel void matrixMultiplyTiled(
    device const float* matrixA [[buffer(0)]],
    device const float* matrixB [[buffer(1)]],
    device float* result [[buffer(2)]],
    device const uint& width [[buffer(3)]],
    uint2 position [[thread_position_in_grid]],
    uint2 threadgroupPosition [[threadgroup_position_in_grid]],
    uint2 threadPosition [[thread_position_in_threadgroup]]
) {
    const uint TILE_SIZE = 16;
    
    threadgroup float tileA[TILE_SIZE][TILE_SIZE];
    threadgroup float tileB[TILE_SIZE][TILE_SIZE];
    
    uint row = position.y;
    uint col = position.x;
    
    float sum = 0.0f;
    
    uint numTiles = (width + TILE_SIZE - 1) / TILE_SIZE;
    
    for (uint t = 0; t < numTiles; t++) {
        uint localRow = threadPosition.y;
        uint localCol = threadPosition.x;

        if (row < width && (t * TILE_SIZE + localCol) < width) {
            tileA[localRow][localCol] = matrixA[row * width + t * TILE_SIZE + localCol];
        } else {
            tileA[localRow][localCol] = 0.0;
        }
        
        if (col < width && (t * TILE_SIZE + localRow) < width) {
            tileB[localRow][localCol] = matrixB[(t * TILE_SIZE + localRow) * width + col];
        } else {
            tileB[localRow][localCol] = 0.0;
        }
        
        threadgroup_barrier(mem_flags::mem_threadgroup);
        
        for (uint i = 0; i < TILE_SIZE; i++) {
            sum += tileA[localRow][i] * tileB[i][localCol];
        }
        
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    
    if (row < width && col < width) {
        result[row * width + col] = sum;
    }
}
)";

class HeterogeneousMatrixMultiplier {
private:
    uint32_t m_matrixSize;
    
    MTL::Device* m_device;
    MTL::CommandQueue* m_commandQueue;
    MTL::ComputePipelineState* m_simplePipelineState;
    MTL::ComputePipelineState* m_tiledPipelineState;
    MTL::Buffer* m_bufferA;
    MTL::Buffer* m_bufferB;
    MTL::Buffer* m_gpuResultBuffer;
    MTL::Library* m_library;
    
    std::vector<float> m_matrixA;
    std::vector<float> m_matrixB;
    std::vector<float> m_result;
    
    float m_cpuWorkPercentage;
    float m_gpuWorkPercentage;
    std::chrono::duration<double> m_cpuExecutionTime{0};
    std::chrono::duration<double> m_gpuExecutionTime{0};
    // std::chrono::duration<double> m_npuExecutionTime{0};
    
    bool setupMetal() {
        m_device = MTL::CreateSystemDefaultDevice();
        if (!m_device) {
            std::cerr << "Failed to create Metal device." << std::endl;
            return false;
        }
        
        m_commandQueue = m_device->newCommandQueue();
        if (!m_commandQueue) {
            std::cerr << "Failed to create command queue." << std::endl;
            return false;
        }
        
        NS::Error* error = nullptr;
        m_library = m_device->newLibrary(NS::String::string(matrixMultiplyShader, 
                                     NS::StringEncoding::UTF8StringEncoding), 
                                     nullptr, 
                                     &error);
        if (!m_library) {
            std::cerr << "Failed to create library: ";
            if (error) {
                std::cerr << error->localizedDescription()->utf8String();
            }
            std::cerr << std::endl;
            return false;
        }
        
        MTL::Function* simpleFunction = m_library->newFunction(NS::String::string("matrixMultiply", 
                                                    NS::StringEncoding::UTF8StringEncoding));
        if (!simpleFunction) {
            std::cerr << "Failed to create simple function." << std::endl;
            return false;
        }
        
        m_simplePipelineState = m_device->newComputePipelineState(simpleFunction, &error);
        if (!m_simplePipelineState) {
            std::cerr << "Failed to create simple pipeline state: ";
            if (error) {
                std::cerr << error->localizedDescription()->utf8String();
            }
            std::cerr << std::endl;
            return false;
        }
        
        MTL::Function* tiledFunction = m_library->newFunction(NS::String::string("matrixMultiplyTiled", 
                                                    NS::StringEncoding::UTF8StringEncoding));
        if (!tiledFunction) {
            std::cerr << "Failed to create tiled function." << std::endl;
            return false;
        }
        
        m_tiledPipelineState = m_device->newComputePipelineState(tiledFunction, &error);
        if (!m_tiledPipelineState) {
            std::cerr << "Failed to create tiled pipeline state: ";
            if (error) {
                std::cerr << error->localizedDescription()->utf8String();
            }
            std::cerr << std::endl;
            return false;
        }
        
        simpleFunction->release();
        tiledFunction->release();
        
        return true;
    }
    
    bool createBuffers() {
        const size_t matrixBytes = m_matrixSize * m_matrixSize * sizeof(float);
        
        m_bufferA = m_device->newBuffer(m_matrixA.data(), matrixBytes, MTL::ResourceStorageModeShared);
        m_bufferB = m_device->newBuffer(m_matrixB.data(), matrixBytes, MTL::ResourceStorageModeShared);
        m_gpuResultBuffer = m_device->newBuffer(matrixBytes, MTL::ResourceStorageModeShared);
        
        if (!m_bufferA || !m_bufferB || !m_gpuResultBuffer) {
            std::cerr << "Failed to create Metal buffers." << std::endl;
            return false;
        }
        
        return true;
    }
    
    void computeOnCPU(size_t startRow, size_t endRow) {
        // std::cout << "CPU processing rows " << startRow << " to " << endRow << std::endl;
        
        auto startTime = std::chrono::high_resolution_clock::now();
        
        for (size_t i = startRow; i < endRow; i++) {
            for (size_t j = 0; j < m_matrixSize; j++) {
                float dotProduct = 0.0f;
                vDSP_dotpr(&m_matrixA[i * m_matrixSize], 1, 
                           &m_matrixB[j], m_matrixSize, 
                           &dotProduct, m_matrixSize);
                m_result[i * m_matrixSize + j] = dotProduct;
            }
        }
        
        auto endTime = std::chrono::high_resolution_clock::now();
        m_cpuExecutionTime = endTime - startTime;
    }

    // void computeOnCPU(size_t startRow, size_t endRow) {
    //     auto t0 = std::chrono::high_resolution_clock::now();
    
    //     const int N = static_cast<int>(m_matrixSize);
    //     const int TILE_I = 32;    // fits L1 lines: 32×32 floats ≈ 4KB
    //     const int TILE_J = 32;    // similarly for columns
    
    //     #pragma omp parallel for collapse(2) schedule(static)
    //     for (int ii = (int)startRow; ii < (int)endRow; ii += TILE_I) {
    //         for (int jj = 0; jj < N; jj += TILE_J) {
    //             int i_max = std::min(ii + TILE_I, (int)endRow);
    //             int j_max = std::min(jj + TILE_J, N);
    
    //             for (int i = ii; i < i_max; ++i) {
    //                 for (int j = jj; j < j_max; ++j) {
    //                     float sum = 0.0f;
    //                     // k‑loop stays in cache due to blocking
    //                     for (int k = 0; k < N; ++k) {
    //                         sum += m_matrixA[i * N + k] * m_matrixB[k * N + j];
    //                     }
    //                     m_result[i * N + j] = sum;
    //                 }
    //             }
    //         }
    //     }
    
    //     auto t1 = std::chrono::high_resolution_clock::now();
    //     m_cpuExecutionTime = t1 - t0;
    // }
    
    // Modify computeOnGPU to include timing
    void computeOnGPU(size_t startRow, size_t endRow) {
        // std::cout << "GPU processing rows " << startRow << " to " << endRow << std::endl;
        
        auto startTime = std::chrono::high_resolution_clock::now();
        
        size_t rowCount = endRow - startRow;
        
        MTL::CommandBuffer* commandBuffer = m_commandQueue->commandBuffer();
        MTL::ComputeCommandEncoder* computeEncoder = commandBuffer->computeCommandEncoder();
        
        computeEncoder->setComputePipelineState(m_tiledPipelineState);
        computeEncoder->setBuffer(m_bufferA, startRow * m_matrixSize * sizeof(float), 0);
        computeEncoder->setBuffer(m_bufferB, 0, 1);
        computeEncoder->setBuffer(m_gpuResultBuffer, 0, 2);
        
        uint32_t width = m_matrixSize;
        computeEncoder->setBytes(&width, sizeof(width), 3);
        
        MTL::Size gridSize = MTL::Size(m_matrixSize, rowCount, 1);
        MTL::Size threadgroupSize = MTL::Size(16, 16, 1);
        
        computeEncoder->dispatchThreads(gridSize, threadgroupSize);
        
        computeEncoder->endEncoding();
        commandBuffer->commit();
        commandBuffer->waitUntilCompleted();
        
        auto endTime = std::chrono::high_resolution_clock::now();
        m_gpuExecutionTime = endTime - startTime;
        
        float* gpuResult = static_cast<float*>(m_gpuResultBuffer->contents());
        std::copy(gpuResult, gpuResult + rowCount * m_matrixSize, &m_result[startRow * m_matrixSize]);
    }
    
public:
    HeterogeneousMatrixMultiplier(uint32_t matrixSize, 
                                 float cpuPercentage = 0.2f, 
                                 float gpuPercentage = 0.6f) 
        : m_matrixSize(matrixSize),
          m_cpuWorkPercentage(cpuPercentage),
          m_gpuWorkPercentage(gpuPercentage),
        //   m_npuWorkPercentage(npuPercentage),
          m_device(nullptr),
          m_commandQueue(nullptr),
          m_simplePipelineState(nullptr),
          m_tiledPipelineState(nullptr),
          m_bufferA(nullptr),
          m_bufferB(nullptr),
          m_gpuResultBuffer(nullptr),
          m_library(nullptr) {
        
        m_matrixA.resize(matrixSize * matrixSize);
        m_matrixB.resize(matrixSize * matrixSize);
        m_result.resize(matrixSize * matrixSize, 0.0f);
    }
    
    ~HeterogeneousMatrixMultiplier() {
        // Clean up Metal resources
        if (m_bufferA) m_bufferA->release();
        if (m_bufferB) m_bufferB->release();
        if (m_gpuResultBuffer) m_gpuResultBuffer->release();
        if (m_simplePipelineState) m_simplePipelineState->release();
        if (m_tiledPipelineState) m_tiledPipelineState->release();
        if (m_library) m_library->release();
        if (m_commandQueue) m_commandQueue->release();
        if (m_device) m_device->release();
    }
    
    bool initialize() {
        if (!setupMetal()) {
            return false;
        }
        
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<float> dist(0.0f, 1.0f);
        
        for (size_t i = 0; i < m_matrixSize * m_matrixSize; i++) {
            m_matrixA[i] = dist(gen);
            m_matrixB[i] = dist(gen);
        }
        
        if (!createBuffers()) {
            return false;
        }
        
        return true;
    }

    void reportProcessorTimings() {
        std::cout << "\n---- Processor Timing Analysis ----\n";
        std::cout << "CPU execution time: " << m_cpuExecutionTime.count() << " seconds\n";
        std::cout << "GPU execution time: " << m_gpuExecutionTime.count() << " seconds\n";
        // std::cout << "NPU execution time: " << m_npuExecutionTime.count() << " seconds\n";
        
        std::chrono::duration<double> maxTime = std::max({m_cpuExecutionTime, m_gpuExecutionTime});
        
        if (maxTime == m_cpuExecutionTime) {
            std::cout << "BOTTLENECK: CPU is the limiting factor\n";
            std::cout << "  - GPU waited for: " << (m_cpuExecutionTime - m_gpuExecutionTime).count() << " seconds\n";
            // std::cout << "  - NPU waited for: " << (m_cpuExecutionTime - m_npuExecutionTime).count() << " seconds\n";
        } else if (maxTime == m_gpuExecutionTime) {
            std::cout << "BOTTLENECK: GPU is the limiting factor\n";
            std::cout << "  - CPU waited for: " << (m_gpuExecutionTime - m_cpuExecutionTime).count() << " seconds\n";
            // std::cout << "  - NPU waited for: " << (m_gpuExecutionTime - m_npuExecutionTime).count() << " seconds\n";
        }
    }
    
    void multiply() {
        size_t totalRows = m_matrixSize;
        
        size_t cpuRows = static_cast<size_t>(totalRows * m_cpuWorkPercentage);
        size_t gpuRows = static_cast<size_t>(totalRows * m_gpuWorkPercentage);
        // size_t npuRows = totalRows - cpuRows - gpuRows;
        
        size_t cpuStart = 0;
        size_t cpuEnd = cpuRows;
        
        size_t gpuStart = cpuEnd;
        size_t gpuEnd = gpuStart + gpuRows;
        
        // size_t npuStart = gpuEnd;
        // size_t npuEnd = totalRows;
        
        printf("CPU processing rows %zu to %zu\n", cpuStart, cpuEnd);
        printf("GPU processing rows %zu to %zu\n", gpuStart, gpuEnd);
        std::thread cpuThread(&HeterogeneousMatrixMultiplier::computeOnCPU, this, cpuStart, cpuEnd);
        std::thread gpuThread(&HeterogeneousMatrixMultiplier::computeOnGPU, this, gpuStart, gpuEnd);
        // std::thread npuThread(&HeterogeneousMatrixMultiplier::simulateNPU, this, npuStart, npuEnd);
        
        cpuThread.join();
        gpuThread.join();
        // npuThread.join();
    }
    
    void printMatrixSection(const std::vector<float>& matrix, const std::string& name, size_t size = 5) {
        size_t display_size = std::min<size_t>(size, m_matrixSize);
        
        std::cout << "Matrix " << name << " (" << display_size << "x" << display_size << " section):" << std::endl;
        for (size_t i = 0; i < display_size; i++) {
            for (size_t j = 0; j < display_size; j++) {
                std::cout << matrix[i * m_matrixSize + j] << " ";
            }
            std::cout << std::endl;
        }
        std::cout << std::endl;
    }
    
    void printMatrices(size_t size = 5) {
        printMatrixSection(m_matrixA, "A", size);
        printMatrixSection(m_matrixB, "B", size);
        printMatrixSection(m_result, "Result", size);
    }

    void multiplyOnCPUOnly() {
        // Use existing computeOnCPU method with full matrix
        computeOnCPU(0, m_matrixSize);
    }
    
    void multiplyOnGPUOnly() {
        // Use existing computeOnGPU method with full matrix
        computeOnGPU(0, m_matrixSize);
    }
};

int main() {
    std::cout << "Heterogeneous Matrix Multiplication on Apple Silicon" << std::endl;
    
    //diff matrix sizes for more comprehensive comparison
    std::vector<uint32_t> sizes = {512, 1024, 2048, 4096, 8192, 16384};
    
    for (auto matrixSize : sizes) {
        std::cout << "\n---- Matrix size: " << matrixSize << "x" << matrixSize << " ----\n";
        
        //FUNCTION for dynamic workload distribution
        auto getCpuPercentage = [](uint32_t matrixSize) -> float {
            return (0.006f / (matrixSize / 1024.0f));
        };
        //ALTERNATIVE FUNCTION for dynamic workload distribution
        // auto getCpuPercentage = [](uint32_t matrixSize) -> float {
        //     // exponent in (0.5,1) trades off our decay speed!!!
        //     constexpr float alpha = 0.7f;
        //     //calibrated C so N=512 is like 1.2%
        //     constexpr float C = 0.9456f;
        //     return C / std::pow(static_cast<float>(matrixSize), alpha);
        // };
        
        float cpuPercentage = getCpuPercentage(matrixSize);
        float gpuPercentage = 1-cpuPercentage;
        // float npuPercentage = 0.0f;
        
        std::cout << "CPU percentage: " << cpuPercentage * 100 << "%\n";
        std::cout << "GPU percentage: " << gpuPercentage * 100 << "%\n";
        // std::cout << "NPU percentage: " << npuPercentage * 100 << "%\n";
        
        HeterogeneousMatrixMultiplier multiplier(matrixSize, cpuPercentage, gpuPercentage);
        if (!multiplier.initialize()) {
            std::cerr << "Failed to initialize matrix multiplier." << std::endl;
            continue;
        }
        
        // CPU only
        auto cpuStart = std::chrono::high_resolution_clock::now();
        multiplier.multiplyOnCPUOnly();
        auto cpuEnd = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> cpuDuration = cpuEnd - cpuStart;
        // std::chrono::duration<double> cpuDuration(0.0);
        // GPU only
        auto gpuStart = std::chrono::high_resolution_clock::now();
        multiplier.multiplyOnGPUOnly();
        auto gpuEnd = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> gpuDuration = gpuEnd - gpuStart;
        
        // Heterogeneous
        auto hetStart = std::chrono::high_resolution_clock::now();
        multiplier.multiply();
        auto hetEnd = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> hetDuration = hetEnd - hetStart;
        
        // Print results
        std::cout << "CPU-only time: " << cpuDuration.count() << " seconds\n";
        std::cout << "GPU-only time: " << gpuDuration.count() << " seconds\n";
        std::cout << "Heterogeneous time: " << hetDuration.count() << " seconds\n";
        
        // Calculate speedups
        float cpuSpeedup = cpuDuration.count() / hetDuration.count();
        float gpuSpeedup = gpuDuration.count() / hetDuration.count();
        
        std::cout << "Speedup vs CPU-only: " << cpuSpeedup << "x\n";
        std::cout << "Speedup vs GPU-only: " << gpuSpeedup << "x\n";
        
        // Report detailed processor timings
        multiplier.reportProcessorTimings();
    }
    
    return 0;
}
