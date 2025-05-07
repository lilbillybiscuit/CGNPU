#include "gpu_executor.h"
#include "metal_buffer_wrapper.h"
#include <Metal/Metal.h>
#include <iostream>
struct GPUExecutor::Impl {
    id<MTLDevice> device;
    id<MTLCommandQueue> commandQueue;
    id<MTLLibrary> library;
    id<MTLComputePipelineState> pipelineState;
};
GPUExecutor::GPUExecutor() : pImpl(new Impl()) {
}
GPUExecutor::~GPUExecutor() {
    delete pImpl;
}
void GPUExecutor::initialize() {
    @autoreleasepool {
        pImpl->device = MTLCreateSystemDefaultDevice();
        if (!pImpl->device) {
            std::cerr << "Metal is not supported on this device" << std::endl;
            return;
        }
        pImpl->commandQueue = [pImpl->device newCommandQueue];
        NSError* error = nil;
        NSString* libraryPath = @"matrix_mult.metallib";
        pImpl->library = [pImpl->device newLibraryWithFile:libraryPath error:&error];
        if (!pImpl->library) {
            NSBundle* mainBundle = [NSBundle mainBundle];
            NSString* bundlePath = [mainBundle resourcePath];
            libraryPath = [bundlePath stringByAppendingPathComponent:@"matrix_mult.metallib"];
            pImpl->library = [pImpl->device newLibraryWithFile:libraryPath error:&error];
        }
        if (!pImpl->library) {
            std::cerr << "Failed to load Metal library: " << error.localizedDescription.UTF8String << std::endl;
            std::cerr << "Looked for: matrix_mult.metallib" << std::endl;
            return;
        }
        id<MTLFunction> kernelFunction = [pImpl->library newFunctionWithName:@"matrix_multiply"];
        if (!kernelFunction) {
            std::cerr << "Failed to find matrix_multiply function in library" << std::endl;
            return;
        }
        pImpl->pipelineState = [pImpl->device newComputePipelineStateWithFunction:kernelFunction error:&error];
        if (!pImpl->pipelineState) {
            std::cerr << "Failed to create pipeline state: " << error.localizedDescription.UTF8String << std::endl;
        }
    }
}
void GPUExecutor::execute(
    MatrixBuffer* a,
    MatrixBuffer* b,
    MatrixBuffer* result,
    std::shared_ptr<WorkScheduler> scheduler,
    std::shared_ptr<Profiler> profiler) {
    std::cout << "DEBUG: GPU executor starting" << std::endl;
    if (!pImpl->device || !pImpl->commandQueue || !pImpl->pipelineState) {
        std::cout << "DEBUG: GPU Metal setup incomplete, cannot execute" << std::endl;
        return;
    }
    @autoreleasepool {
        std::cout << "DEBUG: GPU processing chunks" << std::endl;
        int processedChunks = 0;
        a->prepareForGPUAccess(true);     
        b->prepareForGPUAccess(true);     
        result->prepareForGPUAccess(false);  
        void* aBuffer = a->metalBuffer->getMetalBuffer();
        void* bBuffer = b->metalBuffer->getMetalBuffer();
        void* rBuffer = result->metalBuffer->getMetalBuffer();
        if (!aBuffer || !bBuffer || !rBuffer) {
            std::cerr << "Failed to access GPU buffers" << std::endl;
            return;
        }
        id<MTLCommandBuffer> commandBuffer = [pImpl->commandQueue commandBuffer];
        if (!commandBuffer) {
            std::cerr << "Failed to create Metal command buffer" << std::endl;
            return;
        }
        [commandBuffer addCompletedHandler:^(id<MTLCommandBuffer> buffer) {
            if (buffer.status == MTLCommandBufferStatusError) {
                NSError *error = buffer.error;
                std::cerr << "GPU execution error: " << error.localizedDescription.UTF8String << std::endl;
            }
        }];
        WorkChunk* chunk;
        while ((chunk = scheduler->getWork(DeviceType::GPU))) {
            std::cout << "DEBUG: GPU processing chunk [" << chunk->startRow << ":" << chunk->endRow 
                      << ", " << chunk->startCol << ":" << chunk->endCol << "]" << std::endl;
            try {
                int chunkSize = (chunk->endRow - chunk->startRow) * 
                              (chunk->endCol - chunk->startCol);
                id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
                if (!encoder) {
                    std::cerr << "Failed to create compute encoder" << std::endl;
                    delete chunk;
                    continue;
                }
                [encoder setComputePipelineState:pImpl->pipelineState];
                [encoder setBuffer:(__bridge id<MTLBuffer>)aBuffer offset:0 atIndex:0];
                [encoder setBuffer:(__bridge id<MTLBuffer>)bBuffer offset:0 atIndex:1];
                [encoder setBuffer:(__bridge id<MTLBuffer>)rBuffer offset:0 atIndex:2];
                struct ChunkInfo {
                    int startRow;
                    int endRow;
                    int startCol;
                    int endCol;
                    int size;
                } chunkInfo = {chunk->startRow, chunk->endRow, chunk->startCol, chunk->endCol, a->size};
                id<MTLBuffer> chunkBuffer = [pImpl->device newBufferWithBytes:&chunkInfo
                                                             length:sizeof(ChunkInfo)
                                                            options:MTLResourceStorageModeShared];
                [encoder setBuffer:chunkBuffer offset:0 atIndex:3];
                const int TILE_SIZE = 32;
                const int VECTOR_SIZE = 4;
                int threadgroupWidth = TILE_SIZE / VECTOR_SIZE;
                int threadgroupHeight = TILE_SIZE / VECTOR_SIZE;
                MTLSize threadgroupSize = MTLSizeMake(threadgroupWidth, threadgroupHeight, 1);
                int numRowTiles = (chunk->endRow - chunk->startRow + TILE_SIZE - 1) / TILE_SIZE;
                int numColTiles = (chunk->endCol - chunk->startCol + TILE_SIZE - 1) / TILE_SIZE;
                MTLSize gridSize = MTLSizeMake(numRowTiles * threadgroupWidth, 
                                              numColTiles * threadgroupHeight, 1);
                const int maxGridDimension = 1024;
                if (gridSize.width > maxGridDimension || gridSize.height > maxGridDimension) {
                    float scaleX = (float)maxGridDimension / gridSize.width;
                    float scaleY = (float)maxGridDimension / gridSize.height;
                    float scale = std::min(scaleX, scaleY);
                    gridSize.width = std::max(1u, (uint)std::floor(gridSize.width * scale));
                    gridSize.height = std::max(1u, (uint)std::floor(gridSize.height * scale));
                }
                [encoder dispatchThreads:gridSize threadsPerThreadgroup:threadgroupSize];
                [encoder endEncoding];
                id<MTLCommandBuffer> chunkCommandBuffer = [pImpl->commandQueue commandBuffer];
                id<MTLComputeCommandEncoder> timeEncoder = [chunkCommandBuffer computeCommandEncoder];
                [timeEncoder setComputePipelineState:pImpl->pipelineState];
                [timeEncoder setBuffer:(__bridge id<MTLBuffer>)aBuffer offset:0 atIndex:0];
                [timeEncoder setBuffer:(__bridge id<MTLBuffer>)bBuffer offset:0 atIndex:1];
                [timeEncoder setBuffer:(__bridge id<MTLBuffer>)rBuffer offset:0 atIndex:2];
                [timeEncoder setBuffer:chunkBuffer offset:0 atIndex:3];
                [timeEncoder dispatchThreads:gridSize threadsPerThreadgroup:threadgroupSize];
                [timeEncoder endEncoding];
                auto startTime = std::chrono::steady_clock::now();
                [chunkCommandBuffer commit];
                [chunkCommandBuffer waitUntilCompleted];
                auto endTime = std::chrono::steady_clock::now();
                double seconds = std::chrono::duration_cast<std::chrono::microseconds>(
                    endTime - startTime).count() / 1000000.0;
                if (profiler) {
                    profiler->recordChunkExecution("GPU", chunkSize);
                }
                scheduler->recordChunkProcessingTime(DeviceType::GPU, seconds);
                processedChunks++;
            } catch (const std::exception& e) {
                std::cerr << "GPU chunk processing failed: " << e.what() << std::endl;
            }
            delete chunk;
        }
        if (processedChunks > 0) {
            std::cout << "DEBUG: GPU committing command buffer with all chunks" << std::endl;
            [commandBuffer commit];
            [commandBuffer waitUntilCompleted];
        } else {
            std::cout << "DEBUG: GPU skipping command buffer commit (no chunks processed)" << std::endl;
        }
        a->releaseGPUAccess();
        b->releaseGPUAccess();
        result->releaseGPUAccess();
        auto& queue = scheduler->getQueue(DeviceType::GPU);
        std::unique_lock<std::mutex> lock(queue.mutex);
        if (queue.activeWorkers > 0) {
            std::cout << "DEBUG: GPU executor resetting " << queue.activeWorkers 
                      << " active workers to 0" << std::endl;
            queue.activeWorkers = 0;
        }
        if (profiler) {
            std::cout << "DEBUG: GPU executor stats: processed " << processedChunks << " chunks" << std::endl;
        }
        std::cout << "DEBUG: GPU executor finished after processing " << processedChunks << " chunks" << std::endl;
    }
}