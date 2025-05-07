#include "cpu_executor.h"
#include <vector>
#include <thread>
#include <iostream>

CPUExecutor::CPUExecutor(): numThreads(0) {
}

CPUExecutor::~CPUExecutor() = default;

void CPUExecutor::initialize() {
    numThreads = std::thread::hardware_concurrency();
    #if defined(__APPLE__) && defined(__arm64__)
        if (numThreads >= 8) {
            numThreads = 5;   
        } else if (numThreads >= 6) {
            numThreads = 4;   
        } else {
            numThreads = std::max(1, numThreads - 1);
        }
    #else
        numThreads = std::max(1, numThreads - 2);
    #endif
    std::cout << "DEBUG: CPU executor initialized with " << numThreads << " threads" << std::endl;
}

void CPUExecutor::execute(
    MatrixBuffer* a,
    MatrixBuffer* b,
    MatrixBuffer* result,
    std::shared_ptr<WorkScheduler> scheduler,
    std::shared_ptr<Profiler> profiler) {
    std::vector<std::thread> threads;
    std::cout << "DEBUG: CPU executor starting with " << numThreads << " threads" << std::endl;

    const char* gpuOnlyEnv = std::getenv("GPU_ONLY");
    bool gpuOnly = (gpuOnlyEnv != nullptr);
    if (gpuOnly) {
        std::cout << "DEBUG: CPU executor skipping work stealing (GPU_ONLY mode enabled)" << std::endl;
    } else {
        std::cout << "DEBUG: CPU executor doing minimal work stealing for demo purposes" << std::endl;
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        int targetSteals = 50;   
        {
            auto& gpuQueue = scheduler->getQueue(DeviceType::GPU);
            std::lock_guard<std::mutex> lock(gpuQueue.mutex);
            int queueSize = gpuQueue.queue.size();
            if (queueSize < 20) {
                targetSteals = 2;   
            } else if (queueSize < 100) {
                targetSteals = 10;   
            } else if (queueSize < 500) {
                targetSteals = 30;
            } else {
                targetSteals = 100;
            }
        }
        int successfulSteals = 0;
        for (int attempt = 0; attempt < 30 && successfulSteals < targetSteals; attempt++) {
            DeviceType fromDevice = DeviceType::GPU;
            DeviceType toDevice = DeviceType::CPU;
            WorkChunk* stolen = scheduler->steal(fromDevice, toDevice);
            if (stolen) {
                if (profiler) {
                    profiler->recordStealEvent("GPU", "CPU");
                }
                std::vector<WorkChunk> stolenWork;
                stolenWork.push_back(*stolen);
                scheduler->addWork(stolenWork, toDevice);
                delete stolen;
                successfulSteals++;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
        }
        std::cout << "DEBUG: CPU stole " << successfulSteals << " chunks from GPU" << std::endl;
    }

    for (int i = 0; i < numThreads; i++) {
        threads.emplace_back([this, a, b, result, scheduler, i, profiler]() {
            std::cout << "DEBUG: CPU worker thread " << i << " started" << std::endl;
            WorkChunk* chunk;
            while ((chunk = scheduler->getWork(DeviceType::CPU))) {
                std::cout << "DEBUG: CPU worker " << i << " processing chunk [" 
                          << chunk->startRow << ":" << chunk->endRow << ", "
                          << chunk->startCol << ":" << chunk->endCol << "]" << std::endl;
                int chunkSize = (chunk->endRow - chunk->startRow) *
                              (chunk->endCol - chunk->startCol);
                auto startTime = std::chrono::steady_clock::now();
                executeChunk(a, b, result, *chunk);
                auto endTime = std::chrono::steady_clock::now();
                double seconds = std::chrono::duration_cast<std::chrono::microseconds>(
                    endTime - startTime).count() / 1000000.0;
                if (profiler) {
                    profiler->recordChunkExecution("CPU", chunkSize);
                }
                scheduler->recordChunkProcessingTime(DeviceType::CPU, seconds);
                std::cout << "DEBUG: CPU worker " << i << " finished processing chunk" << std::endl;
                delete chunk;
            }
            std::cout << "DEBUG: CPU worker thread " << i << " exiting" << std::endl;
        });
    }

    for (auto& thread : threads) {
        thread.join();
    }

    auto& queue = scheduler->getQueue(DeviceType::CPU);
    std::unique_lock lock(queue.mutex);
    if (queue.activeWorkers > 0) {
        std::cout << "DEBUG: CPU executor resetting " << queue.activeWorkers 
                  << " active workers to 0" << std::endl;
        queue.activeWorkers = 0;
    }

    if (profiler) {
        std::cout << "DEBUG: CPU executor using profiler data" << std::endl;
    }

    std::cout << "DEBUG: CPU executor finished" << std::endl;
}
void CPUExecutor::executeChunk(
    MatrixBuffer* a,
    MatrixBuffer* b,
    MatrixBuffer* result,
    const WorkChunk& chunk) {
    int* aData = a->getCPUReadPtr();   
    int* bData = b->getCPUReadPtr();   
    int* rData = result->getCPUWritePtr();  
    int size = a->size;
    int BLOCK_SIZE = 32;  
    if (size >= 2048) BLOCK_SIZE = 32;   
    else if (size >= 1024) BLOCK_SIZE = 48;
    else if (size >= 512) BLOCK_SIZE = 32;
    else if (size < 128) BLOCK_SIZE = 16;
    std::cout << "DEBUG: CPU executor using BLOCK_SIZE: " << BLOCK_SIZE 
              << " for matrix size: " << size << std::endl;
    if (size <= 128) {
        const int MINI_BLOCK = 8;  
        for (int i = chunk.startRow; i < chunk.endRow; i += MINI_BLOCK) {
            int iEnd = std::min(i + MINI_BLOCK, chunk.endRow);
            for (int j = chunk.startCol; j < chunk.endCol; j += MINI_BLOCK) {
                int jEnd = std::min(j + MINI_BLOCK, chunk.endCol);
                for (int ii = i; ii < iEnd; ii++) {
                    for (int jj = j; jj < jEnd; jj++) {
                        rData[ii * size + jj] = 0;
                    }
                }
                for (int k = 0; k < size; k += MINI_BLOCK) {
                    int kEnd = std::min(k + MINI_BLOCK, size);
                    for (int ii = i; ii < iEnd; ii++) {
                        for (int jj = j; jj < jEnd; jj++) {
                            long long sum = rData[ii * size + jj];
                            for (int kk = k; kk < kEnd; kk++) {
                                sum += (long long)aData[ii * size + kk] * (long long)bData[kk * size + jj];
                            }
                            rData[ii * size + jj] = (int)sum;
                        }
                    }
                }
            }
        }
        a->releaseCPUAccess();
        b->releaseCPUAccess();
        result->releaseCPUAccess();
        return;
    }

    for (int i = chunk.startRow; i < chunk.endRow; i++) {
        for (int j = chunk.startCol; j < chunk.endCol; j++) {
            rData[i * size + j] = 0;
        }
    }

    if (size >= 1024) {
        int blockAccum[BLOCK_SIZE][BLOCK_SIZE];
        for (int kk = 0; kk < size; kk += BLOCK_SIZE) {
            int kEnd = std::min(kk + BLOCK_SIZE, size);
            for (int ii = chunk.startRow; ii < chunk.endRow; ii += BLOCK_SIZE) {
                int iEnd = std::min(ii + BLOCK_SIZE, chunk.endRow);
                for (int i = ii; i < iEnd; i++) {
                    for (int kb = kk; kb < kEnd; kb += 64) {
                        __builtin_prefetch(&aData[i * size + kb], 0, 3);
                        if (kb + 32 < kEnd) {
                            __builtin_prefetch(&aData[i * size + kb + 32], 0, 3);
                        }
                    }
                }
                for (int jj = chunk.startCol; jj < chunk.endCol; jj += BLOCK_SIZE) {
                    int jEnd = std::min(jj + BLOCK_SIZE, chunk.endCol);
                    for (int i = 0; i < BLOCK_SIZE; i++) {
                        for (int j = 0; j < BLOCK_SIZE; j++) {
                            blockAccum[i][j] = 0;
                        }
                    }
                    for (int k = kk; k < kEnd; k++) {
                        if (k + 1 < kEnd) {
                            __builtin_prefetch(&bData[(k + 1) * size + jj], 0, 3);
                            if (jj + 32 < jEnd) {
                                __builtin_prefetch(&bData[(k + 1) * size + jj + 32], 0, 3);
                            }
                        }
                        for (int i = ii; i < iEnd; i++) {
                            int aVal = aData[i * size + k];
                            if (aVal == 0) continue;
                            int j = jj;
                            for (; j + 7 < jEnd; j += 8) {
                                blockAccum[i - ii][j - jj] += aVal * bData[k * size + j];
                                blockAccum[i - ii][j - jj + 1] += aVal * bData[k * size + j + 1];
                                blockAccum[i - ii][j - jj + 2] += aVal * bData[k * size + j + 2];
                                blockAccum[i - ii][j - jj + 3] += aVal * bData[k * size + j + 3];
                                blockAccum[i - ii][j - jj + 4] += aVal * bData[k * size + j + 4];
                                blockAccum[i - ii][j - jj + 5] += aVal * bData[k * size + j + 5];
                                blockAccum[i - ii][j - jj + 6] += aVal * bData[k * size + j + 6];
                                blockAccum[i - ii][j - jj + 7] += aVal * bData[k * size + j + 7];
                            }
                            for (; j < jEnd; j++) {
                                blockAccum[i - ii][j - jj] += aVal * bData[k * size + j];
                            }
                        }
                    }
                    for (int i = ii; i < iEnd; i++) {
                        for (int j = jj; j < jEnd; j++) {
                            rData[i * size + j] += blockAccum[i - ii][j - jj];
                        }
                    }
                }
            }
        }
    } else {
        for (int ii = chunk.startRow; ii < chunk.endRow; ii += BLOCK_SIZE) {
            int iEnd = std::min(ii + BLOCK_SIZE, chunk.endRow);
            for (int jj = chunk.startCol; jj < chunk.endCol; jj += BLOCK_SIZE) {
                int jEnd = std::min(jj + BLOCK_SIZE, chunk.endCol);
                for (int kk = 0; kk < size; kk += BLOCK_SIZE) {
                    int kEnd = std::min(kk + BLOCK_SIZE, size);
                    for (int i = ii; i < iEnd; i++) {
                        if (i + 1 < iEnd) {
                            __builtin_prefetch(&aData[(i + 1) * size + kk], 0, 3);
                        }
                        for (int k = kk; k < kEnd; k++) {
                            int aVal = aData[i * size + k];
                            if (aVal == 0) continue;
                            if (k + 1 < kEnd) {
                                __builtin_prefetch(&bData[(k + 1) * size + jj], 0, 3);
                            }
                            int j = jj;
                            for (; j + 7 < jEnd; j += 8) {
                                rData[i * size + j] += aVal * bData[k * size + j];
                                rData[i * size + j+1] += aVal * bData[k * size + j+1];
                                rData[i * size + j+2] += aVal * bData[k * size + j+2];
                                rData[i * size + j+3] += aVal * bData[k * size + j+3];
                                rData[i * size + j+4] += aVal * bData[k * size + j+4];
                                rData[i * size + j+5] += aVal * bData[k * size + j+5];
                                rData[i * size + j+6] += aVal * bData[k * size + j+6];
                                rData[i * size + j+7] += aVal * bData[k * size + j+7];
                            }
                            for (; j < jEnd; j++) {
                                rData[i * size + j] += aVal * bData[k * size + j];
                            }
                        }
                    }
                }
            }
        }
    }

    a->releaseCPUAccess();
    b->releaseCPUAccess();
    result->releaseCPUAccess();
}