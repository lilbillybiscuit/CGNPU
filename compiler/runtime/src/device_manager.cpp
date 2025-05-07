#include "device_manager.h"
#include <algorithm>
#include <iostream>
#include <iomanip>  
#include <thread>
#include <string>

DeviceManager::DeviceManager()
    : cpuExecutor(std::make_shared<CPUExecutor>())
    , gpuExecutor(std::make_shared<GPUExecutor>())
    , aneExecutor(std::make_shared<ANEExecutor>())
    , scheduler(std::make_shared<WorkScheduler>())
    , profiler(std::make_shared<Profiler>()) {
}

DeviceManager::~DeviceManager() {
}

void DeviceManager::initialize() {
    cpuExecutor->initialize();
    gpuExecutor->initialize();
    aneExecutor->initialize();
    scheduler->setProfiler(profiler);
    scheduler->initialize();
}

void DeviceManager::executeMatrixMultiplication(
    MatrixBuffer* a,
    MatrixBuffer* b,
    MatrixBuffer* result) {
    std::cout << "DEBUG: Starting device manager matrix multiplication" << std::endl;
    profiler->startTimer("total_execution");
    int matrixSize = a->size;
    std::cout << "DEBUG: Matrix size: " << matrixSize << "x" << matrixSize << std::endl;
    if (matrixSize >= 1024) {
        int* aData = a->getCPUReadPtr();
        int* bData = b->getCPUReadPtr();
        std::cout << "DEBUG: Matrix A (first few elements): ";
        for (int i = 0; i < std::min(5, matrixSize); i++) {
            std::cout << aData[i] << " ";
        }
        std::cout << std::endl;
        std::cout << "DEBUG: Matrix B (first few elements): ";
        for (int i = 0; i < std::min(5, matrixSize); i++) {
            std::cout << bData[i] << " ";
        }
        std::cout << std::endl;
        a->releaseCPUAccess();
        b->releaseCPUAccess();
    }
    int blockSize = 64;  
    if (matrixSize <= 128) {
        blockSize = 32;
        if (matrixSize % blockSize != 0) {
            while (matrixSize % blockSize != 0 && blockSize > 8) {
                blockSize -= 4;
            }
        }
        std::cout << "DEBUG: Using block size: " << blockSize 
                  << " for medium matrix multiplication" << std::endl;
    } 
    else if (matrixSize >= 2048) blockSize = 128;  
    else if (matrixSize >= 1024) blockSize = 128;  
    else if (matrixSize >= 512) blockSize = 96;    
    else if (matrixSize >= 256) blockSize = 64;    
    else blockSize = 64;
    std::cout << "DEBUG: Using block size: " << blockSize << std::endl;
    std::vector<WorkChunk> chunks;
    for (int i = 0; i < matrixSize; i += blockSize) {
        for (int j = 0; j < matrixSize; j += blockSize) {
            int endRow = std::min(i + blockSize, matrixSize);
            int endCol = std::min(j + blockSize, matrixSize);
            chunks.emplace_back(i, endRow, j, endCol);
        }
    }
    std::cout << "DEBUG: Created " << chunks.size() << " work chunks" << std::endl;
    std::vector<WorkChunk> cpuWork, gpuWork, aneWork;
    partitionWork(chunks, cpuWork, gpuWork, aneWork);
    std::cout << "DEBUG: Work distribution - CPU: " << cpuWork.size() 
              << ", GPU: " << gpuWork.size() 
              << ", ANE: " << aneWork.size() << std::endl;
    std::cout << "DEBUG: Adding work to scheduler" << std::endl;
    scheduler->getQueue(DeviceType::CPU).allocatedChunks = cpuWork.size();
    scheduler->getQueue(DeviceType::GPU).allocatedChunks = gpuWork.size();
    scheduler->getQueue(DeviceType::ANE).allocatedChunks = aneWork.size();
    scheduler->addWork(cpuWork, DeviceType::CPU);
    scheduler->addWork(gpuWork, DeviceType::GPU);
    scheduler->addWork(aneWork, DeviceType::ANE);
    std::cout << "DEBUG: Starting device executor threads" << std::endl;
    std::thread cpuThread([this, a, b, result]() {
        std::cout << "DEBUG: Starting CPU execution thread" << std::endl;
        bool hasWork = false;
        {
            auto& queue = scheduler->getQueue(DeviceType::CPU);
            std::lock_guard lock(queue.mutex);
            hasWork = !queue.queue.empty();
        }
        if (hasWork) {
            profiler->startTimer("cpu_execution");
            cpuExecutor->execute(a, b, result, scheduler, profiler);
            profiler->stopTimer("cpu_execution");
        } else {
            cpuExecutor->execute(a, b, result, scheduler, profiler);
            profiler->recordZeroTime("cpu_execution");
        }
    });
    std::thread gpuThread([this, a, b, result]() {
        std::cout << "DEBUG: Starting GPU execution thread" << std::endl;
        bool hasWork = false;
        {
            auto& queue = scheduler->getQueue(DeviceType::GPU);
            std::lock_guard lock(queue.mutex);
            hasWork = !queue.queue.empty();
        }
        if (hasWork) {
            profiler->startTimer("gpu_execution");
            gpuExecutor->execute(a, b, result, scheduler, profiler);
            profiler->stopTimer("gpu_execution");
        } else {
            gpuExecutor->execute(a, b, result, scheduler, profiler);
            profiler->recordZeroTime("gpu_execution");
        }
    });
    std::thread aneThread([this, a, b, result]() {
        std::cout << "DEBUG: Starting ANE execution thread" << std::endl;
        bool hasWork = false;
        {
            auto& queue = scheduler->getQueue(DeviceType::ANE);
            std::lock_guard lock(queue.mutex);
            hasWork = !queue.queue.empty();
        }
        if (hasWork) {
            profiler->startTimer("ane_execution");
            aneExecutor->execute(a, b, result, scheduler, profiler);
            profiler->stopTimer("ane_execution");
        } else {
            aneExecutor->execute(a, b, result, scheduler, profiler);
            profiler->recordZeroTime("ane_execution");
        }
    });
    std::cout << "DEBUG: Waiting for device threads to complete" << std::endl;
    cpuThread.join();
    scheduler->cpuThreadExited = true;
    gpuThread.join();
    scheduler->gpuThreadExited = true;
    aneThread.join();
    scheduler->aneThreadExited = true;
    std::cout << "DEBUG: All execution threads joined, waiting for completion" << std::endl;
    waitForCompletion();
    profiler->stopTimer("total_execution");
    profiler->printReport();

    if (a->size >= 1024) {
        int* resultData = result->getCPUReadPtr();
        std::cout << "DEBUG: Result matrix (first few elements): ";
        for (int i = 0; i < std::min(5, a->size); i++) {
            std::cout << resultData[i] << " ";
        }
        std::cout << std::endl;
        int nonZeroCount = 0;
        int totalChecked = 0;
        for (int region = 0; region < 4; region++) {
            int startIdx = (a->size * a->size / 4) * region;
            for (int i = 0; i < 10 && (startIdx + i) < (a->size * a->size); i++) {
                if (resultData[startIdx + i] != 0) {
                    nonZeroCount++;
                }
                totalChecked++;
            }
        }
        result->releaseCPUAccess();
        std::cout << "DEBUG: Result matrix sampling: " << nonZeroCount << " non-zero values out of " 
                  << totalChecked << " sampled" << std::endl;
        if (nonZeroCount == 0) {
            std::cout << "WARNING: Result matrix appears to contain all zeros in sampled regions!" << std::endl;
        }
    }
    std::cout << "DEBUG: Matrix multiplication completed" << std::endl;
}

void DeviceManager::waitForCompletion() {
    try {
        scheduler->waitForCompletion();
    } catch (const std::exception& e) {
        std::cerr << "ERROR in waitForCompletion: " << e.what() << std::endl;
    }
}

void DeviceManager::partitionWork(
    const std::vector<WorkChunk>& chunks,
    std::vector<WorkChunk>& cpuWork,
    std::vector<WorkChunk>& gpuWork,
    std::vector<WorkChunk>& aneWork) {
    int totalChunks = chunks.size();
    std::cout << "DEBUG: Partitioning " << totalChunks << " work chunks" << std::endl;
    const char* gpuOnlyEnv = std::getenv("GPU_ONLY");
    bool gpuOnly = (gpuOnlyEnv != nullptr);
    int gpuAllocation, cpuAllocation, aneAllocation;
    aneAllocation = 0;
    std::cout << "DEBUG: ANE is disabled in this implementation" << std::endl;
    if (gpuOnly) {
        gpuAllocation = totalChunks;
        cpuAllocation = 0;
        aneAllocation = 0;
        std::cout << "DEBUG: GPU_ONLY mode enabled: 100% GPU execution, work stealing disabled" << std::endl;
        if (profiler) {
            profiler->disableWorkStealing();
        }
    } else {
        const char* distributionEnv = std::getenv("DISTRIBUTION");
        int gpuPercent = 65;  
        if (distributionEnv != nullptr) {
            try {
                gpuPercent = std::stoi(distributionEnv);
                if (gpuPercent < 0 || gpuPercent > 100) {
                    std::cout << "WARNING: Invalid DISTRIBUTION value, using default 80% GPU" << std::endl;
                    gpuPercent = 80;
                }
            } catch (...) {
                std::cout << "WARNING: Invalid DISTRIBUTION value, using default 80% GPU" << std::endl;
            }
        }
        gpuAllocation = (int)(totalChunks * (gpuPercent / 100.0)); 
        cpuAllocation = totalChunks - gpuAllocation;
        aneAllocation = 0;  
        std::cout << "DEBUG: Using " << gpuPercent << "/" << (100-gpuPercent) 
                  << " GPU/CPU distribution" << std::endl;
    }
    std::cout << "DEBUG: Using distribution - CPU: " << cpuAllocation 
              << " (" << (100.0 * cpuAllocation / totalChunks) << "%)"
              << ", GPU: " << gpuAllocation 
              << " (" << (100.0 * gpuAllocation / totalChunks) << "%)"
              << ", ANE: " << aneAllocation
              << " (" << (100.0 * aneAllocation / totalChunks) << "%)" << std::endl;
    profiler->recordInitialAllocation("CPU", cpuAllocation, totalChunks);
    profiler->recordInitialAllocation("GPU", gpuAllocation, totalChunks);
    profiler->recordInitialAllocation("ANE", aneAllocation, totalChunks);
    for (int i = 0; i < chunks.size(); i++) {
        if (i < cpuAllocation) cpuWork.push_back(chunks[i]);
        else if (i < cpuAllocation + gpuAllocation) gpuWork.push_back(chunks[i]);
        else aneWork.push_back(chunks[i]);
    }
    return;
    if (totalChunks >= 4) {
        int cpuCount;
        int gpuCount;
        int aneCount;
        int matrixSize = 0;
        if (!chunks.empty()) {
            const WorkChunk& firstChunk = chunks[0];
            if (firstChunk.startRow == 0 && firstChunk.startCol == 0) {
                matrixSize = std::max(firstChunk.endRow * 4, firstChunk.endCol * 4);
            } else {
                matrixSize = std::max(firstChunk.endRow, firstChunk.endCol);
            }
        }
        if (matrixSize <= 128) {
            cpuCount = (int)(totalChunks * 0.2);   
            gpuCount = (int)(totalChunks * 0.8);   
            aneCount = 0;  
            if (totalChunks >= 5) {
                if (cpuCount < 1) cpuCount = 1;
                if (gpuCount < 1) gpuCount = 1;
            } else {
                cpuCount = (int)(totalChunks * 0.2);
                if (cpuCount < 1) cpuCount = 1;
                gpuCount = totalChunks - cpuCount;
                aneCount = 0;  
            }
            std::cout << "DEBUG: Medium matrix detected, using balanced distribution" << std::endl;
        }
        else if (totalChunks <= 4) {
            cpuCount = std::max(1, totalChunks - 2);
            gpuCount = totalChunks > 1 ? 1 : 0;
            aneCount = totalChunks > 2 ? 1 : 0;
        } else if (totalChunks <= 16) {
            cpuCount = (int)(totalChunks * 0.6);   
            gpuCount = (int)(totalChunks * 0.3);   
            aneCount = totalChunks - cpuCount - gpuCount;  
        } else if (totalChunks <= 64) {
            cpuCount = (int)(totalChunks * 0.2);   
            gpuCount = (int)(totalChunks * 0.8);   
            aneCount = 0;  
        } else if (totalChunks <= 256) {
            int gpuAllocation = (int)(totalChunks * 0.8);   
            int aneAllocation = 0;    
            cpuCount = totalChunks - gpuAllocation - aneAllocation;   
            gpuCount = gpuAllocation;
            aneCount = aneAllocation;
        } else {
            int gpuAllocation = (int)(totalChunks * 0.95);  
            int aneAllocation = 0;   
            cpuCount = totalChunks - gpuAllocation - aneAllocation;  
            gpuCount = gpuAllocation;
            aneCount = aneAllocation;
        }

        profiler->recordInitialAllocation("CPU", cpuCount, totalChunks);
        profiler->recordInitialAllocation("GPU", gpuCount, totalChunks);
        profiler->recordInitialAllocation("ANE", aneCount, totalChunks);

        for (int i = 0; i < chunks.size(); i++) {
            if (i < cpuCount) cpuWork.push_back(chunks[i]);
            else if (i < cpuCount + gpuCount) gpuWork.push_back(chunks[i]);
            else aneWork.push_back(chunks[i]);
        }
    }

    if (totalChunks <= 3) {
        int gpuCount = (int)(totalChunks * 0.8);
        int cpuCount = totalChunks - gpuCount;
        int aneCount = 0;
        profiler->recordInitialAllocation("CPU", cpuCount, totalChunks);
        profiler->recordInitialAllocation("GPU", gpuCount, totalChunks);
        profiler->recordInitialAllocation("ANE", aneCount, totalChunks);
        for (int i = 0; i < chunks.size(); i++) {
            if (i < gpuCount) gpuWork.push_back(chunks[i]);
            else cpuWork.push_back(chunks[i]);
        }
        return;
    }

    int cpuCount = (int)(totalChunks * 0.2);
    int gpuCount = (int)(totalChunks * 0.8);
    int aneCount = 0;  
    if (cpuCount < 1) cpuCount = 1;
    if (gpuCount < 1) gpuCount = 1;
    int total = cpuCount + gpuCount;
    if (total > totalChunks) {
        int excess = total - totalChunks;
        if (gpuCount >= cpuCount) {
            gpuCount -= excess;
        } else {
            cpuCount -= excess;
        }
    }

    std::cout << "DEBUG: Standard distribution - CPU: " << cpuCount 
              << ", GPU: " << gpuCount 
              << ", ANE: " << aneCount << std::endl;
    profiler->recordInitialAllocation("CPU", cpuCount, totalChunks);
    profiler->recordInitialAllocation("GPU", gpuCount, totalChunks);
    profiler->recordInitialAllocation("ANE", aneCount, totalChunks);
    for (int i = 0; i < chunks.size(); i++) {
        if (i < cpuCount) cpuWork.push_back(chunks[i]);
        else gpuWork.push_back(chunks[i]);
    }
}