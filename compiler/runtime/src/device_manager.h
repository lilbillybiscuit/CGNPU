#pragma once
#include <memory>
#include <vector>
#include "matrix_utils.h"
#include "cpu_executor.h"
#include "gpu_executor.h"
#include "ane_executor.h"
#include "work_stealing.h"  
#include "profiler.h"

class DeviceManager {
public:
    DeviceManager();
    ~DeviceManager();
    void initialize();
    void executeMatrixMultiplication(
        MatrixBuffer* a,
        MatrixBuffer* b,
        MatrixBuffer* result);
    void waitForCompletion();
    std::shared_ptr<CPUExecutor> getCPUExecutor() { return cpuExecutor; }
    std::shared_ptr<GPUExecutor> getGPUExecutor() { return gpuExecutor; }
    std::shared_ptr<ANEExecutor> getANEExecutor() { return aneExecutor; }
    std::shared_ptr<WorkScheduler> getScheduler() { return scheduler; }
    std::shared_ptr<Profiler> getProfiler() { return profiler; }
private:
    std::shared_ptr<CPUExecutor> cpuExecutor;
    std::shared_ptr<GPUExecutor> gpuExecutor;
    std::shared_ptr<ANEExecutor> aneExecutor;
    std::shared_ptr<WorkScheduler> scheduler;
    std::shared_ptr<Profiler> profiler;
    void partitionWork(
        const std::vector<WorkChunk>& chunks,
        std::vector<WorkChunk>& cpuWork,
        std::vector<WorkChunk>& gpuWork,
        std::vector<WorkChunk>& aneWork);
};