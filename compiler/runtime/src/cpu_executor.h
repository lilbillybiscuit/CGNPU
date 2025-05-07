#pragma once
#include "matrix_utils.h"
#include "work_stealing.h"
#include "profiler.h"
#include <memory>

class CPUExecutor {
public:
    CPUExecutor();
    ~CPUExecutor();
    void initialize();
    void execute(
        MatrixBuffer* a,
        MatrixBuffer* b,
        MatrixBuffer* result,
        std::shared_ptr<WorkScheduler> scheduler,
        std::shared_ptr<Profiler> profiler = nullptr);
private:
    int numThreads;
    void executeChunk(
        MatrixBuffer* a,
        MatrixBuffer* b,
        MatrixBuffer* result,
        const WorkChunk& chunk);
};