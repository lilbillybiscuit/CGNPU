#pragma once
#include "matrix_utils.h"
#include "work_stealing.h"
#include "profiler.h"
#include <memory>

class GPUExecutor {
public:
    GPUExecutor();
    ~GPUExecutor();
    void initialize();
    void execute(
        MatrixBuffer* a,
        MatrixBuffer* b,
        MatrixBuffer* result,
        std::shared_ptr<WorkScheduler> scheduler,
        std::shared_ptr<Profiler> profiler = nullptr);
private:
    struct Impl;
    Impl* pImpl;
};