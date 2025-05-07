#pragma once
#include "matrix_utils.h"
#include "work_stealing.h"
#include "profiler.h"
#include <memory>

class ANEExecutor {
public:
    ANEExecutor();
    ~ANEExecutor();
    void initialize();
    void execute(
        MatrixBuffer* a,
        MatrixBuffer* b,
        MatrixBuffer* result,
        std::shared_ptr<WorkScheduler> scheduler,
        std::shared_ptr<Profiler> profiler = nullptr);
private:
    bool isANEAvailable;
    void cpuExecuteChunk(
        MatrixBuffer* a,
        MatrixBuffer* b,
        MatrixBuffer* result,
        const WorkChunk& chunk);
};