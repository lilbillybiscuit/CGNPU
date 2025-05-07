#include "ane_executor.h"
#include <iostream>

ANEExecutor::ANEExecutor() : isANEAvailable(false) {
    std::cout << "DEBUG: Creating ANE executor (no-op version)" << std::endl;
}

ANEExecutor::~ANEExecutor() {
    std::cout << "DEBUG: Destroying ANE executor (no-op version)" << std::endl;
}

void ANEExecutor::initialize() {
    std::cout << "DEBUG: Initializing ANE executor (no-op version)" << std::endl;
    isANEAvailable = false;
    std::cout << "ANE support is completely disabled" << std::endl;
}

void ANEExecutor::execute(
    MatrixBuffer* a,
    MatrixBuffer* b,
    MatrixBuffer* result,
    std::shared_ptr<WorkScheduler> scheduler,
    std::shared_ptr<Profiler> profiler) {
    std::cout << "DEBUG: ANE executor starting (no-op mode)" << std::endl;
    auto& queue = scheduler->getQueue(DeviceType::ANE);
    std::unique_lock lock(queue.mutex);
    if (queue.activeWorkers > 0) {
        std::cout << "DEBUG: ANE executor resetting " << queue.activeWorkers 
                  << " active workers to 0" << std::endl;
        queue.activeWorkers = 0;
    }
    if (profiler) {
        std::cout << "DEBUG: ANE executor stats: processed 0 chunks" << std::endl;
        profiler->recordZeroTime("ane_execution");
    }
    std::cout << "DEBUG: ANE executor finished (no-op)" << std::endl;
}

void ANEExecutor::cpuExecuteChunk(
    MatrixBuffer* a,
    MatrixBuffer* b,
    MatrixBuffer* result,
    const WorkChunk& chunk) {
    std::cout << "DEBUG: ANE cpuExecuteChunk should never be called (no-op)" << std::endl;
}