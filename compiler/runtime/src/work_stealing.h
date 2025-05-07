#pragma once
#include <queue>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <chrono>
#include "matrix_utils.h"
#include "profiler.h"

enum class DeviceType {
    CPU,
    GPU,
    ANE
};

class WorkStealingScheduler {
public:
    struct DeviceQueue {
        std::queue<WorkChunk> queue;
        std::mutex mutex;
        std::condition_variable cv;
        std::atomic<int> activeWorkers;
        std::chrono::steady_clock::time_point lastWorkTime;   
        double avgProcessingTime;   
        int chunksProcessed;
        int allocatedChunks;     
    };
    WorkStealingScheduler();
    ~WorkStealingScheduler();
    void setProfiler(std::shared_ptr<Profiler> profiler) { this->profiler = profiler; }
    void initialize();
    void addWork(const std::vector<WorkChunk>& chunks, DeviceType device);
    WorkChunk* getWork(DeviceType device);
    bool hasWork(DeviceType device);
    DeviceQueue& getQueue(DeviceType device);
    void waitForCompletion();
    std::atomic<bool> cpuThreadExited{false};
    std::atomic<bool> gpuThreadExited{false};
    std::atomic<bool> aneThreadExited{false};
    void recordChunkProcessingTime(DeviceType device, double seconds);
    WorkChunk* steal(DeviceType fromDevice, DeviceType toDevice);
    DeviceType selectDeviceToStealFrom(DeviceType idleDevice);
private:
    DeviceQueue queues[3];  
    std::atomic<int> totalWork;
    std::atomic<bool> shutdownRequested;
    std::atomic<bool> monitorActive;
    void monitor();
    std::shared_ptr<Profiler> profiler;
    std::string getDeviceName(DeviceType device);
    std::atomic<int64_t> lastCpuWorkTime;
    std::atomic<int64_t> lastGpuWorkTime;
    std::atomic<int64_t> lastAneWorkTime;
};
using WorkScheduler = WorkStealingScheduler;