#include "work_stealing.h"
#include <chrono>
#include <algorithm>
#include <iostream>

WorkStealingScheduler::WorkStealingScheduler() 
    : totalWork(0), shutdownRequested(false), monitorActive(false),
      cpuThreadExited(false), gpuThreadExited(false), aneThreadExited(false),
      lastCpuWorkTime(0), lastGpuWorkTime(0), lastAneWorkTime(0) {
    for (int i = 0; i < 3; i++) {
        queues[i].activeWorkers = 0;
        queues[i].avgProcessingTime = 0.0;
        queues[i].chunksProcessed = 0;
        queues[i].allocatedChunks = 0;
    }
    std::cout << "DEBUG: WorkStealingScheduler initialized" << std::endl;
}

WorkStealingScheduler::~WorkStealingScheduler() {
    std::cout << "DEBUG: WorkStealingScheduler shutting down" << std::endl;
    shutdownRequested = true;
    int timeout = 0;
    while (monitorActive && timeout < 100) {
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
        timeout++;
    }
    if (monitorActive) {
        std::cout << "WARNING: Monitor thread didn't exit cleanly" << std::endl;
    }
}

void WorkStealingScheduler::initialize() {
    std::cout << "DEBUG: Starting work stealing monitor thread" << std::endl;
    monitorActive = true;
    std::thread monitorThread(&WorkStealingScheduler::monitor, this);
    monitorThread.detach();
}

void WorkStealingScheduler::addWork(const std::vector<WorkChunk>& chunks, DeviceType device) {
    auto& queue = getQueue(device);
    std::lock_guard lock(queue.mutex);
    for (const auto& chunk : chunks) {
        queue.queue.push(chunk);
        totalWork++;
    }
    queue.cv.notify_all();
}

WorkChunk* WorkStealingScheduler::getWork(DeviceType device) {
    std::string deviceName = getDeviceName(device);
    if (device == DeviceType::ANE) {
        std::cout << "DEBUG: ANE is disabled, skipping getWork for ANE" << std::endl;
        return nullptr;
    }
    const char* gpuOnlyEnv = std::getenv("GPU_ONLY");
    bool gpuOnly = (gpuOnlyEnv != nullptr);
    auto& queue = getQueue(device);
    std::unique_lock lock(queue.mutex);
    int queueSize = queue.queue.size();
    if (queueSize == 0 && totalWork == 0) {
        std::cout << "DEBUG: " << deviceName << " has no work and no work remains in system, not incrementing worker count" << std::endl;
        return nullptr;
    }
    std::cout << "DEBUG: " << deviceName << " getting work, active workers before: " << queue.activeWorkers << std::endl;
    queue.activeWorkers++;
    const int MAX_WAIT_MS = 10000;  
    int totalWaitTime = 0;
    int waitIterations = 0;
    while (queue.queue.empty() && totalWork > 0 && totalWaitTime < MAX_WAIT_MS) {
        std::cout << "DEBUG: " << deviceName << " waiting for work, total remaining: " << totalWork << std::endl;
        if (++waitIterations > 10) {
            std::cout << "DEBUG: " << deviceName << " still waiting after " << waitIterations << " attempts" << std::endl;
        }
        auto waitStatus = queue.cv.wait_for(lock, std::chrono::milliseconds(100));
        totalWaitTime += 100;
        if (!gpuOnly && totalWaitTime % 1000 == 0) {  
            lock.unlock();  
            DeviceType busyDevice = selectDeviceToStealFrom(device);
            if (busyDevice != device) {
                std::string fromDevice = getDeviceName(busyDevice);
                std::cout << "DEBUG: " << deviceName << " attempting to directly steal work from " << fromDevice << std::endl;
                WorkChunk* stolen = steal(busyDevice, device);
                if (stolen) {
                    if (profiler) {
                        profiler->recordStealEvent(fromDevice, deviceName);
                    }
                    std::vector<WorkChunk> stolenWork;
                    stolenWork.push_back(*stolen);
                    addWork(stolenWork, device);
                    delete stolen;
                }
            }
            lock.lock();  
        }
    }
    if (queue.queue.empty()) {
        if (queue.activeWorkers > 0) {
            std::cout << "DEBUG: " << deviceName << " found no work, decrementing active workers: " << queue.activeWorkers << " -> " << (queue.activeWorkers-1) << std::endl;
            queue.activeWorkers--;
        } else {
            std::cout << "WARNING: " << deviceName << " worker count already at 0!" << std::endl;
        }
        return nullptr;
    }

    WorkChunk chunk = queue.queue.front();
    queue.queue.pop();
    totalWork--;
    std::cout << "DEBUG: " << deviceName << " got work chunk [" << chunk.startRow << ":" << chunk.endRow 
              << ", " << chunk.startCol << ":" << chunk.endCol << "], remaining: " << totalWork << std::endl;
    int64_t currentTime = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::system_clock::now().time_since_epoch()
    ).count();

    if (device == DeviceType::CPU) {
        lastCpuWorkTime = currentTime;
    } else if (device == DeviceType::GPU) {
        lastGpuWorkTime = currentTime;
    } else if (device == DeviceType::ANE) {
        lastAneWorkTime = currentTime;
    }
    queue.lastWorkTime = std::chrono::steady_clock::now();
    WorkChunk* resultChunk = new WorkChunk(chunk);
    return resultChunk;
}

WorkChunk* WorkStealingScheduler::steal(DeviceType fromDevice, DeviceType toDevice) {
    const char* gpuOnlyEnv = std::getenv("GPU_ONLY");
    bool gpuOnly = (gpuOnlyEnv != nullptr);
    if (gpuOnly) {
        std::cout << "DEBUG: Stealing disabled in GPU_ONLY mode" << std::endl;
        return nullptr;
    }
    if (fromDevice == DeviceType::ANE || toDevice == DeviceType::ANE) {
        std::cout << "DEBUG: ANE is disabled, skipping work stealing involving ANE" << std::endl;
        return nullptr;
    }
    auto& fromQueue = getQueue(fromDevice);
    auto& toQueue = getQueue(toDevice);
    std::string fromDeviceName = getDeviceName(fromDevice);
    std::string toDeviceName = getDeviceName(toDevice);
    std::unique_lock<std::mutex> fromLock(fromQueue.mutex, std::try_to_lock);
    if (!fromLock.owns_lock()) {
        std::cout << "DEBUG: Cannot steal from " << fromDeviceName << " - mutex is locked" << std::endl;
        return nullptr;
    }
    if (fromQueue.queue.size() <= 1) {
        std::cout << "DEBUG: Cannot steal from " << fromDeviceName << " - only " << fromQueue.queue.size() << " chunks (need > 1)" << std::endl;
        return nullptr;
    }
    std::vector<WorkChunk> chunks;
    while (!fromQueue.queue.empty()) {
        chunks.push_back(fromQueue.queue.front());
        fromQueue.queue.pop();
    }
    std::sort(chunks.begin(), chunks.end(), 
              [](const WorkChunk& a, const WorkChunk& b) {
                  int aSize = (a.endRow - a.startRow) * (a.endCol - a.startCol);
                  int bSize = (b.endRow - b.startRow) * (b.endCol - b.startCol);
                  return aSize > bSize;  
              });
    WorkChunk chunk = chunks.front();
    chunks.erase(chunks.begin());
    for (const auto& c : chunks) {
        fromQueue.queue.push(c);
    }
    std::cout << "DEBUG: Stealing chunk of size " 
              << ((chunk.endRow - chunk.startRow) * (chunk.endCol - chunk.startCol))
              << " cells from " << fromDeviceName << " to " << toDeviceName << std::endl;
    std::cout << "DEBUG: Stolen chunk [" << chunk.startRow << ":" << chunk.endRow 
              << ", " << chunk.startCol << ":" << chunk.endCol 
              << "] from " << fromDeviceName << " to " << toDeviceName << std::endl;
    fromQueue.allocatedChunks--;   
    toQueue.allocatedChunks++;     
    int rows = chunk.endRow - chunk.startRow;
    int cols = chunk.endCol - chunk.startCol;
    if (rows > 4 || cols > 4) {
        int midRow = chunk.startRow + rows / 2;
        int midCol = chunk.startCol + cols / 2;
        if (rows >= 32 || cols >= 32) {
            if (rows > cols) {
                WorkChunk bottom(midRow, chunk.endRow, chunk.startCol, chunk.endCol);
                fromQueue.queue.push(WorkChunk(chunk.startRow, midRow, chunk.startCol, chunk.endCol));
                std::cout << "DEBUG: Split and stole bottom half of large chunk" << std::endl;
                return new WorkChunk(bottom);
            } else {
                WorkChunk right(chunk.startRow, chunk.endRow, midCol, chunk.endCol);
                fromQueue.queue.push(WorkChunk(chunk.startRow, chunk.endRow, chunk.startCol, midCol));
                std::cout << "DEBUG: Split and stole right half of large chunk" << std::endl;
                return new WorkChunk(right);
            }
        } else {
            WorkChunk q1(chunk.startRow, midRow, chunk.startCol, midCol);        
            WorkChunk q2(chunk.startRow, midRow, midCol, chunk.endCol);          
            WorkChunk q3(midRow, chunk.endRow, chunk.startCol, midCol);          
            WorkChunk q4(midRow, chunk.endRow, midCol, chunk.endCol);            
            fromQueue.queue.push(q2);
            fromQueue.queue.push(q3);
            fromQueue.queue.push(q4);
            std::cout << "DEBUG: Split chunk into quadrants and stole top-left" << std::endl;
            return new WorkChunk(q1);
        }
    }
    std::cout << "DEBUG: Stole chunk without splitting (too small to split)" << std::endl;
    return new WorkChunk(chunk);
}

bool WorkStealingScheduler::hasWork(DeviceType device) {
    auto& queue = getQueue(device);
    std::lock_guard lock(queue.mutex);
    return !queue.queue.empty();
}

void WorkStealingScheduler::waitForCompletion() {
    std::cout << "DEBUG: Waiting for work completion, total work remaining: " << totalWork << std::endl;
    int checkCounter = 0;  
    while (totalWork > 0) {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        bool workRebalanced = false;
        for (int fromDevice = 0; fromDevice < 3; fromDevice++) {
            if (queues[fromDevice].activeWorkers == 0 && !queues[fromDevice].queue.empty()) {
                for (int toDevice = 0; toDevice < 3; toDevice++) {
                    if (toDevice != fromDevice && queues[toDevice].activeWorkers > 0) {
                        std::string fromName = getDeviceName(static_cast<DeviceType>(fromDevice));
                        std::string toName = getDeviceName(static_cast<DeviceType>(toDevice));
                        std::unique_lock fromLock(queues[fromDevice].mutex);
                        if (!queues[fromDevice].queue.empty()) {
                            int moveCount = queues[fromDevice].queue.size();
                            std::cout << "DEBUG: " << fromName << " executor finished but left " << moveCount
                                    << " chunks. Moving to " << toName << " queue." << std::endl;
                            std::vector<WorkChunk> remainingWork;
                            while (!queues[fromDevice].queue.empty()) {
                                remainingWork.push_back(queues[fromDevice].queue.front());
                                queues[fromDevice].queue.pop();
                            }
                            fromLock.unlock();
                            std::unique_lock toLock(queues[toDevice].mutex);
                            for (auto& chunk : remainingWork) {
                                queues[toDevice].queue.push(chunk);
                            }
                            queues[toDevice].cv.notify_all();
                            toLock.unlock();
                            workRebalanced = true;
                            break;  
                        }
                    }
                }
                if (!workRebalanced) {
                    std::cout << "DEBUG: All executors inactive but work remains. Resetting work counter." << std::endl;
                    totalWork = 0;
                    return;  
                }
            }
        }
        if (workRebalanced) {
            continue;  
        }
        std::cout << "DEBUG: Still waiting for work, remaining: " << totalWork 
                  << " | Workers - CPU: " << queues[0].activeWorkers 
                  << ", GPU: " << queues[1].activeWorkers 
                  << ", ANE: " << queues[2].activeWorkers 
                  << " | Queue sizes - CPU: " << queues[0].queue.size()
                  << ", GPU: " << queues[1].queue.size()
                  << ", ANE: " << queues[2].queue.size() << std::endl;
        checkCounter++;
        if (checkCounter >= 10) {
            checkCounter = 0;  
            if (queues[0].activeWorkers == 0 && queues[1].activeWorkers == 0 && queues[2].activeWorkers == 0) {
                int remainingWorkInQueues = 0;
                for (int i = 0; i < 3; i++) {
                    auto& q = queues[i];
                    std::lock_guard<std::mutex> lock(q.mutex);
                    remainingWorkInQueues += q.queue.size();
                }
                if (remainingWorkInQueues != totalWork) {
                    std::cout << "DEBUG: Work count mismatch. Counter says " << totalWork 
                              << " but queues contain " << remainingWorkInQueues << " chunks. Reconciling." << std::endl;
                    if (remainingWorkInQueues == 0) {
                        totalWork = 0;
                    } 
                    else if (remainingWorkInQueues > 0) {
                        totalWork = remainingWorkInQueues;
                        std::cout << "DEBUG: Activating emergency CPU worker to handle orphaned work" << std::endl;
                        std::thread emergencyWorker([this]() {
                            std::this_thread::sleep_for(std::chrono::milliseconds(200));
                            std::vector<WorkChunk> allWork;
                            for (int i = 0; i < 3; i++) {
                                auto& queue = queues[i];
                                std::unique_lock<std::mutex> lock(queue.mutex);
                                while (!queue.queue.empty()) {
                                    allWork.push_back(queue.queue.front());
                                    queue.queue.pop();
                                }
                            }
                            if (!allWork.empty()) {
                                auto& cpuQueue = queues[0];
                                std::unique_lock<std::mutex> lock(cpuQueue.mutex);
                                for (auto& chunk : allWork) {
                                    cpuQueue.queue.push(chunk);
                                }
                                cpuQueue.activeWorkers = 1;  
                                cpuQueue.cv.notify_all();
                            }
                        });
                        emergencyWorker.detach();
                    }
                } else {
                    std::cout << "DEBUG: Timeout waiting for work completion. Force resetting work counter from " 
                              << totalWork << " to 0." << std::endl;
                    totalWork = 0;
                }
                break;
            }
            int totalWorkInQueues = 0;
            for (int i = 0; i < 3; i++) {
                auto& q = queues[i];
                std::lock_guard<std::mutex> lock(q.mutex);
                totalWorkInQueues += q.queue.size();
            }
            if (totalWorkInQueues != totalWork) {
                std::cout << "DEBUG: Work counter mismatch detected during active execution. " 
                          << "Counter says " << totalWork << " but queues contain " << totalWorkInQueues 
                          << ". Correcting." << std::endl;
                totalWork = totalWorkInQueues;
            }
        }
    }

    std::cout << "DEBUG: All work processed, waiting for active workers to finish" << std::endl;
    for (int i = 0; i < 3; i++) {
        std::string deviceName = getDeviceName(static_cast<DeviceType>(i));
        auto& queue = queues[i];
        std::unique_lock<std::mutex> lock(queue.mutex);
        if (queue.activeWorkers > 0) {
            std::cout << "DEBUG: Waiting for " << queue.activeWorkers << " active " << deviceName << " workers" << std::endl;
        }
        int workerCheckCount = 0;
        while (queue.activeWorkers > 0) {
            lock.unlock();   
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
            lock.lock();     
            std::cout << "DEBUG: Still waiting for " << queue.activeWorkers << " active " << deviceName << " workers" << std::endl;
            workerCheckCount++;
            if (workerCheckCount >= 30) {  
                bool forceReset = false;
                int64_t currentTime = std::chrono::duration_cast<std::chrono::milliseconds>(
                    std::chrono::system_clock::now().time_since_epoch()
                ).count();
                int64_t lastWorkTime = 0;
                if (deviceName == "CPU") {
                    lastWorkTime = lastCpuWorkTime;
                    if (cpuThreadExited) forceReset = true;
                } else if (deviceName == "GPU") {
                    lastWorkTime = lastGpuWorkTime;
                    if (gpuThreadExited) forceReset = true;
                } else if (deviceName == "ANE") {
                    lastWorkTime = lastAneWorkTime;
                    if (aneThreadExited) forceReset = true;
                }
                const int64_t stallThreshold = 5000;  
                if (lastWorkTime > 0 && (currentTime - lastWorkTime) > stallThreshold) {
                    std::cout << "DEBUG: " << deviceName << " worker appears stalled. Last work was " 
                              << (currentTime - lastWorkTime) << "ms ago. Resetting worker count." << std::endl;
                    forceReset = true;
                }
                if (forceReset) {
                    std::cout << "DEBUG: " << deviceName << " thread has exited or stalled but worker count is " 
                              << queue.activeWorkers << ". Resetting to 0." << std::endl;
                    queue.activeWorkers = 0;
                    break;
                } else {
                    workerCheckCount = 0;
                }
            }
        }
    }
    std::cout << "DEBUG: All workers finished, completion successful" << std::endl;
}

WorkStealingScheduler::DeviceQueue& WorkStealingScheduler::getQueue(DeviceType device) {
    return queues[static_cast<int>(device)];
}

std::string WorkStealingScheduler::getDeviceName(DeviceType device) {
    switch (static_cast<int>(device)) {
        case 0: return "CPU";
        case 1: return "GPU";
        case 2: return "ANE";
        default: return "Unknown";
    }
}

void WorkStealingScheduler::recordChunkProcessingTime(DeviceType device, double seconds) {
    auto& queue = getQueue(device);
    std::lock_guard lock(queue.mutex);
    if (queue.chunksProcessed == 0) {
        queue.avgProcessingTime = seconds;
    } else {
        const double weight = 0.7;  
        queue.avgProcessingTime = (queue.avgProcessingTime * (1 - weight)) + (seconds * weight);
    }
    queue.chunksProcessed++;
    queue.lastWorkTime = std::chrono::steady_clock::now();
    std::string name = getDeviceName(device);
    std::cout << "DEBUG: " << name << " processed chunk in " << (seconds * 1000) 
              << "ms (avg: " << (queue.avgProcessingTime * 1000) << "ms)" << std::endl;
}

DeviceType WorkStealingScheduler::selectDeviceToStealFrom(DeviceType idleDevice) {
    const char* gpuOnlyEnv = std::getenv("GPU_ONLY");
    bool gpuOnly = (gpuOnlyEnv != nullptr);
    if (gpuOnly) {
        return idleDevice;  
    }
    if (idleDevice == DeviceType::ANE) {
        std::cout << "DEBUG: ANE is disabled, preventing it from selecting steal targets" << std::endl;
        return idleDevice;
    }
    DeviceType bestDevice = idleDevice;   
    double maxScore = 0.0;
    std::string idleName = getDeviceName(idleDevice);
    std::cout << "DEBUG: " << idleName << " is looking for a device to steal from" << std::endl;
    for (int j = 0; j < 3; j++) {
        DeviceType otherDevice = static_cast<DeviceType>(j);
        if (otherDevice != idleDevice) {
            std::string otherName = getDeviceName(otherDevice);
            auto& queue = getQueue(otherDevice);
            std::lock_guard lock(queue.mutex);
            int queueSize = queue.queue.size();
            if (queueSize <= 1) {
                std::cout << "DEBUG: " << otherName << " has only " << queueSize << " chunks, not enough to steal from" << std::endl;
                continue;  
            }
            double avgTimePerChunk = queue.avgProcessingTime;
            if (avgTimePerChunk <= 0.0001) {
                std::cout << "DEBUG: " << otherName << " avg time is too low, using default 10ms" << std::endl;
                avgTimePerChunk = 0.01;  
            }
            int activeWorkers = queue.activeWorkers.load() > 0 ? queue.activeWorkers.load() : 1;
            double score = (queueSize * avgTimePerChunk) / activeWorkers;
            std::cout << "DEBUG: " << otherName << " steal score: " << score 
                      << " (queue size: " << queueSize 
                      << ", avg time: " << (avgTimePerChunk * 1000) << "ms"
                      << ", active workers: " << activeWorkers << ")" << std::endl;
            if (score > maxScore) {
                maxScore = score;
                bestDevice = otherDevice;
                std::cout << "DEBUG: New best device to steal from: " << otherName << " with score " << score << std::endl;
            }
        }
    }
    std::string bestName = getDeviceName(bestDevice);
    if (bestDevice == idleDevice) {
        std::cout << "DEBUG: No suitable device found to steal from" << std::endl;
    } else {
        std::cout << "DEBUG: Selected " << bestName << " as best device to steal from with score " << maxScore << std::endl;
    }
    return bestDevice;
}

void WorkStealingScheduler::monitor() {
    std::cout << "DEBUG: Monitor thread started" << std::endl;
    int statusCycles = 0;
    const char* gpuOnlyEnv = std::getenv("GPU_ONLY");
    bool gpuOnly = (gpuOnlyEnv != nullptr);
    if (gpuOnly) {
        std::cout << "DEBUG: Monitor thread disabled work stealing (GPU_ONLY mode enabled)" << std::endl;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(200));
    std::cout << "DEBUG: Monitor thread waiting 200ms for initialization" << std::endl;
    while (!shutdownRequested && (
           totalWork > 0 ||
           queues[0].activeWorkers > 0 ||
           queues[1].activeWorkers > 0 ||
           queues[2].activeWorkers > 0)) {
        if (gpuOnly) {
            std::this_thread::sleep_for(std::chrono::milliseconds(500));
            continue;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
        if (++statusCycles % 25 == 0) {
            std::cout << "DEBUG: Monitor status - Work: " << totalWork 
                    << ", Workers (CPU/GPU/ANE): " 
                    << queues[0].activeWorkers << "/" 
                    << queues[1].activeWorkers << "/" 
                    << queues[2].activeWorkers 
                    << " | Queue sizes: "
                    << queues[0].queue.size() << "/"
                    << queues[1].queue.size() << "/"
                    << queues[2].queue.size() << std::endl;
        }
        static int stealingCooldown = 0;
        if (stealingCooldown > 0) {
            stealingCooldown--;
        }
        for (int i = 0; i < 3; i++) {
            if (shutdownRequested) break;
            DeviceType idleDevice = static_cast<DeviceType>(i);
            if (idleDevice == DeviceType::ANE) {
                continue;  
            }
            if (!hasWork(idleDevice) && totalWork > 0 && queues[i].activeWorkers > 0) {
                std::string deviceName = getDeviceName(idleDevice);
                std::cout << "DEBUG: " << deviceName << " is idle but has active workers, attempting to steal work" << std::endl;
                DeviceType busyDevice = selectDeviceToStealFrom(idleDevice);
                if (busyDevice != idleDevice) {  
                    std::string fromDevice = getDeviceName(busyDevice);
                    std::cout << "DEBUG: Attempting to steal work from " << fromDevice << " to " << deviceName << std::endl;
                    WorkChunk* stolen = steal(busyDevice, idleDevice);
                    if (stolen) {
                        if (profiler) {
                            profiler->recordStealEvent(fromDevice, deviceName);
                        }
                        std::vector<WorkChunk> stolenWork;
                        stolenWork.push_back(*stolen);
                        addWork(stolenWork, idleDevice);
                        delete stolen;
                        stealingCooldown = 5;
                    }
                }
            }
        }
        if (stealingCooldown > 0) {
            continue;
        }
        for (int i = 0; i < 3; i++) {
            if (shutdownRequested) break;
            DeviceType device = static_cast<DeviceType>(i);
            if (device == DeviceType::ANE) {
                continue;  
            }
            auto& queue = getQueue(device);
            std::string deviceName = getDeviceName(device);
            std::unique_lock lock(queue.mutex);
            int queueSize = queue.queue.size();
            double avgProcessingTime = queue.avgProcessingTime;
            int activeWorkers = queue.activeWorkers;
            lock.unlock();
            std::cout << "DEBUG: Evaluating " << deviceName << " for proactive stealing:" << std::endl
                      << "  - Queue size: " << queueSize << std::endl
                      << "  - Active workers: " << activeWorkers << std::endl
                      << "  - Total work remaining: " << totalWork << std::endl
                      << "  - Avg processing time: " << (avgProcessingTime * 1000) << "ms" << std::endl
                      << "  - Threshold check: " << (queueSize < activeWorkers * 2 ? "PASSED" : "FAILED") 
                      << " (need < " << (activeWorkers * 2) << " chunks)" << std::endl
                      << "  - Total work check: " << (totalWork > queueSize + 1 ? "PASSED" : "FAILED")
                      << " (need > " << (queueSize + 1) << " total work)" << std::endl;
            if (activeWorkers > 0 && (totalWork > queueSize * 1.2)) {
                std::cout << "DEBUG: Proactive stealing for " << deviceName 
                          << " with queue size " << queueSize 
                          << " and " << activeWorkers << " workers" << std::endl;
                DeviceType targetDevice = selectDeviceToStealFrom(device);
                if (targetDevice != device) {
                    std::string targetName = getDeviceName(targetDevice);
                    std::cout << "DEBUG: Attempting proactive steal from " << targetName << " to " << deviceName << std::endl;
                    WorkChunk* stolen = steal(targetDevice, device);
                    if (stolen) {
                        std::cout << "DEBUG: Proactively stole work from " << targetName 
                                  << " to " << deviceName << std::endl;
                        if (profiler) {
                            profiler->recordStealEvent(targetName, deviceName);
                        }
                        std::vector<WorkChunk> stolenWork;
                        stolenWork.push_back(*stolen);
                        addWork(stolenWork, device);
                        delete stolen;
                        stealingCooldown = 5;
                    } else {
                        std::cout << "DEBUG: Failed to steal work from " << targetName << " to " << deviceName << std::endl;
                    }
                } else {
                    std::cout << "DEBUG: No suitable device found to steal from for " << deviceName << std::endl;
                }
            } else {
                std::cout << "DEBUG: " << deviceName << " doesn't need to steal work proactively" << std::endl;
            }
        }
    }
    std::cout << "DEBUG: Monitor thread exiting" << std::endl;
    monitorActive = false;
}