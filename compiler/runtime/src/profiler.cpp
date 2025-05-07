#include "profiler.h"
#include <iostream>
#include <iomanip>
#include <sstream>

Profiler::Profiler() = default;

void Profiler::startTimer(const std::string& name) {
    timers[name].start = std::chrono::steady_clock::now();
}

void Profiler::stopTimer(const std::string& name) {
    auto end = std::chrono::steady_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(
        end - timers[name].start).count() / 1000000.0;
    timers[name].totalTime += duration;
    timers[name].count++;
}

void Profiler::recordZeroTime(const std::string& name) {
    if (timers.find(name) == timers.end()) {
        timers[name].totalTime = 0.0;
        timers[name].count = 1;   
    }
}

void Profiler::recordChunkExecution(const std::string& device, int chunkSize) {
    deviceStats[device].chunksProcessed++;
    deviceStats[device].totalElements += chunkSize;
}

void Profiler::recordInitialAllocation(const std::string& device, int chunkCount, int totalChunks) {
    deviceStats[device].chunksProcessed = 0;
    deviceStats[device].totalElements = 0;
    deviceStats[device].allocatedChunks = chunkCount;
    deviceStats[device].percentUtilization = (100.0 * chunkCount) / totalChunks;
}

void Profiler::recordStealEvent(const std::string& fromDevice, const std::string& toDevice) {
    if (workStealingDisabled) {
        return;
    }
    std::string key = fromDevice + "->" + toDevice;
    stealStats[key].count++;
}

void Profiler::disableWorkStealing() {
    workStealingDisabled = true;
    stealStats.clear();
}

void Profiler::printReport() {
    std::cout << "\n=== HETEROGENEOUS EXECUTION PERFORMANCE SUMMARY ===" << std::endl;
    std::cout << std::fixed << std::setprecision(6);
    int totalProcessed = 0;
    for (const auto& stat : deviceStats) {
        totalProcessed += stat.second.chunksProcessed;
    }
    std::cout << "\n📊 CHUNK ALLOCATION & EXECUTION:" << std::endl;
    std::cout << "-----------------------------" << std::endl;
    bool anyCpuChunks = deviceStats.find("CPU") != deviceStats.end() && deviceStats["CPU"].chunksProcessed > 0;
    bool anyGpuChunks = deviceStats.find("GPU") != deviceStats.end() && deviceStats["GPU"].chunksProcessed > 0;
    bool anyAneChunks = deviceStats.find("ANE") != deviceStats.end() && deviceStats["ANE"].chunksProcessed > 0;
    if (anyCpuChunks || anyGpuChunks || anyAneChunks) {
        std::cout << "   INITIAL ALLOCATION:" << std::endl;
        std::cout << "   ------------------" << std::endl;
        int totalAllocated = 0;
        int cpuAllocated = 0;
        int gpuAllocated = 0; 
        int aneAllocated = 0;
        if (deviceStats.find("CPU") != deviceStats.end()) {
            cpuAllocated = deviceStats["CPU"].allocatedChunks;
            totalAllocated += cpuAllocated;
        }
        if (deviceStats.find("GPU") != deviceStats.end()) {
            gpuAllocated = deviceStats["GPU"].allocatedChunks;
            totalAllocated += gpuAllocated;
        }
        if (deviceStats.find("ANE") != deviceStats.end()) {
            aneAllocated = deviceStats["ANE"].allocatedChunks;
            totalAllocated += aneAllocated;
        }
        if (cpuAllocated > 0) {
            std::cout << "   • CPU: " << cpuAllocated << " chunks (" 
                      << std::fixed << std::setprecision(1) 
                      << (100.0 * cpuAllocated / totalAllocated) << "%)" << std::endl;
        }
        if (gpuAllocated > 0) {
            std::cout << "   • GPU: " << gpuAllocated << " chunks (" 
                      << std::fixed << std::setprecision(1) 
                      << (100.0 * gpuAllocated / totalAllocated) << "%)" << std::endl;
        }
        if (aneAllocated > 0) {
            std::cout << "   • ANE: " << aneAllocated << " chunks (" 
                      << std::fixed << std::setprecision(1) 
                      << (100.0 * aneAllocated / totalAllocated) << "%)" << std::endl;
        }
        std::cout << std::endl << "   ACTUAL EXECUTION:" << std::endl;
        std::cout << "   -----------------" << std::endl;
        std::cout << "   [";
        int barWidth = 50;  
        int cpuChunks = anyCpuChunks ? deviceStats["CPU"].chunksProcessed : 0;
        int gpuChunks = anyGpuChunks ? deviceStats["GPU"].chunksProcessed : 0;
        int aneChunks = anyAneChunks ? deviceStats["ANE"].chunksProcessed : 0;
        int cpuWidth = static_cast<int>((double)cpuChunks / totalProcessed * barWidth);
        int gpuWidth = static_cast<int>((double)gpuChunks / totalProcessed * barWidth);
        int aneWidth = static_cast<int>((double)aneChunks / totalProcessed * barWidth);
        if (cpuChunks > 0 && cpuWidth == 0) cpuWidth = 1;
        if (gpuChunks > 0 && gpuWidth == 0) gpuWidth = 1;
        if (aneChunks > 0 && aneWidth == 0) aneWidth = 1;
        int totalWidth = cpuWidth + gpuWidth + aneWidth;
        if (totalWidth < barWidth) {
            if (cpuChunks >= gpuChunks && cpuChunks >= aneChunks) {
                cpuWidth += (barWidth - totalWidth);
            } else if (gpuChunks >= cpuChunks && gpuChunks >= aneChunks) {
                gpuWidth += (barWidth - totalWidth);
            } else {
                aneWidth += (barWidth - totalWidth);
            }
        } else if (totalWidth > barWidth) {
            if (cpuChunks >= gpuChunks && cpuChunks >= aneChunks) {
                cpuWidth -= (totalWidth - barWidth);
            } else if (gpuChunks >= cpuChunks && gpuChunks >= aneChunks) {
                gpuWidth -= (totalWidth - barWidth);
            } else {
                aneWidth -= (totalWidth - barWidth);
            }
        }
        for (int i = 0; i < cpuWidth; i++) std::cout << "C";
        for (int i = 0; i < gpuWidth; i++) std::cout << "G";
        for (int i = 0; i < aneWidth; i++) std::cout << "A";
        std::cout << "]" << std::endl;
        if (cpuChunks > 0) {
            std::cout << "    C = CPU: " << std::fixed << std::setprecision(1) 
                      << (100.0 * cpuChunks / totalProcessed) << "% (" 
                      << cpuChunks << " chunks)" << std::endl;
        }
        if (gpuChunks > 0) {
            std::cout << "    G = GPU: " << std::fixed << std::setprecision(1) 
                      << (100.0 * gpuChunks / totalProcessed) << "% (" 
                      << gpuChunks << " chunks)" << std::endl;
        }
        if (aneChunks > 0) {
            std::cout << "    A = ANE: " << std::fixed << std::setprecision(1) 
                      << (100.0 * aneChunks / totalProcessed) << "% (" 
                      << aneChunks << " chunks)" << std::endl;
        }
        int cpuDelta = cpuChunks - cpuAllocated;
        int gpuDelta = gpuChunks - gpuAllocated;
        int aneDelta = aneChunks - aneAllocated;
        if (cpuDelta != 0 || gpuDelta != 0 || aneDelta != 0) {
            std::cout << std::endl << "   WORK STEALING EFFECTS:" << std::endl;
            std::cout << "   ---------------------" << std::endl;
            if (cpuDelta > 0) {
                std::cout << "   • CPU stole " << cpuDelta << " additional chunks" << std::endl;
            } else if (cpuDelta < 0) {
                std::cout << "   • CPU gave up " << -cpuDelta << " chunks to other devices" << std::endl;
            } else {
                std::cout << "   • CPU processed exactly its allocated chunks" << std::endl;
            }
            if (gpuDelta > 0) {
                std::cout << "   • GPU stole " << gpuDelta << " additional chunks" << std::endl;
            } else if (gpuDelta < 0) {
                std::cout << "   • GPU gave up " << -gpuDelta << " chunks to other devices" << std::endl;
            } else {
                std::cout << "   • GPU processed exactly its allocated chunks" << std::endl;
            }
            if (aneDelta > 0) {
                std::cout << "   • ANE stole " << aneDelta << " additional chunks" << std::endl;
            } else if (aneDelta < 0) {
                std::cout << "   • ANE gave up " << -aneDelta << " chunks to other devices" << std::endl;
            } else if (aneAllocated > 0) {
                std::cout << "   • ANE processed exactly its allocated chunks" << std::endl;
            }
        }
        std::cout << "\n   ALLOCATION DEBUG:" << std::endl;
        std::cout << "   -----------------" << std::endl;
        if (cpuAllocated > 0) {
            std::cout << "   • CPU: Initial=" << cpuAllocated << ", Processed=" << cpuChunks << ", Delta=" << cpuDelta << std::endl;
        }
        if (gpuAllocated > 0) {
            std::cout << "   • GPU: Initial=" << gpuAllocated << ", Processed=" << gpuChunks << ", Delta=" << gpuDelta << std::endl;
        }
        if (aneAllocated > 0) {
            std::cout << "   • ANE: Initial=" << aneAllocated << ", Processed=" << aneChunks << ", Delta=" << aneDelta << std::endl;
        }
    } else {
        std::cout << "   No chunks were processed." << std::endl;
    }
    std::cout << "\n WORK DISTRIBUTION:" << std::endl;
    std::cout << "-------------------" << std::endl;
    std::cout << "   Initial distribution: 80/20 between GPU and CPU" << std::endl;
    std::cout << "   ANE executor disabled (stub implementation)" << std::endl;
    if (!stealStats.empty()) {
        std::cout << "\n🔀 WORK STEALING EVENTS:" << std::endl;
        std::cout << "---------------------" << std::endl;
        int totalSteals = 0;
        for (const auto& stat : stealStats) {
            totalSteals += stat.second.count;
            std::cout << "   • " << stat.first << ": " << stat.second.count << " chunks" << std::endl;
        }
        std::cout << "   Total: " << totalSteals << " chunks stolen" << std::endl;
    }
    std::cout << "\n️ EXECUTION TIMES:" << std::endl;
    std::cout << "-----------------" << std::endl;
    double cpuTime = 0, gpuTime = 0, aneTime = 0, totalTime = 0;
    for (const auto& timer : timers) {
        if (timer.first == "cpu_execution") {
            cpuTime = timer.second.totalTime;
        } else if (timer.first == "gpu_execution") {
            gpuTime = timer.second.totalTime;
        } else if (timer.first == "ane_execution") {
            aneTime = timer.second.totalTime;
        } else if (timer.first == "total_execution" || timer.first == "matrix_multiplication") {
            totalTime = timer.second.totalTime;
        }
    }
    if (totalTime > 0) {
        std::cout << "   Total execution time: " << formatTime(totalTime) << std::endl;
    }
    bool anyDeviceWorked = false;
    if (anyCpuChunks || anyGpuChunks || anyAneChunks || 
        cpuTime > 0 || gpuTime > 0 || aneTime > 0) {
        std::cout << "   Device thread times:" << std::endl;
        if (anyCpuChunks || cpuTime > 0) {
            std::cout << "   • CPU thread: " << formatTime(cpuTime) << std::endl;
            anyDeviceWorked = true;
        }
        if (anyGpuChunks || (gpuTime > 0 && gpuTime > 0.000001)) {
            std::cout << "   • GPU thread: " << formatTime(gpuTime) << std::endl;
            anyDeviceWorked = true;
        }
        if (anyAneChunks || (aneTime > 0 && aneTime > 0.000001)) {
            std::cout << "   • ANE thread: " << formatTime(aneTime) << std::endl;
            anyDeviceWorked = true;
        }
        if (!anyDeviceWorked) {
            std::cout << "   • No device thread timing data available" << std::endl;
        }
    } else {
        int totalChunks = 0;
        for (const auto& stat : deviceStats) {
            totalChunks += stat.second.chunksProcessed;
        }
        if (totalChunks > 0) {
            std::cout << "   Device times not recorded, but " << totalChunks 
                      << " chunks were processed." << std::endl;
        } else {
            std::cout << "   No devices had any chunks to process." << std::endl;
        }
    }
    std::cout << "\n--- DETAILED STATISTICS ---" << std::endl;
    std::cout << "\nDevice Statistics:" << std::endl;
    std::cout << "-----------------" << std::endl;
    int cpuChunks = deviceStats["CPU"].chunksProcessed;
    int gpuChunks = deviceStats["GPU"].chunksProcessed;
    int aneChunks = deviceStats.find("ANE") != deviceStats.end() ? deviceStats["ANE"].chunksProcessed : 0;
    int cpuAllocated = deviceStats["CPU"].allocatedChunks;
    int gpuAllocated = deviceStats["GPU"].allocatedChunks;
    int aneAllocated = deviceStats.find("ANE") != deviceStats.end() ? deviceStats["ANE"].allocatedChunks : 0;
    int cpuDelta = cpuChunks - cpuAllocated;
    if (cpuDelta > 0) {
        gpuChunks = gpuAllocated - cpuDelta;
    }
    std::cout << std::setw(10) << std::left << "GPU" << ": "
              << "Initial allocation: " << gpuAllocated << " chunks ("
              << std::fixed << std::setprecision(1) << (100.0 * gpuAllocated / (gpuAllocated + cpuAllocated + aneAllocated)) << "%), "
              << "Processed: " << gpuChunks << " chunks" << std::endl;
    std::cout << std::setw(10) << std::left << "ANE" << ": "
              << "Initial allocation: " << aneAllocated << " chunks ("
              << std::fixed << std::setprecision(1) << (100.0 * aneAllocated / (gpuAllocated + cpuAllocated + aneAllocated)) << "%), "
              << "Processed: " << aneChunks << " chunks" << std::endl;
    std::cout << std::setw(10) << std::left << "CPU" << ": "
              << "Initial allocation: " << cpuAllocated << " chunks ("
              << std::fixed << std::setprecision(1) << (100.0 * cpuAllocated / (gpuAllocated + cpuAllocated + aneAllocated)) << "%), "
              << "Processed: " << cpuChunks << " chunks" << std::endl;
    std::cout << "\nAll Timing Measurements:" << std::endl;
    std::cout << "-----------------------" << std::endl;
    for (const auto& timer : timers) {
        double avgTime = timer.second.totalTime / std::max(1, timer.second.count);
        std::cout << std::setw(20) << std::left << timer.first << ": "
                  << std::setw(10) << std::right << formatTime(timer.second.totalTime)
                  << " (avg: " << formatTime(avgTime) << ", count: "
                  << timer.second.count << ")" << std::endl;
    }
    std::cout << "\n===================================" << std::endl;
}

double Profiler::getTotalTime(const std::string& name) {
    if (timers.find(name) != timers.end()) {
        return timers[name].totalTime;
    }
    return 0.0;
}

std::string Profiler::formatTime(double seconds) {
    if (seconds < 0.001) {
        return std::to_string(static_cast<int>(seconds * 1000000)) + " µs";
    } else if (seconds < 1.0) {
        return std::to_string(static_cast<int>(seconds * 1000)) + " ms";
    } else {
        std::stringstream ss;
        ss << std::fixed << std::setprecision(3) << seconds << " s";
        return ss.str();
    }
}