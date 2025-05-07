#pragma once
#include <chrono>
#include <string>
#include <unordered_map>

class Profiler {
public:
    Profiler();
    void startTimer(const std::string& name);
    void stopTimer(const std::string& name);
    void recordZeroTime(const std::string& name);   
    void recordChunkExecution(const std::string& device, int chunkSize);
    void recordStealEvent(const std::string& fromDevice, const std::string& toDevice);
    void recordInitialAllocation(const std::string& device, int chunkCount, int totalChunks);
    void disableWorkStealing();
    void printReport();
    double getTotalTime(const std::string& name);
private:
    struct TimerData {
        std::chrono::steady_clock::time_point start;
        double totalTime;
        int count;
    };
    struct DeviceStats {
        int chunksProcessed;
        int totalElements;
        double totalTime;
        int allocatedChunks;    
        double percentUtilization;  
    };
    struct StealStats {
        int count;
    };
    std::unordered_map<std::string, TimerData> timers;
    std::unordered_map<std::string, DeviceStats> deviceStats;
    std::unordered_map<std::string, StealStats> stealStats;
    bool workStealingDisabled = false;
    std::string formatTime(double seconds);
};