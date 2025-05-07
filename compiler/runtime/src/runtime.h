#pragma once
#include <unordered_map>
#include <vector>
#include <string>
#include "bytecode_format.h"
#include "device_manager.h"
#include "profiler.h"
#include "matrix_utils.h"

class Runtime {
public:
    Runtime();
    ~Runtime();
    void execute(const Program& program);
    void printProfiler();
    DeviceManager* getDeviceManager() { return &deviceManager; }
private:
    DeviceManager deviceManager;
    Profiler profiler;
    std::unordered_map<std::string, MatrixBuffer*> matrices;
    std::unordered_map<std::string, int> variables;
    void executeInstruction(const BytecodeInstruction& instr);
    void executeMatrixMultiplication(const BytecodeInstruction& instr);
    void readMatrix(int size, std::string name);
    void writeMatrix(const std::string& name);
};