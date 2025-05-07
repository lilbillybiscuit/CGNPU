#include "runtime.h"
#include <iostream>
#include <algorithm>
#include <stdexcept>

Runtime::Runtime() {
    deviceManager.initialize();
}

Runtime::~Runtime() {
    for (auto& pair : matrices) {
        delete pair.second;
    }
}

void Runtime::execute(const Program& program) {
    std::cout << "DEBUG: Starting execution of program with " << program.instructions.size() << " instructions" << std::endl;
    profiler.startTimer("total_execution");
    try {
        for (const auto& instr : program.instructions) {
            std::cout << "DEBUG: Executing instruction: " << instructionToString(instr.operation) << std::endl;
            profiler.startTimer(instructionToString(instr.operation));
            executeInstruction(instr);
            profiler.stopTimer(instructionToString(instr.operation));
            std::cout << "DEBUG: Completed instruction: " << instructionToString(instr.operation) << std::endl;
        }
    } catch (const std::exception& e) {
        std::cerr << "Runtime error: " << e.what() << std::endl;
        throw;
    }
    profiler.stopTimer("total_execution");
    std::cout << "DEBUG: Program execution complete" << std::endl;
}

void Runtime::executeInstruction(const BytecodeInstruction& instr) {
    switch (instr.operation) {
        case Instruction::READ_INTEGER: {
            int value;
            if (!(std::cin >> value)) {
                throw std::runtime_error("Failed to read integer");
            }
            variables["n"] = value;
            break;
        }
        case Instruction::READ_MATRIX: {
            int size = variables["n"];
            if (size <= 0) {
                throw std::runtime_error("Invalid matrix size");
            }
            readMatrix(size, instr.label);
            break;
        }
        case Instruction::ALLOC_MATRIX: {
            int size = variables["n"];
            if (size <= 0) {
                throw std::runtime_error("Invalid matrix size");
            }
            if (matrices.find(instr.label) == matrices.end()) {
                matrices[instr.label] = new MatrixBuffer(size);
            }
            break;
        }
        case Instruction::MATRIX_MULTIPLY: {
            executeMatrixMultiplication(instr);
            break;
        }
        case Instruction::WRITE_MATRIX: {
            std::string outputName = "result";  
            if (!instr.operands.empty()) {
            }
            writeMatrix(outputName);
            break;
        }
        case Instruction::TERMINATE: {
            for (auto& pair : matrices) {
                delete pair.second;
            }
            matrices.clear();
            break;
        }
        default:
            std::cerr << "Unhandled instruction: " << instructionToString(instr.operation) << std::endl;
            break;
    }
}

void Runtime::executeMatrixMultiplication(const BytecodeInstruction& instr) {
    std::cout << "DEBUG: Starting matrix multiplication" << std::endl;
    if (instr.operands.size() < 3) {
        throw std::runtime_error("Invalid matrix multiply operands");
    }
    std::string matrix1Name = "matrix1";
    std::string matrix2Name = "matrix2";
    std::string resultName = "result";
    std::cout << "DEBUG: Verifying matrices - " << matrix1Name << ", " << matrix2Name << ", " << resultName << std::endl;
    if (matrices.find(matrix1Name) == matrices.end() ||
        matrices.find(matrix2Name) == matrices.end() ||
        matrices.find(resultName) == matrices.end()) {
        throw std::runtime_error("Matrix not found for multiplication");
    }
    auto* matrix1 = matrices[matrix1Name];
    auto* matrix2 = matrices[matrix2Name];
    auto* result = matrices[resultName];
    std::cout << "DEBUG: Matrix sizes - A: " << matrix1->size << "x" << matrix1->size 
              << ", B: " << matrix2->size << "x" << matrix2->size 
              << ", Result: " << result->size << "x" << result->size << std::endl;
    std::cout << "DEBUG: Dispatching matrix multiplication to device manager" << std::endl;
    profiler.startTimer("matrix_multiplication");
    deviceManager.executeMatrixMultiplication(matrix1, matrix2, result);
    profiler.stopTimer("matrix_multiplication");
    std::cout << "DEBUG: Matrix multiplication completed" << std::endl;
}

void Runtime::readMatrix(int size, std::string name) {
    if (matrices.find(name) == matrices.end()) {
        matrices[name] = new MatrixBuffer(size);
    }
    int* data = matrices[name]->getCPUWritePtr();
    for (int i = 0; i < size * size; i++) {
        if (!(std::cin >> data[i])) {
            matrices[name]->releaseCPUAccess();
            throw std::runtime_error("Failed to read matrix element");
        }
    }
    matrices[name]->releaseCPUAccess();
}

void Runtime::writeMatrix(const std::string& name) {
    if (matrices.find(name) == matrices.end()) {
        throw std::runtime_error("Matrix not found for output");
    }
    auto* matrix = matrices[name];
    int* data = matrix->getCPUReadPtr();
    for (int i = 0; i < matrix->size; i++) {
        for (int j = 0; j < matrix->size; j++) {
            std::cout << data[i * matrix->size + j];
            if (j < matrix->size - 1) std::cout << " ";
        }
        std::cout << std::endl;
    }
    matrix->releaseCPUAccess();
}

void Runtime::printProfiler() {
    std::cout << "\n>> HETEROGENEOUS EXECUTION PERFORMANCE REPORT" << std::endl;
    std::cout << "   Matrix Operations Performance Analysis" << std::endl;
    std::cout << "   -------------------------------------" << std::endl;
    auto dmProfiler = deviceManager.getProfiler();
    if (dmProfiler) {
        dmProfiler->printReport();
    }
}