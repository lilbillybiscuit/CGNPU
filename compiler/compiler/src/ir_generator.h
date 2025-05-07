#pragma once
#include <string>
#include <vector>
#include <map>
#include "parser.h"
#include "bytecode_format.h"

class IRGenerator {
public:
    IRGenerator();
    bool loadIR(const std::string& filename);
    bool detectMatrixOperations();
    const std::vector<BytecodeInstruction>& getInstructions() const { return instructions; }
    const std::vector<Matrix>& getMatrices() const { return matrices; }
private:
    struct IROperation {
        enum Type {
            INPUT_INT,
            INPUT_MATRIX_ELEMENT,
            OUTPUT_INT,
            MATRIX_ALLOC,
            MATRIX_MULTIPLY,
            LOOP_HEADER
        };
        Type type;
        llvm::BasicBlock* block;
        int loopDepth;
    };
    LLVMParser parser;
    std::vector<BytecodeInstruction> instructions;
    std::vector<Matrix> matrices;
    bool analyzeFunction(llvm::Function* func);
    void analyzeBlock(llvm::BasicBlock* bb, std::vector<IROperation>& operations);
    bool isMatrixMultiplicationBlock(llvm::BasicBlock* bb);
    void generateBytecodeFromOperations(const std::vector<IROperation>& operations);
    void createMatrixInstruction(int size1, int size2, int resultSize);
    std::map<llvm::BasicBlock*, std::vector<llvm::BasicBlock*>> buildControlFlowGraph(llvm::Function* func);
};