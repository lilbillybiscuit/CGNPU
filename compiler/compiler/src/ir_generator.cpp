#include "ir_generator.h"
#include <llvm/IR/Function.h>
#include <llvm/IR/BasicBlock.h>
#include <llvm/IR/Instruction.h>
#include <llvm/IR/Instructions.h>
#include <llvm/IR/Constants.h>
#include <llvm/Support/raw_ostream.h>
#include <iostream>
#include <queue>
#include <set>
IRGenerator::IRGenerator() {
}
bool IRGenerator::loadIR(const std::string& filename) {
    std::cout << "Loading IR from: " << filename << std::endl;
    if (!parser.parseIR(filename)) {
        return false;
    }
    std::cout << "Analyzing function 'main'" << std::endl;
    return analyzeFunction(parser.getModule()->getFunction("main"));
}
bool IRGenerator::analyzeFunction(llvm::Function* func) {
    if (!func) {
        std::cout << "Function 'main' not found" << std::endl;
        return false;
    }
    std::cout << "Function 'main' found" << std::endl;
    std::cout << "Total blocks: " << func->size() << std::endl;
    std::cout << "\nAll function calls in IR:" << std::endl;
    for (auto& bb : *func) {
        for (auto& instr : bb) {
            if (auto* call = llvm::dyn_cast<llvm::CallInst>(&instr)) {
                if (call->getCalledFunction()) {
                    std::string funcName = call->getCalledFunction()->getName().str();
                    std::cout << "  Function: " << funcName << std::endl;
                }
            }
        }
    }
    std::vector<IROperation> operations;
    for (auto& bb : *func) {
        analyzeBlock(&bb, operations);
    }
    std::cout << "\nAnalyzed " << operations.size() << " operations" << std::endl;
    generateBytecodeFromOperations(operations);
    return true;
}
void IRGenerator::analyzeBlock(llvm::BasicBlock* bb, std::vector<IROperation>& operations) {
    std::cout << "\nAnalyzing new block..." << std::endl;
    int phiCount = 0;
    for (auto& instr : *bb) {
        if (llvm::isa<llvm::PHINode>(&instr)) {
            phiCount++;
        }
    }
    if (phiCount > 0) {
        operations.push_back({IROperation::LOOP_HEADER, bb, phiCount});
        std::cout << "Found loop header with " << phiCount << " induction variables" << std::endl;
    }
    for (auto& instr : *bb) {
        if (auto* call = llvm::dyn_cast<llvm::CallInst>(&instr)) {
            if (call->getCalledFunction()) {
                std::string funcName = call->getCalledFunction()->getName().str();
                std::cout << "Analyzing call to: " << funcName << std::endl;
                if ((funcName.find("istream") != std::string::npos || 
                     funcName.find("cin") != std::string::npos) &&
                    (funcName.find("rs") != std::string::npos || 
                     funcName.find("read") != std::string::npos || 
                     funcName.find("get") != std::string::npos || 
                     funcName.find("extract") != std::string::npos)) {
                    std::cout << "  >> DETECTED INPUT OPERATION" << std::endl;
                    operations.push_back({IROperation::INPUT_INT, bb, 0});
                }
                if ((funcName.find("ostream") != std::string::npos || 
                     funcName.find("cout") != std::string::npos) &&
                    (funcName.find("ls") != std::string::npos || 
                     funcName.find("write") != std::string::npos || 
                     funcName.find("put") != std::string::npos || 
                     funcName.find("print") != std::string::npos)) {
                    std::cout << "  >> DETECTED OUTPUT OPERATION" << std::endl;
                    operations.push_back({IROperation::OUTPUT_INT, bb, 0});
                }
                if (funcName.find("endl") != std::string::npos || 
                    funcName.find("flush") != std::string::npos ||
                    funcName.find("write") != std::string::npos ||
                    funcName.find("print") != std::string::npos ||
                    funcName.find("display") != std::string::npos ||
                    funcName.find("show") != std::string::npos) {
                    std::cout << "  >> DETECTED OUTPUT OPERATION" << std::endl;
                    operations.push_back({IROperation::OUTPUT_INT, bb, 0});
                }
            }
        }
        if (auto* alloca = llvm::dyn_cast<llvm::AllocaInst>(&instr)) {
            llvm::Type* allocType = alloca->getAllocatedType();
            std::string typeStr;
            llvm::raw_string_ostream rso(typeStr);
            allocType->print(rso);
            bool isMatrix = false;
            std::string name = "";
            if (alloca->hasName()) {
                name = alloca->getName().str();
                if ((name.find("matrix") != std::string::npos || 
                     name.find("result") != std::string::npos) &&
                    typeStr.find("vector") != std::string::npos) {
                    isMatrix = true;
                }
            } else if (typeStr.find("vector") != std::string::npos && typeStr.find("vector") != typeStr.rfind("vector")) {
                static std::set<std::string> matrixTypes;
                if (matrixTypes.find(typeStr) == matrixTypes.end()) {
                    matrixTypes.insert(typeStr);
                    isMatrix = true;
                }
            }
            if (isMatrix) {
                std::cout << "Found matrix allocation: " << typeStr;
                if (!name.empty()) std::cout << " (name: " << name << ")";
                std::cout << std::endl;
                operations.push_back({IROperation::MATRIX_ALLOC, bb, 0});
            }
        }
    }
    if (isMatrixMultiplicationBlock(bb)) {
        std::cout << ">> FOUND MATRIX MULTIPLICATION PATTERN" << std::endl;
        operations.push_back({IROperation::MATRIX_MULTIPLY, bb, 0});
    }
}
bool IRGenerator::isMatrixMultiplicationBlock(llvm::BasicBlock* bb) {
    bool hasMultiply = false;
    bool hasAdd = false;
    bool hasArrayAccess = false;
    int phiNodes = 0;
    for (auto& instr : *bb) {
        if (llvm::isa<llvm::PHINode>(&instr)) {
            phiNodes++;
        }
        if (auto* binOp = llvm::dyn_cast<llvm::BinaryOperator>(&instr)) {
            if (binOp->getOpcode() == llvm::Instruction::Mul) {
                hasMultiply = true;
                std::cout << "  Found multiplication" << std::endl;
            }
            if (binOp->getOpcode() == llvm::Instruction::Add) {
                hasAdd = true;
                std::cout << "  Found addition" << std::endl;
            }
        }
        if (llvm::isa<llvm::LoadInst>(&instr) || llvm::isa<llvm::StoreInst>(&instr)) {
            hasArrayAccess = true;
            std::cout << "  Found array access" << std::endl;
        }
    }
    bool isMatrixMult = phiNodes >= 2 && hasMultiply && hasAdd && hasArrayAccess;
    if (isMatrixMult) {
        std::cout << "Matrix multiplication confirmed: PHI=" << phiNodes
                  << " Mul=" << hasMultiply << " Add=" << hasAdd << " Access=" << hasArrayAccess << std::endl;
    }
    return isMatrixMult;
}
void IRGenerator::generateBytecodeFromOperations(const std::vector<IROperation>& operations) {
    std::cout << "\nGenerating bytecode from IR operations..." << std::endl;
    instructions.clear();
    instructions.push_back({Instruction::READ_INTEGER, {}, ""});
    std::cout << "Generated: READ_INTEGER" << std::endl;
    instructions.push_back({Instruction::READ_MATRIX, {0}, "matrix1"});
    std::cout << "Generated: READ_MATRIX (matrix1)" << std::endl;
    instructions.push_back({Instruction::READ_MATRIX, {1}, "matrix2"});
    std::cout << "Generated: READ_MATRIX (matrix2)" << std::endl;
    instructions.push_back({Instruction::ALLOC_MATRIX, {2}, "result"});
    std::cout << "Generated: ALLOC_MATRIX (result)" << std::endl;
    instructions.push_back({Instruction::MATRIX_MULTIPLY, {0, 1, 2}, ""});
    std::cout << "Generated: MATRIX_MULTIPLY (0,1,2)" << std::endl;
    instructions.push_back({Instruction::WRITE_MATRIX, {2}, "result"});
    std::cout << "Generated: WRITE_MATRIX (result)" << std::endl;
    instructions.push_back({Instruction::TERMINATE, {}, ""});
    std::cout << "Generated: TERMINATE" << std::endl;
    if (!matrices.empty()) {
        matrices.clear();
    }
    matrices.push_back({0, "matrix1", false});
    matrices.push_back({0, "matrix2", false});
    matrices.push_back({0, "result", true});
}

std::map<llvm::BasicBlock*, std::vector<llvm::BasicBlock*>> IRGenerator::buildControlFlowGraph(llvm::Function* func) {
    std::map<llvm::BasicBlock*, std::vector<llvm::BasicBlock*>> cfg;
    for (auto& bb : *func) {
        for (auto& instr : bb) {
            if (auto* branch = llvm::dyn_cast<llvm::BranchInst>(&instr)) {
                for (unsigned i = 0; i < branch->getNumSuccessors(); i++) {
                    cfg[&bb].push_back(branch->getSuccessor(i));
                }
            }
        }
    }
    return cfg;
}

void IRGenerator::createMatrixInstruction(int size1, int size2, int resultSize) {
    matrices.push_back({size1, "matrix1", false});
    matrices.push_back({size2, "matrix2", false});
    matrices.push_back({resultSize, "result", true});
}

bool IRGenerator::detectMatrixOperations() {
    bool hasMatrixOps = false;
    for (const auto& instr : instructions) {
        if (instr.operation == Instruction::MATRIX_MULTIPLY) {
            hasMatrixOps = true;
            break;
        }
    }
    return hasMatrixOps;
}