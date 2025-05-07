#include "bytecode_generator.h"
#include <unordered_set>

BytecodeGenerator::BytecodeGenerator() {
}

Program BytecodeGenerator::generateFromIR(const IRGenerator& irGen) {
    Program program;
    program.instructions = irGen.getInstructions();
    program.matrices = irGen.getMatrices();
    addIOHandling(program);
    detectMatrixOperations(program);
    optimizeBytecode(program);
    return program;
}

void BytecodeGenerator::addIOHandling(Program& program) {
    // TODO : Remove
}

void BytecodeGenerator::detectMatrixOperations(Program& program) {
    // TODO : Remove
}

void BytecodeGenerator::optimizeBytecode(Program& program) {
    std::unordered_set<std::string> allocated;
    auto newEnd = std::remove_if(program.instructions.begin(), program.instructions.end(),
        [&allocated](const BytecodeInstruction& instr) {
            if (instr.operation == Instruction::ALLOC_MATRIX) {
                if (allocated.count(instr.label)) {
                    return true;  
                }
                allocated.insert(instr.label);
            }
            return false;
        });
    program.instructions.erase(newEnd, program.instructions.end());
}