#include "instruction_set.h"
const char* instructionToString(Instruction inst) {
    switch (inst) {
        case Instruction::READ_INTEGER: return "READ_INTEGER";
        case Instruction::READ_MATRIX: return "READ_MATRIX";
        case Instruction::ALLOC_MATRIX: return "ALLOC_MATRIX";
        case Instruction::WRITE_MATRIX: return "WRITE_MATRIX";
        case Instruction::MATRIX_MULTIPLY: return "MATRIX_MULTIPLY";
        case Instruction::ADD: return "ADD";
        case Instruction::SUB: return "SUB";
        case Instruction::JUMP: return "JUMP";
        case Instruction::JUMP_IF_ZERO: return "JUMP_IF_ZERO";
        case Instruction::LOOP_BEGIN: return "LOOP_BEGIN";
        case Instruction::LOOP_END: return "LOOP_END";
        case Instruction::STORE: return "STORE";
        case Instruction::LOAD: return "LOAD";
        case Instruction::TERMINATE: return "TERMINATE";
        default: return "UNKNOWN";
    }
}
Instruction stringToInstruction(const std::string& str) {
    if (str == "READ_INTEGER") return Instruction::READ_INTEGER;
    if (str == "READ_MATRIX") return Instruction::READ_MATRIX;
    if (str == "ALLOC_MATRIX") return Instruction::ALLOC_MATRIX;
    if (str == "WRITE_MATRIX") return Instruction::WRITE_MATRIX;
    if (str == "MATRIX_MULTIPLY") return Instruction::MATRIX_MULTIPLY;
    if (str == "ADD") return Instruction::ADD;
    if (str == "SUB") return Instruction::SUB;
    if (str == "JUMP") return Instruction::JUMP;
    if (str == "JUMP_IF_ZERO") return Instruction::JUMP_IF_ZERO;
    if (str == "LOOP_BEGIN") return Instruction::LOOP_BEGIN;
    if (str == "LOOP_END") return Instruction::LOOP_END;
    if (str == "STORE") return Instruction::STORE;
    if (str == "LOAD") return Instruction::LOAD;
    if (str == "TERMINATE") return Instruction::TERMINATE;
    return Instruction::TERMINATE;  
}