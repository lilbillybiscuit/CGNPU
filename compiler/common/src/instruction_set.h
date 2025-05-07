#pragma once
#include <string>
enum class Instruction {
    READ_INTEGER,       
    READ_MATRIX,        
    ALLOC_MATRIX,       
    WRITE_MATRIX,       
    MATRIX_MULTIPLY,    
    ADD,                
    SUB,                
    JUMP,               
    JUMP_IF_ZERO,       
    LOOP_BEGIN,         
    LOOP_END,           
    STORE,              
    LOAD,               
    TERMINATE           
};
const char* instructionToString(Instruction inst);
Instruction stringToInstruction(const std::string& str);