#pragma once
#include "ir_generator.h"
#include "bytecode_format.h"
#include <vector>
class BytecodeGenerator {
public:
    BytecodeGenerator();
    Program generateFromIR(const IRGenerator& irGen);
    void optimizeBytecode(Program& program);
private:
    void transformInstructions(std::vector<BytecodeInstruction>& instructions);
    void addIOHandling(Program& program);
    void detectMatrixOperations(Program& program);
};