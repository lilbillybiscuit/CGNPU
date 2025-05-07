#pragma once
#include <string>
#include <vector>
#include <nlohmann/json.hpp>
#include "instruction_set.h"

struct BytecodeInstruction {
    Instruction operation;
    std::vector<int> operands;
    std::string label;
    nlohmann::json toJson() const;
    static BytecodeInstruction fromJson(const nlohmann::json& j);
};

struct Matrix {
    int size;
    std::string name;
    bool isOutput;
};

struct Program {
    std::vector<BytecodeInstruction> instructions;
    std::vector<Matrix> matrices;
    nlohmann::json toJson() const;
    static Program fromJson(const nlohmann::json& j);
};

inline void to_json(nlohmann::json& j, const Instruction& inst) {
    j = instructionToString(inst);
}

inline void from_json(const nlohmann::json& j, Instruction& inst) {
    inst = stringToInstruction(j.get<std::string>());
}