#include "bytecode_format.h"

nlohmann::json BytecodeInstruction::toJson() const {
    nlohmann::json j;
    j["operation"] = operation;
    j["operands"] = operands;
    j["label"] = label;
    return j;
}

BytecodeInstruction BytecodeInstruction::fromJson(const nlohmann::json& j) {
    BytecodeInstruction instr;
    instr.operation = j["operation"];
    instr.operands = j["operands"].get<std::vector<int>>();
    instr.label = j["label"].get<std::string>();
    return instr;
}

nlohmann::json Program::toJson() const {
    nlohmann::json j;
    j["instructions"] = nlohmann::json::array();
    for (const auto& instr : instructions) {
        j["instructions"].push_back(instr.toJson());
    }
    j["matrices"] = nlohmann::json::array();
    for (const auto& mat : matrices) {
        nlohmann::json matJson;
        matJson["size"] = mat.size;
        matJson["name"] = mat.name;
        matJson["isOutput"] = mat.isOutput;
        j["matrices"].push_back(matJson);
    }
    return j;
}

Program Program::fromJson(const nlohmann::json& j) {
    Program prog;
    for (const auto& instrJson : j["instructions"]) {
        prog.instructions.push_back(BytecodeInstruction::fromJson(instrJson));
    }
    for (const auto& matJson : j["matrices"]) {
        Matrix mat;
        mat.size = matJson["size"];
        mat.name = matJson["name"];
        mat.isOutput = matJson["isOutput"];
        prog.matrices.push_back(mat);
    }
    return prog;
}