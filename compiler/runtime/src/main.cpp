#include <iostream>
#include <fstream>
#include <string>
#include "runtime.h"

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <bytecode.jsonl> [options]" << std::endl;
        std::cerr << "Options:" << std::endl;
        std::cerr << "  --use-gpu-for-large   Enable GPU for large matrices (normally CPU-only)" << std::endl;
        std::cerr << "  --use-ane-for-large   Enable ANE for large matrices (normally CPU-only)" << std::endl;
        return 1;
    }
    std::string bytecodeFile = argv[1];

    std::ifstream file(bytecodeFile);
    if (!file.is_open()) {
        std::cerr << "Failed to open bytecode file" << std::endl;
        return 1;
    }
    Program program;
    std::string line;
    while (std::getline(file, line)) {
        try {
            auto json = nlohmann::json::parse(line);
            program.instructions.push_back(BytecodeInstruction::fromJson(json));
        } catch (const std::exception& e) {
            std::cerr << "Error parsing bytecode: " << e.what() << std::endl;
            return 1;
        }
    }
    Runtime runtime;
    runtime.execute(program);
    runtime.printProfiler();
    return 0;
}