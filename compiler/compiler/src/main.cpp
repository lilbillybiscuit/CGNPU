#include <iostream>
#include <fstream>
#include <string>
#include <filesystem>
#include "parser.h"
#include "ir_generator.h"
#include "bytecode_generator.h"

int main(int argc, char* argv[]) {
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <input.cpp>" << std::endl;
        return 1;
    }
    std::string inputFile = argv[1];
    std::string outputFile = inputFile + ".jsonl";

    std::filesystem::path inputPath = std::filesystem::absolute(inputFile);
    std::filesystem::path tempPath = std::filesystem::temp_directory_path() / "matrix_mult.ll";
    std::cout << "Compiling: " << inputPath << std::endl;
    std::cout << "IR will be saved to: " << tempPath << std::endl;
    std::string compileCmd = "g++ -S -emit-llvm -O2 " + inputPath.string() + " -o " + tempPath.string();
    std::cout << "Running command: " << compileCmd << std::endl;

    if (system(compileCmd.c_str()) != 0) {
        std::cerr << "Failed to compile to LLVM IR" << std::endl;
        return 1;
    }
    if (!std::filesystem::exists(tempPath)) {
        std::cerr << "IR file was not created at: " << tempPath << std::endl;
        return 1;
    }

    std::cout << "IR file created successfully" << std::endl;
    IRGenerator irGen;
    std::cout << "Loading IR..." << std::endl;
    if (!irGen.loadIR(tempPath.string())) {
        std::cerr << "Failed to load LLVM IR" << std::endl;
        return 1;
    }

    BytecodeGenerator bcGen;
    Program program = bcGen.generateFromIR(irGen);
    std::cout << "Generated " << program.instructions.size() << " instructions" << std::endl;
    std::cout << "Output file: " << outputFile << std::endl;
    std::ofstream outFile(outputFile);
    if (!outFile.is_open()) {
        std::cerr << "Failed to open output file" << std::endl;
        return 1;
    }

    for (const auto& instr : program.instructions) {
        outFile << instr.toJson().dump() << std::endl;
    }

    outFile.close();
    std::cout << "Compiled successfully to " << outputFile << std::endl;
    std::filesystem::remove(tempPath);
    return 0;
}