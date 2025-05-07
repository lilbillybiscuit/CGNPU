#include "parser.h"

LLVMParser::LLVMParser() = default;

LLVMParser::~LLVMParser() = default;

bool LLVMParser::parseIR(const std::string& filename) {
    llvm::SMDiagnostic error;
    module = llvm::parseIRFile(filename, error, context);
    if (!module) {
        error.print("parser", llvm::errs());
        return false;
    }
    return true;
}