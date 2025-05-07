#pragma once
#include <string>
#include <vector>
#include <llvm/IR/Module.h>
#include <llvm/IR/LLVMContext.h>
#include <llvm/IRReader/IRReader.h>
#include <llvm/Support/SourceMgr.h>

class LLVMParser {
public:
    LLVMParser();
    ~LLVMParser();
    bool parseIR(const std::string& filename);
    llvm::Module* getModule() { return module.get(); }
    llvm::LLVMContext& getContext() { return context; }
private:
    llvm::LLVMContext context;
    std::unique_ptr<llvm::Module> module;
};