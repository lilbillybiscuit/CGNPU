#pragma once
#include <cstddef>  
class MTLBufferWrapper {
public:
    MTLBufferWrapper();
    ~MTLBufferWrapper();
    void* createBuffer(std::size_t size, bool useSharedMemory);
    void* getBufferContents();
    void markRangeModified(std::size_t start, std::size_t length);
    void syncContents();
    void* getMetalBuffer() const { return metalBuffer; }
private:
    void* metalBuffer;       
    void* metalDevice;       
};