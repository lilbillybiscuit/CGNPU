#pragma once
#include <vector>
#include <atomic>
#include <mutex>

class MTLBufferWrapper;
enum class MemoryAccessState {
    CPU_READING,      
    CPU_WRITING,      
    GPU_READING,      
    GPU_WRITING,      
    ANE_READING,      
    ANE_WRITING,      
    SHARED            
};

struct MatrixBuffer {
    int size;                            
    std::mutex accessMutex;              
    std::atomic<MemoryAccessState> state;  
    void* unifiedBuffer;                 
    MTLBufferWrapper* metalBuffer;       
    void* aneModel;                      
    MatrixBuffer(int size);
    ~MatrixBuffer();
    void* getUnifiedBufferPtr();         
    int* getCPUReadPtr();                
    int* getCPUWritePtr();               
    void releaseCPUAccess();             
    void prepareForGPUAccess(bool readOnly);   
    void releaseGPUAccess();             
    void prepareForANEAccess(bool readOnly);   
    void releaseANEAccess();             
    void syncToDevice() const;
    void syncFromDevice();
    void releaseResources();             
    int get(int row, int col) const;     
    void set(int row, int col, int value);  
    int& operator[](size_t index);       
    const int& operator[](size_t index) const;  
    int* getRawData();                   
};

struct WorkChunk {
    int startRow;
    int endRow;
    int startCol;
    int endCol;
    WorkChunk(int sr, int er, int sc, int ec)
        : startRow(sr), endRow(er), startCol(sc), endCol(ec) {}
};

std::vector<WorkChunk> createWorkChunks(int matrixSize, int numChunks);

void partitionChunks(std::vector<WorkChunk>& chunks,
                    std::vector<WorkChunk>& cpu,
                    std::vector<WorkChunk>& gpu,
                    std::vector<WorkChunk>& ane);