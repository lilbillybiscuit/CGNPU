#include "metal_buffer_wrapper.h"
#include <Metal/Metal.h>
#include <iostream>
#include <Foundation/Foundation.h>
MTLBufferWrapper::MTLBufferWrapper() : metalBuffer(nullptr), metalDevice(nullptr) {
    @autoreleasepool {
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        if (device) {
            metalDevice = (void*)CFBridgingRetain(device);
        } else {
            std::cerr << "ERROR: Failed to create Metal device" << std::endl;
        }
    }
}
MTLBufferWrapper::~MTLBufferWrapper() {
    if (metalBuffer) {
        CFBridgingRelease(metalBuffer);
        metalBuffer = nullptr;
    }
    if (metalDevice) {
        CFBridgingRelease(metalDevice);
        metalDevice = nullptr;
    }
}
void* MTLBufferWrapper::createBuffer(std::size_t size, bool useSharedMemory) {
    if (!metalDevice) {
        std::cerr << "ERROR: Cannot create buffer, Metal device is null" << std::endl;
        return nullptr;
    }
    @autoreleasepool {
        if (metalBuffer) {
            CFBridgingRelease(metalBuffer);
            metalBuffer = nullptr;
        }
        id<MTLDevice> device = (__bridge id<MTLDevice>)metalDevice;
        MTLResourceOptions options = useSharedMemory ? 
            MTLResourceStorageModeShared : MTLResourceStorageModeManaged;
        id<MTLBuffer> buffer = [device newBufferWithLength:size options:options];
        if (buffer) {
            metalBuffer = (void*)CFBridgingRetain(buffer);
            return [buffer contents];
        } else {
            std::cerr << "ERROR: Failed to create Metal buffer" << std::endl;
            return nullptr;
        }
    }
}
void* MTLBufferWrapper::getBufferContents() {
    if (!metalBuffer) {
        return nullptr;
    }
    id<MTLBuffer> buffer = (__bridge id<MTLBuffer>)metalBuffer;
    return [buffer contents];
}
void MTLBufferWrapper::markRangeModified(std::size_t start, std::size_t length) {
    if (!metalBuffer) {
        return;
    }
    @autoreleasepool {
        id<MTLBuffer> buffer = (__bridge id<MTLBuffer>)metalBuffer;
        if (buffer.storageMode == MTLStorageModeManaged) {
            [buffer didModifyRange:NSMakeRange(start, length)];
        }
    }
}
void MTLBufferWrapper::syncContents() {
    if (!metalBuffer) {
        return;
    }
    @autoreleasepool {
        id<MTLBuffer> buffer = (__bridge id<MTLBuffer>)metalBuffer;
        if (buffer.storageMode == MTLStorageModeManaged) {
            [buffer didModifyRange:NSMakeRange(0, buffer.length)];
        }
    }
}