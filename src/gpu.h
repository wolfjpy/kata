// Thin Metal context: device, queue, offline-compiled library, pipeline cache.
#pragma once
#include <Foundation/Foundation.hpp>
#include <Metal/Metal.hpp>
#include <string>
#include <unordered_map>

struct Gpu {
    MTL::Device* dev = nullptr;
    MTL::CommandQueue* queue = nullptr;
    MTL::Library* lib = nullptr;
    std::unordered_map<std::string, MTL::ComputePipelineState*> psos;

    bool init(const char* metallibPath);
    MTL::ComputePipelineState* pipe(const char* kernelName);
    MTL::Buffer* alloc(size_t bytes);                                  // shared, zero-initialised
    MTL::Buffer* wrap(void* pagePtr, size_t pageRoundedLen);           // bytesNoCopy
};
