#define NS_PRIVATE_IMPLEMENTATION
#define MTL_PRIVATE_IMPLEMENTATION
#define CA_PRIVATE_IMPLEMENTATION
#include "gpu.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>

bool Gpu::init(const char* metallibPath) {
    dev = MTL::CreateSystemDefaultDevice(); if (!dev) { fprintf(stderr, "no Metal device\n"); return false; }
    queue = dev->newCommandQueue();
    NS::Error* err = nullptr;
    lib = dev->newLibrary(NS::String::string(metallibPath, NS::UTF8StringEncoding), &err);
    if (!lib) { fprintf(stderr, "cannot load %s: %s\n", metallibPath, err ? err->localizedDescription()->utf8String() : "?"); return false; }
    return true;
}
MTL::ComputePipelineState* Gpu::pipe(const char* name) {
    auto it = psos.find(name); if (it != psos.end()) return it->second;
    NS::Error* err = nullptr;
    MTL::Function* fn = lib->newFunction(NS::String::string(name, NS::UTF8StringEncoding));
    if (!fn) { fprintf(stderr, "missing kernel %s\n", name); exit(1); }
    MTL::ComputePipelineState* p = dev->newComputePipelineState(fn, &err);
    if (!p) { fprintf(stderr, "pipeline %s failed: %s\n", name, err ? err->localizedDescription()->utf8String() : "?"); exit(1); }
    fn->release(); psos[name] = p; return p;
}
MTL::Buffer* Gpu::alloc(size_t bytes) {
    MTL::Buffer* b = dev->newBuffer(bytes, MTL::ResourceStorageModeShared);
    if (!b) { fprintf(stderr, "buffer alloc of %zu bytes failed\n", bytes); exit(1); }
    memset(b->contents(), 0, bytes); return b;
}
MTL::Buffer* Gpu::wrap(void* p, size_t len) {
    MTL::Buffer* b = dev->newBuffer(p, len, MTL::ResourceStorageModeShared, nullptr);
    if (!b) { fprintf(stderr, "bytesNoCopy wrap of %zu bytes failed\n", len); exit(1); }
    return b;
}
