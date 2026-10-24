#define NS_PRIVATE_IMPLEMENTATION
#define MTL_PRIVATE_IMPLEMENTATION
#define CA_PRIVATE_IMPLEMENTATION
#include <Foundation/Foundation.hpp>
#include <Metal/Metal.hpp>
#include <cstdio>
int main(){ auto* dev = MTL::CreateSystemDefaultDevice();
  printf("device: %s | apple11=%d metal4=%d\n", dev->name()->utf8String(), (int)dev->supportsFamily(MTL::GPUFamilyApple11), (int)dev->supportsFamily(MTL::GPUFamilyMetal4));
  printf("recommendedMaxWorkingSet MB: %llu | maxBuffer MB: %llu | maxTGmem KB: %llu\n", (unsigned long long)(dev->recommendedMaxWorkingSetSize()>>20), (unsigned long long)(dev->maxBufferLength()>>20), (unsigned long long)(dev->maxThreadgroupMemoryLength()/1024));
  auto* o = MTL::CompileOptions::alloc()->init(); o->setLanguageVersion(MTL::LanguageVersion4_1); NS::Error* e=nullptr;
  auto* l = dev->newLibrary(NS::String::string("#include <metal_stdlib>\nkernel void k(){}\n", NS::UTF8StringEncoding), o, &e);
  printf("LanguageVersion4_1 compile: %s\n", l ? "OK" : "FAIL");
  auto* q4 = dev->newMTL4CommandQueue(); printf("MTL4CommandQueue: %s\n", q4 ? "OK" : "FAIL");
  auto* alloc = dev->newCommandAllocator(); printf("MTL4CommandAllocator: %s\n", alloc ? "OK" : "FAIL");
  auto* cb = dev->newCommandBuffer(); printf("MTL4CommandBuffer: %s\n", cb ? "OK" : "FAIL"); }
