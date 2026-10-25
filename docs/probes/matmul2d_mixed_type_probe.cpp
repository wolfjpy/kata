#define NS_PRIVATE_IMPLEMENTATION
#define MTL_PRIVATE_IMPLEMENTATION
#define CA_PRIVATE_IMPLEMENTATION
#include <Foundation/Foundation.hpp>
#include <Metal/Metal.hpp>
#include <cstdio>
#include <string>
#include <cstring>
static std::string trycompile(MTL::Device* dev, const char* label, const std::string& src, MTL::Library** out) {
  MTL::CompileOptions* opt = MTL::CompileOptions::alloc()->init();
  opt->setLanguageVersion((MTL::LanguageVersion)262145);
  NS::Error* err = nullptr;
  MTL::Library* lib = dev->newLibrary(NS::String::string(src.c_str(), NS::UTF8StringEncoding), opt, &err);
  if (!lib) { std::string m = err ? err->localizedDescription()->utf8String() : "?"; size_t p = m.find("error:"); printf("[%s] FAIL: %s\n", label, m.substr(p==std::string::npos?0:p, 300).c_str()); return "fail"; }
  printf("[%s] compiled OK\n", label); *out = lib; return "ok";
}
int main() {
  NS::AutoreleasePool* pool = NS::AutoreleasePool::alloc()->init();
  MTL::Device* dev = MTL::CreateSystemDefaultDevice();
  MTL::CommandQueue* q = dev->newCommandQueue();
  const char* HDR = "#include <metal_stdlib>\n#include <metal_tensor>\n#include <metal_packed_numeric>\n#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>\nusing namespace metal; using namespace mpp::tensor_ops;\n";
  struct V { const char* label; const char* aT; const char* bT; const char* cT; const char* bPtr; int bBitsPerElem; } vs[] = {
    {"half x half -> float", "half", "half", "float", "half", 16},
    {"half x int8 -> float", "half", "int8_t", "float", "int8_t", 8},
    {"half x int8 -> half", "half", "int8_t", "half", "int8_t", 8},
    {"int8 x int8 -> int", "int8_t", "int8_t", "int", "int8_t", 8},
    {"half x half -> float", "half", "half", "float", "half", 16},
    {"half x int8 -> float", "half", "int8_t", "float", "int8_t", 8},
  };
  for (auto& v : vs) {
    char buf[3000];
    snprintf(buf, sizeof buf,
      "kernel void k(device %s* A [[buffer(0)]], device %s* B [[buffer(1)]], device %s* C [[buffer(2)]], constant uint3& dims [[buffer(3)]], uint2 tgid [[threadgroup_position_in_grid]]) {\n"
      "  int M=dims.x,N=dims.y,K=dims.z;\n"
      "  tensor<device %s, dextents<int32_t,2>, tensor_inline> tA(A, dextents<int32_t,2>(K, M));\n"
      "  tensor<device %s, dextents<int32_t,2>, tensor_inline> tB((device %s*)B, dextents<int32_t,2>(N, K));\n"
      "  tensor<device %s, dextents<int32_t,2>, tensor_inline> tC(C, dextents<int32_t,2>(N, M));\n"
      "  constexpr auto desc = matmul2d_descriptor(64, 128, static_cast<int>(dynamic_extent));\n"
      "  matmul2d<desc, execution_simdgroups<4>> op;\n"
      "  auto mA = tA.slice(0, tgid.y*64); auto mB = tB.slice(tgid.x*128, 0); auto mC = tC.slice(tgid.x*128, tgid.y*64);\n"
      "  op.run(mA, mB, mC);\n}\n",
      v.aT, v.bPtr, v.cT, v.aT, v.bT, v.bT, v.cT);
    std::string src = std::string(HDR) + buf;
    MTL::Library* lib = nullptr;
    if (trycompile(dev, v.label, src, &lib) != "ok") continue;
    NS::Error* err = nullptr;
    MTL::Function* fn = lib->newFunction(NS::String::string("k", NS::UTF8StringEncoding));
    MTL::ComputePipelineState* pso = dev->newComputePipelineState(fn, &err);
    if (!pso) { printf("   PSO fail\n"); continue; }
    const int M = 4096, N = 4096, K = 4096;
    int aB = (v.aT[0]=='h') ? 2 : 1, cB = 4; if (v.cT[0]=='h') cB = 2;
    MTL::Buffer* A = dev->newBuffer((size_t)M*K*aB, MTL::ResourceStorageModeShared);
    MTL::Buffer* B = dev->newBuffer((size_t)K*N*v.bBitsPerElem/8, MTL::ResourceStorageModeShared);
    MTL::Buffer* C = dev->newBuffer((size_t)M*N*cB, MTL::ResourceStorageModeShared);
    memset(A->contents(), 0x11, A->length()); memset(B->contents(), 0x11, B->length());
    uint32_t dims[4] = {M,N,K,0}; double best = 1e30;
    for (int it = 0; it < 8; it++) {
      MTL::CommandBuffer* cb = q->commandBuffer(); MTL::ComputeCommandEncoder* e = cb->computeCommandEncoder();
      e->setComputePipelineState(pso); e->setBuffer(A,0,0); e->setBuffer(B,0,1); e->setBuffer(C,0,2); e->setBytes(dims,sizeof dims,3);
      e->dispatchThreadgroups(MTL::Size(N/128, M/64, 1), MTL::Size(128,1,1)); e->endEncoding(); cb->commit(); cb->waitUntilCompleted();
      if (cb->status() != MTL::CommandBufferStatusCompleted) { printf("   run error\n"); break; }
      double t = cb->GPUEndTime()-cb->GPUStartTime(); if (it>=2 && t<best) best=t;
    }
    printf("   tile 64x128 SG4: %.2f ms => %.2f T(FL)OPS\n", best*1e3, 2.0*M*N*K/best/1e12);
  }
  pool->release(); return 0;
}
