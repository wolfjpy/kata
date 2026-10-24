#define NS_PRIVATE_IMPLEMENTATION
#define MTL_PRIVATE_IMPLEMENTATION
#define CA_PRIVATE_IMPLEMENTATION
#include <Foundation/Foundation.hpp>
#include <Metal/Metal.hpp>
#include <cstdio>
#include <string>
#include <vector>
#include <cstring>
#include <cstdlib>

static const char* HDR = R"MSL(
#include <metal_stdlib>
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace metal;
using namespace mpp::tensor_ops;
)MSL";

// Kernel generator: TA/TB/TC element types, tile TM x TN, simdgroups SG
static std::string kern(const char* name, const char* TA, const char* TB, const char* TC, int TM, int TN, int SG, const char* ptrB = nullptr) {
  char buf[4096];
  snprintf(buf, sizeof buf, R"MSL(
kernel void %s(device %s* A [[buffer(0)]], device %s* B [[buffer(1)]], device %s* C [[buffer(2)]],
               constant uint3& dims [[buffer(3)]], uint2 tgid [[threadgroup_position_in_grid]]) {
  int M = dims.x, N = dims.y, K = dims.z;
  tensor<device %s, dextents<int32_t,2>, tensor_inline> tA(A, dextents<int32_t,2>(K, M));
  tensor<device %s, dextents<int32_t,2>, tensor_inline> tB((device %s*)B, dextents<int32_t,2>(N, K));
  tensor<device %s, dextents<int32_t,2>, tensor_inline> tC(C, dextents<int32_t,2>(N, M));
  constexpr auto desc = matmul2d_descriptor(%d, %d, static_cast<int>(dynamic_extent));
  matmul2d<desc, execution_simdgroups<%d>> op;
  auto mA = tA.slice(0, tgid.y*%d);
  auto mB = tB.slice(tgid.x*%d, 0);
  auto mC = tC.slice(tgid.x*%d, tgid.y*%d);
  op.run(mA, mB, mC);
}
)MSL", name, TA, TB, TC, TA, TB, TB, TC, TM, TN, SG, TM, TN, TN, TM);
  return std::string(HDR) + buf;
}

struct Cfg { const char* name; const char* TA; const char* TB; const char* TC; int bytesA, bytesB, bytesC; int TM, TN, SG; };

int main(int argc, char** argv) {
  NS::AutoreleasePool* pool = NS::AutoreleasePool::alloc()->init();
  MTL::Device* dev = MTL::CreateSystemDefaultDevice();
  MTL::CommandQueue* q = dev->newCommandQueue();
  const int M = 4096, N = 4096, K = 4096;
  std::vector<Cfg> cfgs = {
    {"hh_f","half","half","float",2,2,4,64,128,4},
    {"hh_f","half","half","float",2,2,4,32,128,4},
    {"hh_f","half","half","float",2,2,4,128,32,4},
    {"hh_f","half","half","float",2,2,4,64,96,4},
    {"hh_f","half","half","float",2,2,2,64,64,2},
    {"hh_f","half","half","float",2,2,8,64,128,8},
    {"hh_f","half","half","float",2,2,8,128,128,8},
    {"hh_f","half","half","float",2,2,1,32,64,1},
    {"ii_i","int8_t","int8_t","int",1,1,4,64,256,4},
    {"ii_i","int8_t","int8_t","int",1,1,4,32,128,4},
    {"ii_i","int8_t","int8_t","int",1,1,8,64,128,8},
    {"ii_i","int8_t","int8_t","int",1,1,2,64,128,2},
    {"ii_i","int8_t","int8_t","int",1,1,4,64,128,4},
    {"ii_i","int8_t","int8_t","int",1,1,8,128,128,8},
  };
  int idx = 0;
  for (auto& c : cfgs) {
    std::string src = kern("k", c.TA, c.TB, c.TC, c.TM, c.TN, c.SG);
    MTL::CompileOptions* opt = MTL::CompileOptions::alloc()->init();
    opt->setLanguageVersion((MTL::LanguageVersion)262145); // Metal 4.1
    NS::Error* err = nullptr;
    MTL::Library* lib = dev->newLibrary(NS::String::string(src.c_str(), NS::UTF8StringEncoding), opt, &err);
    if (!lib) { printf("[%s %dx%d] COMPILE FAIL: %s\n", c.name, c.TM, c.TN, err ? err->localizedDescription()->utf8String() : "?"); continue; }
    MTL::Function* fn = lib->newFunction(NS::String::string("k", NS::UTF8StringEncoding));
    MTL::ComputePipelineState* pso = dev->newComputePipelineState(fn, &err);
    if (!pso) { printf("[%s %dx%d] PSO FAIL: %s\n", c.name, c.TM, c.TN, err ? err->localizedDescription()->utf8String() : "?"); continue; }
    MTL::Buffer* A = dev->newBuffer((size_t)M * K * c.bytesA, MTL::ResourceStorageModeShared);
    MTL::Buffer* B = dev->newBuffer((size_t)K * N * c.bytesB, MTL::ResourceStorageModeShared);
    MTL::Buffer* C = dev->newBuffer((size_t)M * N * c.bytesC, MTL::ResourceStorageModeShared);
    memset(A->contents(), 0x11, A->length()); memset(B->contents(), 0x11, B->length());
    uint32_t dims[4] = {M, N, K, 0};
    double best = 1e30;
    for (int it = 0; it < 8; it++) {
      MTL::CommandBuffer* cb = q->commandBuffer();
      MTL::ComputeCommandEncoder* e = cb->computeCommandEncoder();
      e->setComputePipelineState(pso);
      e->setBuffer(A, 0, 0); e->setBuffer(B, 0, 1); e->setBuffer(C, 0, 2); e->setBytes(dims, sizeof dims, 3);
      e->dispatchThreadgroups(MTL::Size(N / c.TN, M / c.TM, 1), MTL::Size(32 * c.SG, 1, 1));
      e->endEncoding(); cb->commit(); cb->waitUntilCompleted();
      double t = cb->GPUEndTime() - cb->GPUStartTime();
      if (it >= 2 && t < best) best = t;
    }
    double flops = 2.0 * M * N * K;
    printf("[%-5s tile %3dx%-3d SG%d] best %.2f ms  => %.2f T%s/s\n", c.name, c.TM, c.TN, c.SG, best * 1e3, flops / best / 1e12, c.TA[0]=='c' ? "OPS" : "FLOPS");
    A->release(); B->release(); C->release(); pso->release(); fn->release(); lib->release(); opt->release();
    idx++;
  }
  pool->release();
  return 0;
}
