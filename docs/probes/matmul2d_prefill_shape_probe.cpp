// Prefill-shape probe: does weight-tile reuse (grid order, weight layout) decide whether fp16-expanded weights stay compute-bound?
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
#include <cstdint>
static const char* HDR = "#include <metal_stdlib>\n#include <metal_tensor>\n#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>\nusing namespace metal; using namespace mpp::tensor_ops;\n";
static std::string kern(const char* TA, const char* TB, const char* TC, int TM, int TN, int SG, int ORDER, int LAYOUT) {
  char b[4096];
  snprintf(b, sizeof b,
  "kernel void k(device %s* A [[buffer(0)]], device %s* B [[buffer(1)]], device %s* C [[buffer(2)]], constant uint3& dims [[buffer(3)]], uint2 tgid [[threadgroup_position_in_grid]]) {\n"
  " int M=dims.x,N=dims.y,K=dims.z; uint mt, nt; if (%d==0) { nt=tgid.x; mt=tgid.y; } else { mt=tgid.x; nt=tgid.y; }\n"
  " tensor<device %s, dextents<int32_t,2>, tensor_inline> tA(A, dextents<int32_t,2>(K, M));\n"
  " tensor<device %s, dextents<int32_t,2>, tensor_inline> tC(C, dextents<int32_t,2>(N, M));\n"
  " constexpr auto desc = matmul2d_descriptor(%d, %d, static_cast<int>(dynamic_extent));\n"
  " matmul2d<desc, execution_simdgroups<%d>> op;\n"
  " auto mA = tA.slice(0, mt*%d); auto mC = tC.slice(nt*%d, mt*%d);\n"
  " if (%d==0) { tensor<device %s, dextents<int32_t,2>, tensor_inline> tB(B, dextents<int32_t,2>(N, K)); auto mB = tB.slice(nt*%d, 0); op.run(mA, mB, mC); }\n"
  " else { tensor<device %s, dextents<int32_t,2>, tensor_inline> tB(B + (ulong)nt*(ulong)K*%d, dextents<int32_t,2>(%d, K)); op.run(mA, tB, mC); }\n}\n",
  TA, TB, TC, ORDER, TA, TC, TM, TN, SG, TM, TN, TM, LAYOUT, TB, TN, TB, TN, TN);
  return std::string(HDR) + b;
}
struct Cfg { const char* tag; const char* TA; const char* TB; const char* TC; int bA, bB, bC; int TM, TN, SG; };
int main() {
  NS::AutoreleasePool* pool = NS::AutoreleasePool::alloc()->init();
  MTL::Device* dev = MTL::CreateSystemDefaultDevice(); MTL::CommandQueue* q = dev->newCommandQueue();
  const int N = 17408, K = 5120;  // ffn gate/up shape
  std::vector<Cfg> cfgs = {
    {"fp16", "half", "half", "float", 2,2,4, 32, 64, 1},
    {"fp16", "half", "half", "float", 2,2,4, 64, 64, 4},
    {"fp16", "half", "half", "float", 2,2,4, 64, 128, 4},
    {"fp16", "half", "half", "float", 2,2,4, 32, 128, 4},
    {"int8", "int8_t", "int8_t", "int", 1,1,4, 64, 128, 4},
    {"int8", "int8_t", "int8_t", "int", 1,1,4, 32, 128, 4},
  };
  for (int M : {512, 2048}) {
    printf("=== M=%d tokens, N=%d, K=%d (weights fp16 %.0f MB / int8 %.0f MB) ===\n", M, N, K, 2.0*N*K/1e6, 1.0*N*K/1e6);
    for (auto& c : cfgs) for (int order = 0; order < 2; order++) for (int layout = 0; layout < 2; layout++) {
      std::string src = kern(c.TA, c.TB, c.TC, c.TM, c.TN, c.SG, order, layout);
      MTL::CompileOptions* o = MTL::CompileOptions::alloc()->init(); o->setLanguageVersion(MTL::LanguageVersion4_1);
      NS::Error* e = nullptr; MTL::Library* lib = dev->newLibrary(NS::String::string(src.c_str(), NS::UTF8StringEncoding), o, &e);
      if (!lib) { printf("  compile fail\n"); continue; }
      MTL::Function* fn = lib->newFunction(NS::String::string("k", NS::UTF8StringEncoding));
      MTL::ComputePipelineState* pso = dev->newComputePipelineState(fn, &e); if (!pso) { printf("  pso fail\n"); continue; }
      MTL::Buffer* A = dev->newBuffer((size_t)M*K*c.bA, MTL::ResourceStorageModeShared);
      MTL::Buffer* B = dev->newBuffer((size_t)K*N*c.bB, MTL::ResourceStorageModeShared);
      MTL::Buffer* C = dev->newBuffer((size_t)M*N*c.bC, MTL::ResourceStorageModeShared);
      // random data (avoid constant-fill artefacts); small magnitudes
      uint32_t s = 12345; auto rnd = [&]() { s ^= s << 13; s ^= s >> 17; s ^= s << 5; return s; };
      if (c.bA == 2) { _Float16* p = (_Float16*)A->contents(); for (size_t i = 0; i < (size_t)M*K; i++) p[i] = (_Float16)(((int)(rnd() & 0xffff) - 32768) / 65536.0f);
                       _Float16* w = (_Float16*)B->contents(); for (size_t i = 0; i < (size_t)K*N; i++) w[i] = (_Float16)(((int)(rnd() % 3) - 1) * 0.01f); }
      else { int8_t* p = (int8_t*)A->contents(); for (size_t i = 0; i < (size_t)M*K; i++) p[i] = (int8_t)(rnd() & 0xff);
             int8_t* w = (int8_t*)B->contents(); for (size_t i = 0; i < (size_t)K*N; i++) w[i] = (int8_t)((int)(rnd() % 3) - 1); }
      uint32_t dims[4] = {(uint32_t)M, (uint32_t)N, (uint32_t)K, 0}; double best = 1e30;
      for (int it = 0; it < 6; it++) {
        MTL::CommandBuffer* cb = q->commandBuffer(); MTL::ComputeCommandEncoder* enc = cb->computeCommandEncoder();
        enc->setComputePipelineState(pso); enc->setBuffer(A,0,0); enc->setBuffer(B,0,1); enc->setBuffer(C,0,2); enc->setBytes(dims, sizeof dims, 3);
        MTL::Size grid = order == 0 ? MTL::Size(N / c.TN, M / c.TM, 1) : MTL::Size(M / c.TM, N / c.TN, 1);
        enc->dispatchThreadgroups(grid, MTL::Size(32 * c.SG, 1, 1)); enc->endEncoding(); cb->commit(); cb->waitUntilCompleted();
        double t = cb->GPUEndTime() - cb->GPUStartTime(); if (it >= 2 && t < best) best = t;
      }
      double flops = 2.0 * M * N * K; double wbytes = (double)K * N * c.bB;
      printf("  %-4s tile %3dx%-3d SG%d  %s-fast  B-layout %-9s : %7.2f ms  %6.2f T%s   (weights once = %.1f GB/s)\n",
             c.tag, c.TM, c.TN, c.SG, order == 0 ? "N" : "M", layout == 0 ? "row-major" : "tile-major", best * 1e3, flops / best / 1e12, c.tag[0]=='i' ? "OPS" : "FLOPS", wbytes / best / 1e9);
      A->release(); B->release(); C->release(); pso->release(); fn->release(); lib->release(); o->release();
    }
  }
  pool->release(); return 0;
}
