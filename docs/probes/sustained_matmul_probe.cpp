#define NS_PRIVATE_IMPLEMENTATION
#define MTL_PRIVATE_IMPLEMENTATION
#define CA_PRIVATE_IMPLEMENTATION
#include <Foundation/Foundation.hpp>
#include <Metal/Metal.hpp>
#include <cstdio>
#include <cstring>
#include <string>
static const char* HDR = "#include <metal_stdlib>\n#include <metal_tensor>\n#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>\nusing namespace metal; using namespace mpp::tensor_ops;\n";
static std::string kern(const char* TA, const char* TC, int TM, int TN, int SG) { char b[2048];
  snprintf(b, sizeof b, "kernel void k(device %s* A [[buffer(0)]], device %s* B [[buffer(1)]], device %s* C [[buffer(2)]], constant uint3& dims [[buffer(3)]], uint2 tgid [[threadgroup_position_in_grid]]) {\n int M=dims.x,N=dims.y,K=dims.z; uint nt=tgid.x, mt=tgid.y;\n"
  " tensor<device %s, dextents<int32_t,2>, tensor_inline> tA(A, dextents<int32_t,2>(K, M)); tensor<device %s, dextents<int32_t,2>, tensor_inline> tC(C, dextents<int32_t,2>(N, M));\n constexpr auto desc = matmul2d_descriptor(%d, %d, static_cast<int>(dynamic_extent)); matmul2d<desc, execution_simdgroups<%d>> op;\n"
  " auto mA = tA.slice(0, mt*%d); auto mC = tC.slice(nt*%d, mt*%d); tensor<device %s, dextents<int32_t,2>, tensor_inline> tB(B + (ulong)nt*(ulong)K*%d, dextents<int32_t,2>(%d, K)); op.run(mA, tB, mC); }\n",
  TA, TA, TC, TA, TC, TM, TN, SG, TM, TN, TM, TA, TN, TN); return std::string(HDR) + b; }
int main() { setvbuf(stdout, NULL, _IONBF, 0);
  NS::AutoreleasePool* pool = NS::AutoreleasePool::alloc()->init(); MTL::Device* dev = MTL::CreateSystemDefaultDevice(); MTL::CommandQueue* q = dev->newCommandQueue();
  const int M = 2048, N = 17408, K = 5120; uint32_t dims[4] = {M, N, K, 0}; double fl = 2.0 * M * N * K;
  struct C { const char* name; const char* T; const char* TC; int es, TM, TN, SG; } cs[] = { {"int8 64x128 SG4", "int8_t", "int", 1, 64, 128, 4}, {"fp16 32x64 SG1", "half", "float", 2, 32, 64, 1} };
  for (auto& c : cs) { std::string src = kern(c.T, c.TC, c.TM, c.TN, c.SG); MTL::CompileOptions* o = MTL::CompileOptions::alloc()->init(); o->setLanguageVersion(MTL::LanguageVersion4_1); NS::Error* e = nullptr;
    MTL::Library* lib = dev->newLibrary(NS::String::string(src.c_str(), NS::UTF8StringEncoding), o, &e); if (!lib) { printf("compile fail\n"); continue; }
    auto* pso = dev->newComputePipelineState(lib->newFunction(NS::String::string("k", NS::UTF8StringEncoding)), &e);
    MTL::Buffer* A = dev->newBuffer((size_t)M*K*c.es, MTL::ResourceStorageModeShared); MTL::Buffer* B = dev->newBuffer((size_t)K*N*c.es, MTL::ResourceStorageModeShared); MTL::Buffer* Cb = dev->newBuffer((size_t)M*N*4, MTL::ResourceStorageModeShared);
    { uint32_t s = 5; auto rnd = [&]() { s ^= s << 13; s ^= s >> 17; s ^= s << 5; return s; }; if (c.es == 1) { int8_t* a = (int8_t*)A->contents(); for (size_t i = 0; i < (size_t)M*K; i++) a[i] = (int8_t)(rnd() & 0xff); int8_t* w = (int8_t*)B->contents(); for (size_t i = 0; i < (size_t)K*N; i++) w[i] = (int8_t)((int)(rnd() % 3) - 1); }
      else { _Float16* a = (_Float16*)A->contents(); for (size_t i = 0; i < (size_t)M*K; i++) a[i] = (_Float16)(((int)(rnd() & 0xffff) - 32768) / 65536.0f); _Float16* w = (_Float16*)B->contents(); for (size_t i = 0; i < (size_t)K*N; i++) w[i] = (_Float16)(((int)(rnd() % 3) - 1) * 0.01f); } }
    auto run = [&](int iters) { auto* cb = q->commandBuffer(); auto* en = cb->computeCommandEncoder(); en->setComputePipelineState(pso); en->setBuffer(A,0,0); en->setBuffer(B,0,1); en->setBuffer(Cb,0,2); en->setBytes(dims, sizeof dims, 3);
      for (int i = 0; i < iters; i++) en->dispatchThreadgroups(MTL::Size(N / c.TN, M / c.TM, 1), MTL::Size(32 * c.SG, 1, 1)); en->endEncoding(); cb->commit(); cb->waitUntilCompleted(); return cb->GPUEndTime() - cb->GPUStartTime(); };
    run(3); printf("== %s (M=%d N=%d K=%d, tile-major B, N-fast)\n", c.name, M, N, K);
    for (int it : {3, 10, 30, 100, 300}) { double best = 1e9; for (int r = 0; r < 2; r++) best = std::min(best, run(it)); printf("  %3d back-to-back: %.2f ms each => %.2f T%s\n", it, best / it * 1e3, fl * it / best / 1e12, c.es == 1 ? "OPS" : "FLOPS"); } }
  pool->release(); return 0; }
