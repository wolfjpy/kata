// Path B probe: int8 x int8 -> int32 matmul2d per K=128 group, with the per-group weight scale (and per-token activation scale) applied in a cooperative-tensor epilogue.
#define NS_PRIVATE_IMPLEMENTATION
#define MTL_PRIVATE_IMPLEMENTATION
#define CA_PRIVATE_IMPLEMENTATION
#include <Foundation/Foundation.hpp>
#include <Metal/Metal.hpp>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <string>
#include <vector>
static const char* SRC = R"MSL(
#include <metal_stdlib>
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace metal; using namespace mpp::tensor_ops;
#ifndef TM
#define TM 64
#endif
#ifndef TN
#define TN 128
#endif
#ifndef SGN
#define SGN 4
#endif
// A: int8 [M][K]; B: int8 tile-major [N/TN][K][TN]; SW: float [N/TN][K/128][TN]; SX: float [M]; C: float [M][N]
kernel void k_gemm_i8(device int8_t* A [[buffer(0)]], device int8_t* B [[buffer(1)]], device float* SW [[buffer(2)]], device float* SX [[buffer(3)]], device float* C [[buffer(4)]],
                      constant uint3& dims [[buffer(5)]], uint2 tgid [[threadgroup_position_in_grid]]) {
    const int M = int(dims.x), N = int(dims.y), K = int(dims.z); const int nt = int(tgid.x), mt = int(tgid.y); const int G = K / 128;
    tensor<device int8_t, dextents<int32_t, 2>, tensor_inline> tA(A, dextents<int32_t, 2>(K, M));
    tensor<device float, dextents<int32_t, 2>, tensor_inline> tC(C, dextents<int32_t, 2>(N, M));
    constexpr auto desc = matmul2d_descriptor(TM, TN, 128, false, false, false, matmul2d_descriptor::mode::multiply);
    matmul2d<desc, execution_simdgroups<SGN>> op;
    array<int, 2> strA = {1, K};
    tensor<device int8_t, dextents<int32_t, 2>, tensor_inline> a0(A + ulong(mt) * TM * ulong(K), dextents<int32_t, 2>(128, TM), strA);
    tensor<device int8_t, dextents<int32_t, 2>, tensor_inline> b0(B + ulong(nt) * ulong(K) * TN, dextents<int32_t, 2>(TN, 128));
    auto cInt = op.get_destination_cooperative_tensor<decltype(a0), decltype(b0), int>();
    auto cAcc = op.get_destination_cooperative_tensor<decltype(a0), decltype(b0), float>();
    const uint16_t cap = cAcc.get_capacity(); uint16_t col[128], row[128];
    for (uint16_t i = 0; i < cap; i++) { cAcc[i] = 0.0f; auto ix = cAcc.get_multidimensional_index(i); col[i] = uint16_t(ix[0]); row[i] = uint16_t(ix[1]); }
    for (int g = 0; g < G; g++) {
        tensor<device int8_t, dextents<int32_t, 2>, tensor_inline> ag(A + ulong(mt) * TM * ulong(K) + g * 128, dextents<int32_t, 2>(128, TM), strA);
        tensor<device int8_t, dextents<int32_t, 2>, tensor_inline> bg(B + ulong(nt) * ulong(K) * TN + ulong(g) * 128 * TN, dextents<int32_t, 2>(TN, 128));
        op.run(ag, bg, cInt);
        device const float* swg = SW + (ulong(nt) * G + g) * TN;
        for (uint16_t i = 0; i < cap; i++) if (cInt.is_valid_element(i)) cAcc[i] += float(cInt[i]) * swg[col[i]];
    }
    for (uint16_t i = 0; i < cap; i++) if (cAcc.is_valid_element(i)) cAcc[i] *= SX[mt * TM + row[i]];
    auto mC = tC.slice(nt * TN, mt * TM);
    cAcc.store(mC);
}
// reference-quality fallback for the same math (plain loops), small sizes only
)MSL";
int main(int argc, char** argv) {
    setvbuf(stdout, NULL, _IONBF, 0);
    NS::AutoreleasePool* pool = NS::AutoreleasePool::alloc()->init();
    MTL::Device* dev = MTL::CreateSystemDefaultDevice(); MTL::CommandQueue* q = dev->newCommandQueue();
    struct Cfg { int TM, TN, SG; } cfgs[] = {{64, 128, 4}, {32, 64, 1}, {64, 64, 4}, {32, 128, 4}};
    for (auto& cf : cfgs) {
        std::string src = std::string("#define TM ") + std::to_string(cf.TM) + "\n#define TN " + std::to_string(cf.TN) + "\n#define SGN " + std::to_string(cf.SG) + "\n" + SRC;
        MTL::CompileOptions* o = MTL::CompileOptions::alloc()->init(); o->setLanguageVersion(MTL::LanguageVersion4_1); NS::Error* e = nullptr;
        MTL::Library* lib = dev->newLibrary(NS::String::string(src.c_str(), NS::UTF8StringEncoding), o, &e);
        if (!lib) { std::string m = e ? e->localizedDescription()->utf8String() : "?"; size_t p = m.find("error:"); printf("[tile %dx%d SG%d] COMPILE FAIL: %s\n", cf.TM, cf.TN, cf.SG, m.substr(p == std::string::npos ? 0 : p, 600).c_str()); continue; }
        auto* pso = dev->newComputePipelineState(lib->newFunction(NS::String::string("k_gemm_i8", NS::UTF8StringEncoding)), &e);
        if (!pso) { printf("[tile %dx%d SG%d] PSO FAIL\n", cf.TM, cf.TN, cf.SG); continue; }
        // ---- correctness on a small problem
        { const int M = 128, N = 256, K = 512, G = K / 128; if (M % cf.TM || N % cf.TN) continue;
          std::vector<int8_t> A((size_t)M * K), Bt((size_t)(N / cf.TN) * K * cf.TN), Bm((size_t)N * K); std::vector<float> SW((size_t)(N / cf.TN) * G * cf.TN), SWm((size_t)N * G), SX(M);
          srand(3); for (auto& v : A) v = (int8_t)(rand() % 255 - 127); for (int n = 0; n < N; n++) for (int k = 0; k < K; k++) Bm[(size_t)n * K + k] = (int8_t)(rand() % 3 - 1);
          for (int n = 0; n < N; n++) for (int g = 0; g < G; g++) SWm[(size_t)n * G + g] = 0.005f + 0.001f * ((n * 7 + g * 3) % 11); for (int m = 0; m < M; m++) SX[m] = 0.01f + 0.002f * (m % 5);
          for (int n = 0; n < N; n++) { int nt = n / cf.TN, nl = n % cf.TN; for (int k = 0; k < K; k++) Bt[((size_t)nt * K + k) * cf.TN + nl] = Bm[(size_t)n * K + k]; for (int g = 0; g < G; g++) SW[((size_t)nt * G + g) * cf.TN + nl] = SWm[(size_t)n * G + g]; }
          MTL::Buffer* bA = dev->newBuffer(A.data(), A.size(), MTL::ResourceStorageModeShared); MTL::Buffer* bB = dev->newBuffer(Bt.data(), Bt.size(), MTL::ResourceStorageModeShared); MTL::Buffer* bSW = dev->newBuffer(SW.data(), SW.size() * 4, MTL::ResourceStorageModeShared);
          MTL::Buffer* bSX = dev->newBuffer(SX.data(), SX.size() * 4, MTL::ResourceStorageModeShared); MTL::Buffer* bC = dev->newBuffer((size_t)M * N * 4, MTL::ResourceStorageModeShared); memset(bC->contents(), 0, (size_t)M * N * 4);
          uint32_t dims[4] = {(uint32_t)M, (uint32_t)N, (uint32_t)K, 0};
          auto* cb = q->commandBuffer(); auto* en = cb->computeCommandEncoder(); en->setComputePipelineState(pso); en->setBuffer(bA, 0, 0); en->setBuffer(bB, 0, 1); en->setBuffer(bSW, 0, 2); en->setBuffer(bSX, 0, 3); en->setBuffer(bC, 0, 4); en->setBytes(dims, 16, 5);
          en->dispatchThreadgroups(MTL::Size(N / cf.TN, M / cf.TM, 1), MTL::Size(32 * cf.SG, 1, 1)); en->endEncoding(); cb->commit(); cb->waitUntilCompleted();
          double maxerr = 0, maxref = 0; const float* c = (const float*)bC->contents();
          for (int m = 0; m < M; m += 7) for (int n = 0; n < N; n += 5) { double ref = 0; for (int g = 0; g < G; g++) { long s = 0; for (int k = g * 128; k < g * 128 + 128; k++) s += (long)A[(size_t)m * K + k] * Bm[(size_t)n * K + k]; ref += (double)s * SWm[(size_t)n * G + g]; } ref *= SX[m];
              maxerr = fmax(maxerr, fabs(ref - c[(size_t)m * N + n])); maxref = fmax(maxref, fabs(ref)); }
          printf("[tile %dx%d SG%d] correctness: max|err| %.3e (max|ref| %.3f) status=%d %s\n", cf.TM, cf.TN, cf.SG, maxerr, maxref, (int)cb->status(), cb->error() ? cb->error()->localizedDescription()->utf8String() : ""); }
        // ---- performance on the real prefill shape
        for (int M : {512, 2048}) { const int N = 17408, K = 5120, G = K / 128; if (M % cf.TM) continue;
          MTL::Buffer* bA = dev->newBuffer((size_t)M * K, MTL::ResourceStorageModeShared); MTL::Buffer* bB = dev->newBuffer((size_t)N * K, MTL::ResourceStorageModeShared); MTL::Buffer* bSW = dev->newBuffer((size_t)(N / cf.TN) * G * cf.TN * 4, MTL::ResourceStorageModeShared);
          MTL::Buffer* bSX = dev->newBuffer((size_t)M * 4, MTL::ResourceStorageModeShared); MTL::Buffer* bC = dev->newBuffer((size_t)M * N * 4, MTL::ResourceStorageModeShared);
          { uint32_t s = 9; auto rnd = [&]() { s ^= s << 13; s ^= s >> 17; s ^= s << 5; return s; }; int8_t* a = (int8_t*)bA->contents(); for (size_t i = 0; i < (size_t)M * K; i++) a[i] = (int8_t)(rnd() & 0xff); int8_t* b = (int8_t*)bB->contents(); for (size_t i = 0; i < (size_t)N * K; i++) b[i] = (int8_t)((int)(rnd() % 3) - 1);
            float* sw = (float*)bSW->contents(); for (size_t i = 0; i < bSW->length() / 4; i++) sw[i] = 0.01f; float* sx = (float*)bSX->contents(); for (int i = 0; i < M; i++) sx[i] = 0.02f; }
          uint32_t dims[4] = {(uint32_t)M, (uint32_t)N, (uint32_t)K, 0}; double best = 1e9;
          for (int it = 0; it < 6; it++) { auto* cb = q->commandBuffer(); auto* en = cb->computeCommandEncoder(); en->setComputePipelineState(pso); en->setBuffer(bA, 0, 0); en->setBuffer(bB, 0, 1); en->setBuffer(bSW, 0, 2); en->setBuffer(bSX, 0, 3); en->setBuffer(bC, 0, 4); en->setBytes(dims, 16, 5);
              for (int r = 0; r < 10; r++) en->dispatchThreadgroups(MTL::Size(N / cf.TN, M / cf.TM, 1), MTL::Size(32 * cf.SG, 1, 1)); en->endEncoding(); cb->commit(); cb->waitUntilCompleted(); double t = (cb->GPUEndTime() - cb->GPUStartTime()) / 10; if (it >= 1 && t < best) best = t; }
          printf("   M=%4d N=%d K=%d : %.2f ms => %.1f TOPS (with per-group scale epilogue)\n", M, N, K, best * 1e3, 2.0 * M * N * K / best / 1e12); }
    }
    pool->release(); return 0; }
