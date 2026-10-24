#define NS_PRIVATE_IMPLEMENTATION
#define MTL_PRIVATE_IMPLEMENTATION
#define CA_PRIVATE_IMPLEMENTATION
#include <Foundation/Foundation.hpp>
#include <Metal/Metal.hpp>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>
#include <cmath>
static const char* SRC = R"MSL(
#include <metal_stdlib>
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace metal; using namespace mpp::tensor_ops;
// M=64, N=128 (one tile), K=512 = 4 groups.  A int8 [64][512]; B int8 tile-major [K=512][128]; SW float [4][128]; SX float [64]; output float [64][128]
// MODE 0: sum of groups, no scales.  MODE 1: per-group weight scale via broadcast load.  MODE 2: MODE 1 + per-token scale via broadcast load.  MODE 3: MODE 2 with a per-element indexed scale instead of load().
kernel void k_dbg(device int8_t* A [[buffer(0)]], device int8_t* B [[buffer(1)]], device float* SW [[buffer(2)]], device float* SX [[buffer(3)]], device float* Cf [[buffer(4)]]) {
    const int K = 512, G = 4;
    array<int, 2> strA = {1, K};
    tensor<device int8_t, dextents<int32_t, 2>, tensor_inline> a0(A, dextents<int32_t, 2>(128, 64), strA);
    tensor<device int8_t, dextents<int32_t, 2>, tensor_inline> b0(B, dextents<int32_t, 2>(128, 128));
    tensor<device float, dextents<int32_t, 2>, tensor_inline> cf(Cf, dextents<int32_t, 2>(128, 64));
    constexpr auto desc = matmul2d_descriptor(64, 128, 128, false, false, false, matmul2d_descriptor::mode::multiply);
    matmul2d<desc, execution_simdgroups<4>> op;
    auto cInt = op.get_destination_cooperative_tensor<decltype(a0), decltype(b0), int>();
    auto cAcc = op.get_destination_cooperative_tensor<decltype(a0), decltype(b0), float>();
    auto cS = op.get_destination_cooperative_tensor<decltype(a0), decltype(b0), float>();
    for (uint16_t i = 0; i < cAcc.get_capacity(); i++) cAcc[i] = 0.0f;
    for (int g = 0; g < G; g++) {
        tensor<device int8_t, dextents<int32_t, 2>, tensor_inline> ag(A + g * 128, dextents<int32_t, 2>(128, 64), strA);
        tensor<device int8_t, dextents<int32_t, 2>, tensor_inline> bg(B + ulong(g) * 128 * 128, dextents<int32_t, 2>(128, 128));
        op.run(ag, bg, cInt);
#if MODE == 0
        for (uint16_t i = 0; i < cInt.get_capacity(); i++) if (cInt.is_valid_element(i)) cAcc[i] += float(cInt[i]);
#elif MODE == 3
        for (uint16_t i = 0; i < cInt.get_capacity(); i++) if (cInt.is_valid_element(i)) { auto ix = cInt.get_multidimensional_index(i); cAcc[i] += float(cInt[i]) * SW[g * 128 + ix[0]]; }
#else
        tensor<device float, dextents<int32_t, 2>, tensor_inline> tsw(SW + g * 128, dextents<int32_t, 2>(128, 1));
        cS.load(tsw);
        for (uint16_t i = 0; i < cInt.get_capacity(); i++) if (cInt.is_valid_element(i)) cAcc[i] += float(cInt[i]) * cS[i];
#endif
    }
#if MODE >= 2
#if MODE == 3
    for (uint16_t i = 0; i < cAcc.get_capacity(); i++) if (cAcc.is_valid_element(i)) { auto ix = cAcc.get_multidimensional_index(i); cAcc[i] *= SX[ix[1]]; }
#else
    tensor<device float, dextents<int32_t, 2>, tensor_inline> tsx(SX, dextents<int32_t, 2>(1, 64));
    cS.load(tsx);
    for (uint16_t i = 0; i < cAcc.get_capacity(); i++) if (cAcc.is_valid_element(i)) cAcc[i] *= cS[i];
#endif
#endif
    cAcc.store(cf);
}
)MSL";
int main() {
    NS::AutoreleasePool* pool = NS::AutoreleasePool::alloc()->init(); MTL::Device* dev = MTL::CreateSystemDefaultDevice(); MTL::CommandQueue* q = dev->newCommandQueue();
    const int K = 512, G = 4; std::vector<int8_t> A(64 * K), B(K * 128); std::vector<float> SW(G * 128), SX(64); srand(5);
    for (auto& v : A) v = (int8_t)(rand() % 21 - 10); for (auto& v : B) v = (int8_t)(rand() % 3 - 1); for (int i = 0; i < G * 128; i++) SW[i] = 0.01f + 0.001f * (i % 7); for (int i = 0; i < 64; i++) SX[i] = 0.5f + 0.01f * i;
    for (int mode = 0; mode < 4; mode++) {
        std::string src = "#define MODE " + std::to_string(mode) + "\n" + SRC; MTL::CompileOptions* o = MTL::CompileOptions::alloc()->init(); o->setLanguageVersion(MTL::LanguageVersion4_1); NS::Error* e = nullptr;
        MTL::Library* lib = dev->newLibrary(NS::String::string(src.c_str(), NS::UTF8StringEncoding), o, &e); if (!lib) { std::string m = e->localizedDescription()->utf8String(); size_t p = m.find("error:"); printf("MODE %d compile fail: %s\n", mode, m.substr(p, 300).c_str()); continue; }
        auto* pso = dev->newComputePipelineState(lib->newFunction(NS::String::string("k_dbg", NS::UTF8StringEncoding)), &e); if (!pso) { printf("MODE %d pso fail\n", mode); continue; }
        auto* bA = dev->newBuffer(A.data(), A.size(), MTL::ResourceStorageModeShared); auto* bB = dev->newBuffer(B.data(), B.size(), MTL::ResourceStorageModeShared); auto* bSW = dev->newBuffer(SW.data(), SW.size() * 4, MTL::ResourceStorageModeShared); auto* bSX = dev->newBuffer(SX.data(), SX.size() * 4, MTL::ResourceStorageModeShared);
        auto* bF = dev->newBuffer(64 * 128 * 4, MTL::ResourceStorageModeShared); memset(bF->contents(), 0, 64 * 128 * 4);
        auto* cb = q->commandBuffer(); auto* en = cb->computeCommandEncoder(); en->setComputePipelineState(pso); en->setBuffer(bA, 0, 0); en->setBuffer(bB, 0, 1); en->setBuffer(bSW, 0, 2); en->setBuffer(bSX, 0, 3); en->setBuffer(bF, 0, 4);
        en->dispatchThreadgroups(MTL::Size(1, 1, 1), MTL::Size(128, 1, 1)); en->endEncoding(); cb->commit(); cb->waitUntilCompleted();
        float* c = (float*)bF->contents(); double maxerr = 0, maxref = 0; int bad = 0;
        for (int m = 0; m < 64; m++) for (int n = 0; n < 128; n++) { double ref = 0; for (int g = 0; g < G; g++) { long s = 0; for (int k = g * 128; k < g * 128 + 128; k++) s += (long)A[m * K + k] * B[k * 128 + n]; ref += (double)s * (mode == 0 ? 1.0 : SW[g * 128 + n]); } if (mode >= 2) ref *= SX[m];
            double err = fabs(ref - c[m * 128 + n]); maxerr = fmax(maxerr, err); maxref = fmax(maxref, fabs(ref)); if (err > 1e-3 * (1 + fabs(ref))) bad++; }
        printf("MODE %d: max|err| %.3e  max|ref| %.2f  bad %d/8192\n", mode, maxerr, maxref, bad);
    }
    pool->release(); return 0; }
