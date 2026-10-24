// Correctness + speed probe of the matrix-unit prefill attention (k_flash_prefill) against a CPU reference.
#include "gpu.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>
int main(int argc, char** argv) {
    Gpu gpu; if (!gpu.init("build/bonsai.metallib")) return 1;
    const uint32_t maxCtx = 8448; const char* kname = argc > 1 ? argv[1] : "k_flash_prefill";
    MTL::Buffer* kc = gpu.alloc((size_t)maxCtx * 1024 * 2); MTL::Buffer* vc = gpu.alloc((size_t)maxCtx * 1024 * 2);
    MTL::Buffer* q = gpu.alloc((size_t)1024 * 6144 * 2); MTL::Buffer* gate = gpu.alloc((size_t)1024 * 6144 * 4); MTL::Buffer* out = gpu.alloc((size_t)1024 * 6144 * 4);
    std::mt19937 rng(5); std::normal_distribution<float> nd(0.0f, 1.0f);
    _Float16* K = (_Float16*)kc->contents(); _Float16* V = (_Float16*)vc->contents(); _Float16* Q = (_Float16*)q->contents(); float* G = (float*)gate->contents();
    for (size_t i = 0; i < (size_t)maxCtx * 1024; i++) { K[i] = (_Float16)(nd(rng)); V[i] = (_Float16)(nd(rng)); }
    for (size_t i = 0; i < (size_t)1024 * 6144; i++) { Q[i] = (_Float16)(nd(rng) * 0.0625f); G[i] = nd(rng); }
    auto run = [&](uint32_t posBase, uint32_t M, int reps) {
        MTL::CommandBuffer* cb = gpu.queue->commandBuffer(); MTL::ComputeCommandEncoder* e = cb->computeCommandEncoder();
        for (int r = 0; r < reps; r++) { e->setComputePipelineState(gpu.pipe(kname)); e->setBuffer(q, 0, 0); e->setBuffer(kc, 0, 1); e->setBuffer(vc, 0, 2); e->setBuffer(gate, 0, 3); e->setBuffer(out, 0, 4); e->setBytes(&posBase, 4, 5); e->setBytes(&M, 4, 6);
            e->dispatchThreadgroups(MTL::Size((M + 15) / 16, 4, 1), MTL::Size(192, 1, 1)); }
        e->endEncoding(); cb->commit(); cb->waitUntilCompleted(); if (cb->status() != MTL::CommandBufferStatusCompleted) { printf("GPU error\n"); exit(1); } return cb->GPUEndTime() - cb->GPUStartTime(); };
    // ---- correctness: 70 tokens (partial last tile) at posBase 300, poison the cache beyond the valid keys
    { const uint32_t pb = 300, M = 70; for (size_t i = (size_t)(pb + M) * 1024; i < (size_t)maxCtx * 1024; i++) { K[i] = (_Float16)NAN; V[i] = (_Float16)NAN; }
      memset(out->contents(), 0, (size_t)1024 * 6144 * 4); run(pb, M, 1);
      const float* o = (const float*)out->contents(); double worst = 0, mx = 0; size_t bad = 0;
      for (uint32_t t = 0; t < M; t += 3) for (uint32_t h = 0; h < 24; h += 5) {
          const uint32_t kvh = h / 6, pos = pb + t; std::vector<double> s(pos + 1); double m = -1e30;
          for (uint32_t j = 0; j <= pos; j++) { double a = 0; for (int d = 0; d < 256; d++) a += (double)(float)Q[(size_t)t * 6144 + h * 256 + d] * (double)(float)K[(size_t)j * 1024 + kvh * 256 + d]; s[j] = a; m = std::max(m, a); }
          double z = 0; for (auto& v : s) { v = std::exp(v - m); z += v; }
          for (int d = 0; d < 256; d++) { double a = 0; for (uint32_t j = 0; j <= pos; j++) a += s[j] * (double)(float)V[(size_t)j * 1024 + kvh * 256 + d]; a /= z; a *= 1.0 / (1.0 + std::exp(-(double)G[(size_t)t * 6144 + h * 256 + d]));
              const double g = o[(size_t)t * 6144 + h * 256 + d]; if (!(std::fabs(g - a) < 1e9)) bad++; worst = std::max(worst, std::fabs(g - a)); mx = std::max(mx, std::fabs(a)); } }
      printf("correctness (%s): max |gpu - cpu| = %.3e (max |ref| %.3f), non-finite %zu\n", kname, worst, mx, bad); }
    for (size_t i = 0; i < (size_t)maxCtx * 1024; i++) { K[i] = (_Float16)(nd(rng)); V[i] = (_Float16)(nd(rng)); }
    struct C { uint32_t pb, M; } cs[] = { {0, 1024}, {1024, 1024}, {3072, 1024}, {7168, 1024} };
    for (auto& c : cs) { run(c.pb, c.M, 1); double best = 1e9; for (int i = 0; i < 4; i++) best = std::min(best, run(c.pb, c.M, 4) / 4);
        const double keys = (double)c.M * (c.pb + c.M / 2.0); const double flops = keys * 24 * 256 * 4; printf("chunk of %u tokens at position %5u: %.2f ms per layer  (%.1f TFLOPS useful; x16 layers = %.2f s)\n", c.M, c.pb, best * 1e3, flops / best / 1e12, best * 16); }
}
