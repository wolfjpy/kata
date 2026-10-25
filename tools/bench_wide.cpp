// M14 gate: per-matrix time of wide (T = 4, 8, 16 token) verification GEMVs on real tensors, against the T=2 LUT kernel and the T=8 FMA kernel.
#include "gguf.h"
#include "gpu.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
int main(int argc, char** argv) {
    Gguf gg; if (!gg.open(argv[1])) return 1; Gpu gpu; if (!gpu.init(argc > 2 ? argv[2] : "build/bonsai.metallib")) return 1; MTL::Buffer* wbuf = gpu.wrap(gg.map, gg.mapLen);
    struct Sh { const char* t; const char* label; uint32_t K, N; } shs[] = { {"blk.%d.ffn_gate.weight", "ffn_gate/up 5120->17408", 5120, 17408}, {"blk.%d.ffn_down.weight", "ffn_down 17408->5120", 17408, 5120}, {"blk.%d.attn_qkv.weight", "gdn_qkv 5120->10240", 5120, 10240} };
    std::vector<std::string> kernels; for (int i = 3; i < argc; i++) kernels.push_back(argv[i]);
    for (auto& sh : shs) {
        std::vector<uint64_t> offs; for (int l = 0; l < 8; l++) { char nm[128]; snprintf(nm, sizeof nm, sh.t, l * 4); offs.push_back(gg.tensor(nm).fileOffset); }
        MTL::Buffer* x = gpu.alloc(16 * sh.K * 2); _Float16* xp = (_Float16*)x->contents(); srand(1); for (uint32_t i = 0; i < 16 * sh.K; i++) xp[i] = (_Float16)((float)(rand() % 2001 - 1000) / 1000.0f);
        MTL::Buffer* xi = gpu.alloc(16 * sh.K * 2); MTL::Buffer* xf = gpu.alloc(sh.K * 4); MTL::Buffer* y = gpu.alloc(16 * sh.N * 4); MTL::Buffer* yr = gpu.alloc(sh.N * 4);
        printf("== %s\n", sh.label);
        for (auto& k : kernels) {
            uint32_t nr = 16, nsg = 4, T = 16; int kind = 0;          // kind 0 = wide (x [T][K]); 1 = lut2 (interleaved half2, T=2); 2 = lutp NP (interleaved)
            if (k.find("k_wide_hyb") == 0) { T = 8; nr = 16; nsg = 4; }
            else if (k.find("k_wide") == 0) { size_t m = k.find("_m"), n = k.find("_n"), g = k.find("_g"); T = atoi(k.c_str() + m + 2); nr = atoi(k.c_str() + n + 2); nsg = atoi(k.c_str() + g + 2); }
            else if (k.find("k_gemv_lut2") == 0) { kind = 1; T = 2; nr = 4; nsg = 4; }
            else if (k.find("k_gemv_lutp") == 0) { kind = 2; const uint32_t NP = atoi(k.c_str() + 11); T = 2 * NP; nr = atoi(k.c_str() + k.find("_nr") + 3); nsg = 4; }
            if (kind) { _Float16* q = (_Float16*)xi->contents(); const uint32_t NP = kind == 1 ? 1 : T / 2; for (uint32_t i = 0; i < sh.K; i++) for (uint32_t p = 0; p < NP; p++) { q[(i * NP + p) * 2] = xp[(2 * p) * sh.K + i]; q[(i * NP + p) * 2 + 1] = xp[(2 * p + 1) * sh.K + i]; } }
            MTL::Buffer* xb = kind ? xi : x; const uint32_t rowsPerTg = nr * nsg;
            const bool hyb = k.find("k_wide_hyb") == 0; uint32_t splits[3] = {0, 0, 0};
            if (hyb) { _Float16* q = (_Float16*)xi->contents(); for (uint32_t i = 0; i < sh.K; i++) for (uint32_t p = 0; p < 8; p++) { q[(i * 4 + p) * 2] = xp[(2 * p) * sh.K + i]; q[(i * 4 + p) * 2 + 1] = xp[(2 * p + 1) * sh.K + i]; }
                const uint32_t mmaRowsPerTg = 16 * 4, aluRowsPerTg = 4 * 4; const double frac = argc > 4 ? 0.66 : 0.66;
                uint32_t sr = (uint32_t)(sh.N * frac) / mmaRowsPerTg * mmaRowsPerTg; splits[0] = sr; splits[1] = sr / mmaRowsPerTg; splits[2] = (sh.N - sr + aluRowsPerTg - 1) / aluRowsPerTg; }
            auto run = [&](int reps, uint64_t off0) { MTL::CommandBuffer* cb = gpu.queue->commandBuffer(); MTL::ComputeCommandEncoder* e = cb->computeCommandEncoder();
                for (int r = 0; r < reps; r++) for (auto off : offs) { if (off0 != ~0ull) off = off0; e->setComputePipelineState(gpu.pipe(k.c_str())); e->setBuffer(wbuf, off, 0);
                    if (hyb) { e->setBuffer(x, 0, 1); e->setBuffer(xi, 0, 2); e->setBuffer(y, 0, 3); e->setBytes(&sh.K, 4, 4); e->setBytes(&sh.N, 4, 5); e->setBytes(&splits[0], 4, 6); e->setBytes(&splits[1], 4, 7);
                        e->dispatchThreadgroups(MTL::Size(splits[1] + splits[2], 1, 1), MTL::Size(128, 1, 1)); }
                    else { e->setBuffer(xb, 0, 1); e->setBuffer(y, 0, 2); e->setBytes(&sh.K, 4, 3); e->setBytes(&sh.N, 4, 4);
                        e->dispatchThreadgroups(MTL::Size((sh.N + rowsPerTg - 1) / rowsPerTg, 1, 1), MTL::Size(32 * nsg, 1, 1)); } if (off0 != ~0ull) break; }
                e->endEncoding(); cb->commit(); cb->waitUntilCompleted(); if (cb->status() != MTL::CommandBufferStatusCompleted) { printf("  %s: GPU error\n", k.c_str()); return -1.0; } return cb->GPUEndTime() - cb->GPUStartTime(); };
            // correctness on matrix 0 for every token
            memset(y->contents(), 0, 16 * sh.N * 4); if (run(1, offs[0]) < 0) continue; double md = 0, mr = 0;
            for (uint32_t t = 0; t < std::min<uint32_t>(T, 16); t++) { float* xfp = (float*)xf->contents(); for (uint32_t i = 0; i < sh.K; i++) xfp[i] = (float)xp[t * sh.K + i];
                MTL::CommandBuffer* cb = gpu.queue->commandBuffer(); MTL::ComputeCommandEncoder* e = cb->computeCommandEncoder(); e->setComputePipelineState(gpu.pipe("k_gemv_v1_nr4_sg4")); e->setBuffer(wbuf, offs[0], 0); e->setBuffer(xf, 0, 1); e->setBuffer(yr, 0, 2); e->setBytes(&sh.K, 4, 3); e->setBytes(&sh.N, 4, 4); e->dispatchThreadgroups(MTL::Size((sh.N + 15) / 16, 1, 1), MTL::Size(128, 1, 1)); e->endEncoding(); cb->commit(); cb->waitUntilCompleted();
                const float* a = (const float*)y->contents() + (size_t)t * sh.N; const float* b = (const float*)yr->contents(); for (uint32_t i = 0; i < sh.N; i++) { md = std::max(md, (double)std::fabs(a[i] - b[i])); mr = std::max(mr, (double)std::fabs(b[i])); } }
            double best = 1e9; for (int it = 0; it < 8; it++) { double t = run(3, ~0ull); if (t < 0) { best = -1; break; } best = std::min(best, t); } if (best < 0) continue;
            const double bytes = (double)sh.N * (sh.K / 128) * 28;
            printf("  %-26s T=%-2u : %8.1f us/matrix  %6.1f GB/s   max|diff| %.2e (max|y| %.2f)\n", k.c_str(), T, best / (8 * 3) * 1e6, bytes * 24 / best / 1e9, md, mr);
        }
    }
}
