// Microbenchmark of PTQ1_0 GEMV kernels with T fp16 activation vectors (verification pass), on real model tensors.
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
    std::string model = argv[1], lib = argv[2];
    Gguf gg; if (!gg.open(model.c_str())) return 1; Gpu gpu; if (!gpu.init(lib.c_str())) return 1;
    MTL::Buffer* wbuf = gpu.wrap(gg.map, gg.mapLen);
    struct Shape { const char* tensor; const char* label; uint32_t K, N; };
    std::vector<Shape> shapes = { {"blk.%d.ffn_gate.weight", "ffn_gate/up 5120->17408", 5120, 17408}, {"blk.%d.ffn_down.weight", "ffn_down 17408->5120", 17408, 5120}, {"blk.%d.attn_qkv.weight", "gdn_qkv 5120->10240", 5120, 10240} };
    std::vector<std::string> kernels; for (int i = 3; i < argc; i++) kernels.push_back(argv[i]);
    for (auto& sh : shapes) {
        std::vector<uint64_t> offs; for (int l = 0; l < 8; l++) { char nm[128]; snprintf(nm, sizeof nm, sh.tensor, l * 4 + (std::string(sh.tensor).find("qkv") != std::string::npos ? 0 : 0)); offs.push_back(gg.tensor(nm).fileOffset); }
        const double bytesOne = (double)sh.N * (sh.K / 128) * 28;
        MTL::Buffer* x = gpu.alloc(2 * sh.K * 2); _Float16* xp = (_Float16*)x->contents(); srand(1); for (uint32_t i = 0; i < 2 * sh.K; i++) xp[i] = (_Float16)((float)(rand() % 2001 - 1000) / 1000.0f);
        MTL::Buffer* xi = gpu.alloc(2 * sh.K * 2); { _Float16* q = (_Float16*)xi->contents(); for (uint32_t i = 0; i < sh.K; i++) { q[2 * i] = xp[i]; q[2 * i + 1] = xp[sh.K + i]; } }
        MTL::Buffer* xf = gpu.alloc(sh.K * 4); float* xfp = (float*)xf->contents();
        MTL::Buffer* y = gpu.alloc(2 * sh.N * 4); MTL::Buffer* yr = gpu.alloc(sh.N * 4);
        printf("== %s (%.1f MB per matrix)\n", sh.label, bytesOne / 1e6);
        auto run = [&](const std::string& k, int reps, uint32_t nr, uint32_t nsg, MTL::Buffer* xx, MTL::Buffer* yy) {
            MTL::ComputePipelineState* p = gpu.pipe(k.c_str()); MTL::CommandBuffer* cb = gpu.queue->commandBuffer(); MTL::ComputeCommandEncoder* e = cb->computeCommandEncoder();
            for (int r = 0; r < reps; r++) for (auto off : offs) { e->setComputePipelineState(p); e->setBuffer(wbuf, off, 0); e->setBuffer(xx, 0, 1); e->setBuffer(yy, 0, 2); e->setBytes(&sh.K, 4, 3); e->setBytes(&sh.N, 4, 4);
                e->dispatchThreadgroups(MTL::Size((sh.N + nr * nsg - 1) / (nr * nsg), 1, 1), MTL::Size(32 * nsg, 1, 1)); }
            e->endEncoding(); cb->commit(); cb->waitUntilCompleted(); return cb->GPUEndTime() - cb->GPUStartTime(); };
        for (auto& k : kernels) {
            uint32_t nr = 4, nsg = 4, T = 2; { size_t a = k.find("_nr"), b = k.find("_sg"), t = k.find("_t"); if (a != std::string::npos) { nr = atoi(k.c_str() + a + 3); nsg = atoi(k.c_str() + b + 3); } if (k.find("_t") != std::string::npos && k.find("lut") == std::string::npos) T = atoi(k.c_str() + t + 2); }
            // correctness against the fp32 single-token kernel on matrix 0, each token
            double md = 0; for (uint32_t tk = 0; tk < T; tk++) {
                for (uint32_t i = 0; i < sh.K; i++) xfp[i] = (float)xp[tk * sh.K + i];
                { MTL::CommandBuffer* cb = gpu.queue->commandBuffer(); MTL::ComputeCommandEncoder* e = cb->computeCommandEncoder(); e->setComputePipelineState(gpu.pipe("k_gemv_v1_nr4_sg4")); e->setBuffer(wbuf, offs[0], 0); e->setBuffer(xf, 0, 1); e->setBuffer(yr, 0, 2); e->setBytes(&sh.K, 4, 3); e->setBytes(&sh.N, 4, 4); e->dispatchThreadgroups(MTL::Size((sh.N + 15) / 16, 1, 1), MTL::Size(128, 1, 1)); e->endEncoding(); cb->commit(); cb->waitUntilCompleted(); }
                memset(y->contents(), 0, 2 * sh.N * 4);
                { MTL::CommandBuffer* cb = gpu.queue->commandBuffer(); MTL::ComputeCommandEncoder* e = cb->computeCommandEncoder(); e->setComputePipelineState(gpu.pipe(k.c_str())); e->setBuffer(wbuf, offs[0], 0); e->setBuffer(k.find("lut") != std::string::npos ? xi : x, 0, 1); e->setBuffer(y, 0, 2); e->setBytes(&sh.K, 4, 3); e->setBytes(&sh.N, 4, 4); e->dispatchThreadgroups(MTL::Size((sh.N + nr * nsg - 1) / (nr * nsg), 1, 1), MTL::Size(32 * nsg, 1, 1)); e->endEncoding(); cb->commit(); cb->waitUntilCompleted(); }
                const float* a = (const float*)y->contents() + (size_t)tk * sh.N; const float* b = (const float*)yr->contents(); for (uint32_t i = 0; i < sh.N; i++) md = std::max(md, (double)std::fabs(a[i] - b[i]));
            }
            double best = 1e9; const int reps = 3; MTL::Buffer* xb = k.find("lut") != std::string::npos ? xi : x; for (int it = 0; it < 8; it++) best = std::min(best, run(k, reps, nr, nsg, xb, y));
            printf("  %-24s : %8.1f us/matrix  %6.1f GB/s   max|diff| %.2e\n", k.c_str(), best / (8 * reps) * 1e6, bytesOne * 8 * reps / best / 1e9, md);
        }
    }
    return 0;
}
