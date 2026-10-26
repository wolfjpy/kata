// Microbenchmark + correctness check of PTQ1_0 GEMV kernels on real model tensors.
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
    std::string model = argv[1], lib = argc > 2 ? argv[2] : "build/kata.metallib";
    Gguf gg; if (!gg.open(model.c_str())) return 1; Gpu gpu; if (!gpu.init(lib.c_str())) return 1;
    MTL::Buffer* wbuf = gpu.wrap(gg.map, gg.mapLen);
    struct Shape { const char* tensor; const char* label; uint32_t K, N; int layers; };
    std::vector<Shape> shapes = { {"blk.%d.ffn_gate.weight", "ffn_gate/up 5120->17408", 5120, 17408, 8}, {"blk.%d.ffn_down.weight", "ffn_down 17408->5120", 17408, 5120, 8}, {"blk.%d.attn_qkv.weight", "gdn_qkv 5120->10240", 5120, 10240, 8}, {"output.weight", "lm_head 5120->248320", 5120, 248320, 1} };
    std::vector<std::string> kernels; for (int i = 3; i < argc; i++) kernels.push_back(argv[i]);
    if (kernels.empty()) kernels = {"k_gemv_ptq1", "k_gemv_v1_nr1_sg4", "k_gemv_v1_nr2_sg4", "k_gemv_v1_nr4_sg4", "k_gemv_v1_nr5_sg4", "k_gemv_v1_nr4_sg2", "k_gemv_v1_nr4_sg8", "k_gemv_v1_nr5_sg8"};
    for (auto& sh : shapes) {
        std::vector<uint64_t> offs; for (int l = 0; l < sh.layers; l++) { char nm[128]; int idx = sh.layers == 1 ? 0 : (std::string(sh.tensor).find("qkv") != std::string::npos ? l * 4 : l * 4); if (strchr(sh.tensor, '%')) { snprintf(nm, sizeof nm, sh.tensor, idx == 3 ? 4 : idx); } else snprintf(nm, sizeof nm, "%s", sh.tensor); offs.push_back(gg.tensor(nm).fileOffset); }
        const double bytesOne = (double)sh.N * (sh.K / 128) * 28;
        MTL::Buffer* x = gpu.alloc(sh.K * 4); float* xp = (float*)x->contents(); srand(1); for (uint32_t i = 0; i < sh.K; i++) xp[i] = (float)(rand() % 2001 - 1000) / 1000.0f;
        MTL::Buffer* yref = gpu.alloc(sh.N * 4); MTL::Buffer* y = gpu.alloc(sh.N * 4);
        auto run = [&](const std::string& k, MTL::Buffer* out, int reps, uint32_t nr, uint32_t nsg) {
            MTL::ComputePipelineState* p = gpu.pipe(k.c_str()); MTL::CommandBuffer* cb = gpu.queue->commandBuffer(); MTL::ComputeCommandEncoder* e = cb->computeCommandEncoder();
            for (int r = 0; r < reps; r++) for (auto off : offs) {
                e->setComputePipelineState(p); e->setBuffer(wbuf, off, 0); e->setBuffer(x, 0, 1); e->setBuffer(out, 0, 2); e->setBytes(&sh.K, 4, 3); e->setBytes(&sh.N, 4, 4);
                e->dispatchThreadgroups(MTL::Size((sh.N + nr * nsg - 1) / (nr * nsg), 1, 1), MTL::Size(32 * nsg, 1, 1)); }
            e->endEncoding(); cb->commit(); cb->waitUntilCompleted(); return cb->GPUEndTime() - cb->GPUStartTime(); };
        run("k_gemv_ptq1", yref, 1, 1, 8);   // reference result for the first matrix only needs one dispatch; recompute precisely below
        { MTL::ComputePipelineState* p = gpu.pipe("k_gemv_ptq1"); MTL::CommandBuffer* cb = gpu.queue->commandBuffer(); MTL::ComputeCommandEncoder* e = cb->computeCommandEncoder(); e->setComputePipelineState(p); e->setBuffer(wbuf, offs[0], 0); e->setBuffer(x, 0, 1); e->setBuffer(yref, 0, 2); e->setBytes(&sh.K, 4, 3); e->setBytes(&sh.N, 4, 4); e->dispatchThreadgroups(MTL::Size((sh.N + 7) / 8, 1, 1), MTL::Size(256, 1, 1)); e->endEncoding(); cb->commit(); cb->waitUntilCompleted(); }
        printf("== %s  (%.1f MB per matrix, %d matrices per run)\n", sh.label, bytesOne / 1e6, sh.layers);
        for (auto& k : kernels) {
            uint32_t nr = 1, nsg = 8; { size_t a = k.find("_nr"), b = k.find("_sg"); if (a != std::string::npos) { nr = atoi(k.c_str() + a + 3); nsg = atoi(k.c_str() + b + 3); } }
            // correctness on matrix 0
            memset(y->contents(), 0, sh.N * 4);
            { MTL::ComputePipelineState* p = gpu.pipe(k.c_str()); MTL::CommandBuffer* cb = gpu.queue->commandBuffer(); MTL::ComputeCommandEncoder* e = cb->computeCommandEncoder(); e->setComputePipelineState(p); e->setBuffer(wbuf, offs[0], 0); e->setBuffer(x, 0, 1); e->setBuffer(y, 0, 2); e->setBytes(&sh.K, 4, 3); e->setBytes(&sh.N, 4, 4); e->dispatchThreadgroups(MTL::Size((sh.N + nr * nsg - 1) / (nr * nsg), 1, 1), MTL::Size(32 * nsg, 1, 1)); e->endEncoding(); cb->commit(); cb->waitUntilCompleted(); }
            const float* a = (const float*)y->contents(); const float* b = (const float*)yref->contents(); double md = 0, mr = 0; for (uint32_t i = 0; i < sh.N; i++) { md = std::max(md, (double)std::fabs(a[i] - b[i])); mr = std::max(mr, (double)std::fabs(b[i])); }
            double best = 1e9; int reps = sh.layers == 1 ? 6 : 3; for (int it = 0; it < 8; it++) { double t = run(k, y, reps, nr, nsg); best = std::min(best, t); }
            const double bytes = bytesOne * sh.layers * reps; printf("  %-22s : %8.1f us/matrix  %6.1f GB/s   max|diff| %.2e (max|y| %.2f)\n", k.c_str(), best / (sh.layers * reps) * 1e6, bytes / best / 1e9, md, mr);
        }
    }
    return 0;
}
