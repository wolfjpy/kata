// Measures the draft matrix multiply (Q4_K / Q6_K) on the real DFlash2 tensors and checks it against a CPU reference row.
#include "gguf.h"
#include "gpu.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>
extern "C" void dequantize_row_q4_K(const void*, float*, long long);
extern "C" void dequantize_row_q6_K(const void*, float*, long long);
extern "C" void dequantize_row_q2_K(const void*, float*, long long);
extern "C" void dequantize_row_q3_K(const void*, float*, long long);
extern "C" void dequantize_row_q4_K(const void*, float*, long long);
int main(int argc, char** argv) {
    Gguf gg; if (!gg.open(argv[1])) return 1; Gpu gpu; if (!gpu.init("build/bonsai.metallib")) return 1;
    MTL::Buffer* wbuf = gpu.wrap(gg.map, gg.mapLen);
    struct Sh { const char* t; const char* label; int MT; } shs[] = {
        {"blk.0.ffn_gate.weight", "ffn_gate 5120->17408 Q4_K", 8}, {"blk.0.ffn_down.weight", "ffn_down 17408->5120 Q4_K", 8},
        {"blk.2.ffn_down.weight", "ffn_down 17408->5120 Q6_K", 8}, {"blk.0.attn_q.weight", "attn_q 5120->4096 Q4_K", 8},
        {"blk.0.attn_conv_proj.weight", "conv_proj 5120->1280 Q4_K", 8}, {"fc.weight", "fc 25600->5120 Q4_K", 8}, {"blk.0.ffn_gate.weight", "ffn_gate 5120->17408 (native type)", 8}};
    for (auto& sh : shs) {
        if (gg.tensors.count(sh.t) == 0) continue;
        const GgufTensor& t = gg.tensor(sh.t); const uint32_t K = (uint32_t)t.dims[0], N = (uint32_t)t.dims[1];
        const bool q6 = t.type == GGML_Q6_K; const uint64_t rowBytes = t.nbytes / N;
        MTL::Buffer* x = gpu.alloc(16 * K * 2); _Float16* xp = (_Float16*)x->contents(); srand(7);
        for (uint32_t i = 0; i < 16 * K; i++) xp[i] = (_Float16)((float)(rand() % 2001 - 1000) / 1000.0f);
        MTL::Buffer* y = gpu.alloc(16 * N * 4);
        const int qt = q6 ? 1 : (t.type == GGML_Q2_K ? 2 : (t.type == GGML_Q3_K ? 3 : 0));
        const std::string kn = std::string("k_draft_mm_m") + std::to_string(sh.MT) + "_n16_g4_q" + std::to_string(qt);
        // correctness: row 0 against a dequantised CPU dot product for token 0
        { std::vector<float> ref(K); const uint8_t* rp = gg.bytes(t.fileOffset);
          if (t.type == GGML_Q6_K) dequantize_row_q6_K(rp, ref.data(), K);
          else if (t.type == GGML_Q2_K) dequantize_row_q2_K(rp, ref.data(), K);
          else if (t.type == GGML_Q3_K) dequantize_row_q3_K(rp, ref.data(), K);
          else dequantize_row_q4_K(rp, ref.data(), K);
          memset(y->contents(), 0, 16 * N * 4);
          MTL::CommandBuffer* cb = gpu.queue->commandBuffer(); MTL::ComputeCommandEncoder* e = cb->computeCommandEncoder(); e->setComputePipelineState(gpu.pipe(kn.c_str()));
          e->setBuffer(wbuf, t.fileOffset, 0); e->setBuffer(x, 0, 1); e->setBuffer(y, 0, 2); e->setBytes(&K, 4, 3); e->setBytes(&N, 4, 4);
          e->dispatchThreadgroups(MTL::Size((N + 63) / 64, 1, 1), MTL::Size(128, 1, 1)); e->endEncoding(); cb->commit(); cb->waitUntilCompleted();
          double got = ((const float*)y->contents())[0], want = 0;
          for (uint32_t k = 0; k < K; k++) want += (double)ref[k] * (double)xp[k];
          printf("  %-34s n=%u k=%u  row0: gpu %.6f ref %.6f  %s\n", sh.label, N, K, got, want, std::fabs(got - want) < 2e-2 * (1 + std::fabs(want)) ? "ok" : "MISMATCH"); }
        double best = 1e9;
        for (int it = 0; it < 8; it++) { MTL::CommandBuffer* cb = gpu.queue->commandBuffer(); MTL::ComputeCommandEncoder* e = cb->computeCommandEncoder();
            for (int rep = 0; rep < 4; rep++) { e->setComputePipelineState(gpu.pipe(kn.c_str())); e->setBuffer(wbuf, t.fileOffset, 0); e->setBuffer(x, 0, 1); e->setBuffer(y, 0, 2); e->setBytes(&K, 4, 3); e->setBytes(&N, 4, 4);
                e->dispatchThreadgroups(MTL::Size((N + 63) / 64, 1, 1), MTL::Size(128, 1, 1)); }
            e->endEncoding(); cb->commit(); cb->waitUntilCompleted(); best = std::min(best, (cb->GPUEndTime() - cb->GPUStartTime()) / 4); }
        printf("     %7.2f ms/matrix  %6.1f GB/s\n", best * 1e3, (double)t.nbytes / best / 1e9);
    }
}
