// Validates the MSL Q4_K / Q6_K decoders against ggml's dequantize_row_q4_K / q6_K on real draft tensors.
#include "gguf.h"
#include "gpu.h"
#include <cstdio>
#include <cstring>
#include <cmath>
#include <string>
#include <vector>
extern "C" void dequantize_row_q4_K(const void*, float*, long long);
extern "C" void dequantize_row_q6_K(const void*, float*, long long);
int main(int argc, char** argv) {
    Gguf gg; if (!gg.open(argv[1])) return 1; Gpu gpu; if (!gpu.init("build/bonsai.metallib")) return 1;
    const char* names[] = {"blk.0.ffn_gate.weight", "blk.0.attn_q.weight", "blk.0.attn_conv_proj.weight", "fc.weight",
                           "blk.2.ffn_down.weight", "blk.2.attn_v.weight", "blk.4.ffn_down.weight", "selector_hidden.weight"};
    double worst = 0; int bad = 0, n = 0;
    for (const char* nm : names) {
        const GgufTensor& t = gg.tensor(nm);
        const uint32_t K = (uint32_t)t.dims[0];                 // elements per row (fastest-varying dim)
        const uint32_t rows = (uint32_t)(t.nbytes / (uint64_t)t.dims[0] * 0);   // unused
        const uint64_t rowBytes = t.nbytes / t.dims[1]; (void)rows;
        std::vector<float> ref(K);
        MTL::Buffer* src = gpu.alloc(rowBytes); MTL::Buffer* dst = gpu.alloc(K * 4);
        for (uint32_t r = 0; r < 3; r++) {                       // a few rows of each tensor
            const uint8_t* rowPtr = gg.bytes(t.fileOffset + r * rowBytes);
            if (t.type == GGML_Q4_K) dequantize_row_q4_K(rowPtr, ref.data(), (long long)K);
            else if (t.type == GGML_Q6_K) dequantize_row_q6_K(rowPtr, ref.data(), (long long)K);
            else continue;
            memcpy(src->contents(), rowPtr, rowBytes);
            const uint32_t nblk = K / 256; double md = 0, mr = 0;
            for (uint32_t b = 0; b < nblk; b++) {
                MTL::CommandBuffer* cb = gpu.queue->commandBuffer(); MTL::ComputeCommandEncoder* e = cb->computeCommandEncoder();
                e->setComputePipelineState(gpu.pipe(t.type == GGML_Q4_K ? "k_dq_q4k" : "k_dq_q6k")); e->setBuffer(src, 0, 0); e->setBuffer(dst, 0, 1);
                e->setBytes(&K, 4, 2); e->setBytes(&b, 4, 3); e->dispatchThreads(MTL::Size(256, 1, 1), MTL::Size(256, 1, 1));
                e->endEncoding(); cb->commit(); cb->waitUntilCompleted();
                const float* got = (const float*)dst->contents();
                for (uint32_t i = 0; i < 256; i++) { const double a = got[i], b2 = ref[b * 256 + i]; md = std::max(md, std::fabs(a - b2)); mr = std::max(mr, std::fabs(b2)); n++; }
            }
            if (md > 1e-5 * (1.0 + mr)) { printf("  MISMATCH %-28s row %u: max|diff| %.3e (max|ref| %.3f)\n", nm, r, md, mr); bad++; }
            worst = std::max(worst, md);
        }
        printf("  %-28s type %3u  K %5u  blocks %3u  checked\n", nm, t.type, K, K / 256);
    }
    printf("%s: %d elements checked, worst |diff| %.3e, %d mismatching rows\n", bad ? "FAIL" : "PASS", n, worst, bad);
    return bad ? 1 : 0;
}
