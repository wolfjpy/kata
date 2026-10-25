#include "gguf.h"
#include "gpu.h"
#include <cstdio>
#include <cstdlib>
#include <vector>
int main(int argc, char** argv) {
    Gguf gg; if (!gg.open(argv[1])) return 1; Gpu gpu; if (!gpu.init("build/bonsai.metallib")) return 1; MTL::Buffer* wbuf = gpu.wrap(gg.map, gg.mapLen);
    struct Sh { const char* t; uint32_t K, N; } shs[] = { {"blk.%d.ffn_gate.weight", 5120, 17408}, {"blk.%d.ffn_down.weight", 17408, 5120}, {"blk.%d.attn_qkv.weight", 5120, 10240} };
    MTL::Buffer* out = gpu.alloc(1 << 22);
    for (auto& sh : shs) { std::vector<uint64_t> offs; for (int l = 0; l < 8; l++) { char nm[128]; snprintf(nm, sizeof nm, sh.t, l * 4); offs.push_back(gg.tensor(nm).fileOffset); }
        const double bytes = (double)sh.N * (sh.K / 128) * 28;
        for (const char* k : {"k_read_native", "k_read_wide", "k_read_native", "k_read_wide"}) { double best = 1e9;
            for (int it = 0; it < 8; it++) { MTL::CommandBuffer* cb = gpu.queue->commandBuffer(); MTL::ComputeCommandEncoder* e = cb->computeCommandEncoder();
                for (int rep = 0; rep < 3; rep++) for (auto off : offs) { e->setComputePipelineState(gpu.pipe(k)); e->setBuffer(wbuf, off, 0); e->setBuffer(out, 0, 1); e->setBytes(&sh.K, 4, 2); e->setBytes(&sh.N, 4, 3); e->dispatchThreadgroups(MTL::Size((sh.N + 15) / 16, 1, 1), MTL::Size(128, 1, 1)); }
                e->endEncoding(); cb->commit(); cb->waitUntilCompleted(); best = std::min(best, (cb->GPUEndTime() - cb->GPUStartTime()) / (3 * offs.size())); }
            printf("%-24s %-14s %7.1f us/matrix  %6.1f GB/s\n", sh.t, k, best * 1e6, bytes / best / 1e9); } }
}
