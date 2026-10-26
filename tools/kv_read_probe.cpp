#include "gpu.h"
#include <cstdio>
int main() {
    Gpu gpu; if (!gpu.init("build/kata.metallib")) return 1; const uint32_t maxCtx = 16384;
    MTL::Buffer* kc = gpu.alloc((size_t)maxCtx * 1024 * 2), *vc = gpu.alloc((size_t)maxCtx * 1024 * 2), *out = gpu.alloc(1024 * 4 * 32 * 4);
    for (uint32_t keys : {7280u, 16384u}) for (uint32_t contiguous : {0u, 1u}) for (uint32_t per : {64u, 128u, 256u}) {
        const uint32_t splits = (keys + per - 1) / per; uint32_t prm[4] = {per, maxCtx, contiguous, 0}; double best = 1e9;
        for (int it = 0; it < 6; it++) { MTL::CommandBuffer* cb = gpu.queue->commandBuffer(); MTL::ComputeCommandEncoder* e = cb->computeCommandEncoder();
            for (int rep = 0; rep < 16; rep++) { e->setComputePipelineState(gpu.pipe("k_kv_read")); e->setBuffer(kc, 0, 0); e->setBuffer(vc, 0, 1); e->setBuffer(out, 0, 2); e->setBytes(prm, 16, 3); e->dispatchThreadgroups(MTL::Size(splits, 4, 1), MTL::Size(32, 1, 1)); }
            e->endEncoding(); cb->commit(); cb->waitUntilCompleted(); best = std::min(best, (cb->GPUEndTime() - cb->GPUStartTime()) / 16); }
        const double bytes = (double)splits * per * 4 * 512 * 2; printf("keys %5u  %-10s  %3u keys/split (%3u splits x 4 kv heads): %7.1f us per layer  %6.1f GB/s\n", keys, contiguous ? "contiguous" : "strided", per, splits, best * 1e6, bytes / best / 1e9); }
}
