#include "gpu.h"
#include <cstdio>
int main() {
    Gpu gpu; if (!gpu.init("build/bonsai.metallib")) return 1; const uint32_t threads = 12 * 1024 * 8, iters = 2048; MTL::Buffer* out = gpu.alloc(threads * 4);
    for (const char* k : {"k_mma_h_f", "k_mma_h_h", "k_mma_f_f", "k_mma_h_f"}) { double best = 1e9;
        for (int rep = 0; rep < 6; rep++) { MTL::CommandBuffer* cb = gpu.queue->commandBuffer(); MTL::ComputeCommandEncoder* e = cb->computeCommandEncoder(); e->setComputePipelineState(gpu.pipe(k)); e->setBuffer(out, 0, 0); uint32_t it = iters; e->setBytes(&it, 4, 1); e->dispatchThreads(MTL::Size(threads, 1, 1), MTL::Size(256, 1, 1)); e->endEncoding(); cb->commit(); cb->waitUntilCompleted(); best = std::min(best, cb->GPUEndTime() - cb->GPUStartTime()); }
        const double sg = threads / 32.0; const double flops = sg * iters * 4 * 2.0 * 8 * 8 * 8; printf("%-10s %7.2f TFLOPS (simdgroup 8x8x8 MMA)\n", k, flops / best / 1e12); }
}
