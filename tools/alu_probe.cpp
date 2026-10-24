#include "gpu.h"
#include <cstdio>
#include <string>
int main(int argc, char** argv) {
    Gpu gpu; if (!gpu.init(argc > 1 ? argv[1] : "build/bonsai.metallib")) return 1;
    const uint32_t threads = 12 * 1024 * 16, iters = 4096; MTL::Buffer* out = gpu.alloc(threads * 16);
    struct T { const char* k; double opsPerIter; const char* what; };
    T tests[] = { {"k_alu_f32", 8, "fp32 fma"}, {"k_alu_f16", 8, "fp16 fma (scalar half)"}, {"k_alu_h2", 16, "half2 fma (2 lanes/instr?)"}, {"k_alu_h4", 32, "half4 fma"}, {"k_alu_floor", 16, "fp32 mul-add + floor (2 ops)"}, {"k_alu_fract", 16, "fp32 mul-add + fract (2 ops)"}, {"k_alu_int", 16, "int32 mul-add"}, {"k_alu_i16", 16, "int16 mul-add"} };
    for (auto& t : tests) {
        double best = 1e9; for (int rep = 0; rep < 5; rep++) {
            MTL::CommandBuffer* cb = gpu.queue->commandBuffer(); MTL::ComputeCommandEncoder* e = cb->computeCommandEncoder(); e->setComputePipelineState(gpu.pipe(t.k)); e->setBuffer(out, 0, 0); uint32_t it = iters; e->setBytes(&it, 4, 1);
            e->dispatchThreads(MTL::Size(threads, 1, 1), MTL::Size(256, 1, 1)); e->endEncoding(); cb->commit(); cb->waitUntilCompleted(); best = std::min(best, cb->GPUEndTime() - cb->GPUStartTime()); }
        const double ops = (double)threads * iters * t.opsPerIter; printf("%-12s %-34s %8.1f G instr-lanes/s\n", t.k, t.what, ops / best / 1e9);
    }
}
