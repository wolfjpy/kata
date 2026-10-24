#include "gpu.h"
#include <cstdio>
int main() {
    Gpu gpu; if (!gpu.init("build/bonsai.metallib")) return 1;
    MTL::Buffer* a = gpu.alloc(64 * 128), *b = gpu.alloc(64 * 128), *d = gpu.alloc((64 * 128 * 2 + 4) * 2);
    MTL::CommandBuffer* cb = gpu.queue->commandBuffer(); MTL::ComputeCommandEncoder* e = cb->computeCommandEncoder(); e->setComputePipelineState(gpu.pipe("k_layout_probe")); e->setBuffer(a, 0, 0); e->setBuffer(b, 0, 1); e->setBuffer(d, 0, 2); e->dispatchThreadgroups(MTL::Size(1, 1, 1), MTL::Size(128, 1, 1)); e->endEncoding(); cb->commit(); cb->waitUntilCompleted();
    const unsigned short* p = (const unsigned short*)d->contents(); printf("capacity per thread: %u\n", p[64 * 128 * 2]);
    for (int t : {0, 1, 2, 3, 4, 31, 32, 33, 64, 96}) { printf("thread %3d (col,row): ", t); for (int i = 0; i < 32; i++) printf("(%u,%u) ", p[(t * 64 + i) * 2], p[(t * 64 + i) * 2 + 1]); printf("\n"); }
}
