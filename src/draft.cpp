// DFlash2 draft: weight binding, ring buffers and the draft's own copy of the shared target tensors.
#include "draft.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>

using namespace bonsai;

static std::string blkName(const char* fmt, int il) { char b[96]; snprintf(b, sizeof b, fmt, il); return b; }

bool Draft::load(Engine& eng, const std::string& path) {
    if (!gg.open(path.c_str())) return false;
    auto need = [&](const std::string& n) -> uint64_t {
        auto it = gg.tensors.find(n);
        if (it == gg.tensors.end()) { fprintf(stderr, "draft: missing tensor %s\n", n.c_str()); exit(1); }
        return it->second.fileOffset;
    };
    uint64_t bytes = 0;
    for (uint32_t il = 0; il < kLayers; il++) {
        LayerW& w = L[il];
        w.attnNorm = need(blkName("blk.%d.attn_norm.weight", il)); w.q = need(blkName("blk.%d.attn_q.weight", il));
        w.k = need(blkName("blk.%d.attn_k.weight", il)); w.v = need(blkName("blk.%d.attn_v.weight", il));
        w.o = need(blkName("blk.%d.attn_output.weight", il)); w.qNorm = need(blkName("blk.%d.attn_q_norm.weight", il));
        w.kNorm = need(blkName("blk.%d.attn_k_norm.weight", il)); w.attnCb = need(blkName("blk.%d.attn_conv_base", il));
        w.attnCp = need(blkName("blk.%d.attn_conv_proj.weight", il)); w.ffnNorm = need(blkName("blk.%d.ffn_norm.weight", il));
        w.gate = need(blkName("blk.%d.ffn_gate.weight", il)); w.up = need(blkName("blk.%d.ffn_up.weight", il));
        w.down = need(blkName("blk.%d.ffn_down.weight", il)); w.ffnCb = need(blkName("blk.%d.ffn_conv_base", il));
        w.ffnCp = need(blkName("blk.%d.ffn_conv_proj.weight", il));
    }
    fc = need("fc.weight"); encNorm = need("enc.output_norm.weight"); outNorm = need("output_norm.weight");
    for (auto& kv : gg.tensors) bytes += kv.second.nbytes;
    // the draft's weight buffer: one zero-copy wrap of its own mapping, like the trunk
    wbuf = eng.gpu.wrap(gg.map, gg.mapLen);
    // ring K/V (one ring per layer) and the block scratch
    const size_t ringBytes = (size_t)kLayers * kRing * kKvHeads * kHeadDim * 2;
    kbuf = eng.gpu.alloc(ringBytes); vbuf = eng.gpu.alloc(ringBytes);
    catH = eng.gpu.alloc((size_t)kNMax * 25600 * 2); cH = eng.gpu.alloc((size_t)kNMax * kH * 2);
    fcf = eng.gpu.alloc((size_t)kNMax * kH * 4); kvf = eng.gpu.alloc((size_t)kNMax * kKvHeads * kHeadDim * 4);
    nrmH = eng.gpu.alloc(kBlock * kH * 2);
    x = eng.gpu.alloc(kBlock * kH * 4); nrm = eng.gpu.alloc(kBlock * kH * 4);
    q = eng.gpu.alloc(kBlock * kHeads * kHeadDim * 4); k = eng.gpu.alloc(kBlock * kKvHeads * kHeadDim * 4);
    v = eng.gpu.alloc(kBlock * kKvHeads * kHeadDim * 4); aout = eng.gpu.alloc(kBlock * kHeads * kHeadDim * 4);
    tmp = eng.gpu.alloc(kBlock * kH * 4); dyn = eng.gpu.alloc(kBlock * kConvProj * 4);
    fout = eng.gpu.alloc(kBlock * kFfn * 4); probs = eng.gpu.alloc(kBlock * kH * 2);
    tapsBuf = eng.gpu.alloc((size_t)kTargets * kNMax * kH * 2); tokbuf = eng.gpu.alloc(kBlock * 4);
    ok = true;
    printf("[draft] %u layers, %llu MB of weights, ring %u positions, %zu MB of K/V\n", kLayers, (unsigned long long)(bytes >> 20), kRing,
           (ringBytes * 2) >> 20);
    return true;
}
