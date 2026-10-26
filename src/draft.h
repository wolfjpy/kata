// DFlash2 draft model (5 layers, Q4_K/Q6_K) moved from the incoai/z-lab GGUFs: it proposes a block of tokens from the target's residuals, one masked forward per round.
#pragma once
#include "engine.h"
#include <cstdint>

struct Draft {
    static constexpr uint32_t kLayers = 5, kH = 5120, kFfn = 17408, kHeads = 32, kKvHeads = 8, kHeadDim = 128;
    static constexpr uint32_t kBlock = 8;          // anchor + 7 proposals
    static constexpr uint32_t kWindow = 2048;      // sliding window
    static constexpr uint32_t kRing = 4096;        // power of two, >= window + block
    static constexpr uint32_t kConvProj = 1280;    // n_groups(320) * 2 taps * 2 sides
    static constexpr uint32_t kTargets = 5;
    static constexpr int32_t kMaskTok = 248070;

    Gguf gg; bool ok = false; uint32_t frontier = 0;
    struct LayerW { uint64_t q = 0, k = 0, v = 0, o = 0, qNorm = 0, kNorm = 0, attnNorm = 0, attnCb = 0, attnCp = 0,
                    gate = 0, up = 0, down = 0, ffnNorm = 0, ffnCb = 0, ffnCp = 0; };
    LayerW L[kLayers];
    uint64_t fc = 0, encNorm = 0, outNorm = 0;
    int32_t targetLayers[kTargets] = {6, 20, 34, 48, 62};

    static constexpr uint32_t kNMax = 64;                  // positions per materialise call
    MTL::Buffer *kbuf = nullptr, *vbuf = nullptr;          // [kLayers][kRing][kKvHeads*128] fp16 ring
    MTL::Buffer *catH = nullptr, *cH = nullptr;            // fp16 matmul operands: [kNMax][25600] taps, [kNMax][kH] context vector
    MTL::Buffer *fcf = nullptr, *kvf = nullptr, *nrmf = nullptr, *nrmH = nullptr;
    MTL::Buffer *x = nullptr, *nrm = nullptr, *q = nullptr, *k = nullptr, *v = nullptr, *aout = nullptr, *tmp = nullptr,
                *dyn = nullptr, *fout = nullptr, *probs = nullptr, *tapsBuf = nullptr, *wbuf = nullptr;
    MTL::Buffer *tokbuf = nullptr;

    bool load(Engine& eng, const std::string& path);
    void reset() { frontier = 0; }
    // Materialises context K/V for positions [frontier, frontier+n) from the trunk's residual taps (kTargets x n x kH, fp16, tap-major).
    void materialise(Engine& eng, const void* tapsFp16, const int32_t* toks, uint32_t n);   // synchronous (once per commit boundary)
    // One masked forward at position pos: anchor token + masks; writes the block's logits (kBlock x vocab) to `logitsOut`.
    void forwardBlock(Engine& eng, int32_t anchor, uint32_t pos, MTL::Buffer* logitsOut);   // synchronous
};
