// Reference-quality decode engine (milestone M1/M2): correctness first, one token per step.
#pragma once
#include "gguf.h"
#include "gpu.h"
#include "bonsai_m6.h"
#include <string>
#include <vector>

struct LayerW {
    bool isAttn = false;
    uint64_t attnNorm = 0, postNorm = 0;
    // gated delta net
    uint64_t qkv = 0, gate = 0, alpha = 0, beta = 0, conv = 0, ssmA = 0, dt = 0, ssmNorm = 0, ssmOut = 0;
    // full attention
    uint64_t wq = 0, wk = 0, wv = 0, qNorm = 0, kNorm = 0, wo = 0;
    // ffn
    uint64_t ffnGate = 0, ffnUp = 0, ffnDown = 0;
    int stateIdx = -1;   // index among GDN layers (state) or among attention layers (KV cache)
};

class Engine {
public:
    Gguf gg;
    Gpu gpu;
    uint32_t maxCtx = 0, dbgTokens = 0;
    LayerW layers[bonsai::kLayers];
    uint64_t tokEmbd = 0, outNorm = 0, outW = 0;

    bool load(const std::string& model, const std::string& metallib, uint32_t maxCtx, uint32_t dbgTokens);
    // Runs one token at position `pos`; returns the greedy next token. If dbgSlot >= 0 the per-layer residual and block outputs are recorded.
    int step(int token, uint32_t pos, int dbgSlot = -1);
    void resetState();
    const float* logits() const { return (const float*)bufLogits->contents(); }
    const float* dbgResidual(int layer, int slot) const { return (const float*)dbgRes->contents() + ((size_t)layer * dbgTokens + slot) * bonsai::kHidden; }
    const float* dbgBlock(int layer, int slot) const { return (const float*)dbgBlk->contents() + ((size_t)layer * dbgTokens + slot) * bonsai::kHidden; }

private:
    MTL::Buffer *wbuf = nullptr;
    MTL::Buffer *bufX, *bufXp, *bufH, *bufQkv, *bufConv, *bufZ, *bufAlpha, *bufBeta, *bufGdnOut, *bufTmp, *bufQfull, *bufK, *bufV, *bufQ, *bufGate, *bufAttnO, *bufG, *bufU, *bufLogits, *bufTok, *bufArg;
    MTL::Buffer *sg5120, *sg6144, *sg17408, *gsAll, *gdnState, *convState, *kcache, *vcache, *dbgRes, *dbgBlk;
    uint32_t nGdn = 0, nAttn = 0;

    void dGemvPtq1(MTL::ComputeCommandEncoder* e, uint64_t off, MTL::Buffer* x, MTL::Buffer* y, uint32_t K, uint32_t N);
    void dGemvBf16(MTL::ComputeCommandEncoder* e, uint64_t off, MTL::Buffer* x, MTL::Buffer* y, uint32_t K, uint32_t N);
    void dRmsFwht(MTL::ComputeCommandEncoder* e, MTL::Buffer* src, MTL::Buffer* src2, MTL::Buffer* gs, uint64_t gsOff, MTL::Buffer* out, uint32_t W, uint32_t mode);
    void dAdd(MTL::ComputeCommandEncoder* e, MTL::Buffer* x, MTL::Buffer* y, uint32_t n);
    void dCopy(MTL::ComputeCommandEncoder* e, MTL::Buffer* src, MTL::Buffer* dst, uint64_t dstOff, uint32_t n);
};
