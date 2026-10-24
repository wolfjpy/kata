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
    std::vector<std::string> skip;   // profiling ablation: names of ops to leave out of the command stream (timing only; results are garbage)
    bool sk(const char* n) const { for (auto& s : skip) if (s == n) return true; return false; }
    LayerW layers[bonsai::kLayers];
    uint64_t tokEmbd = 0, outNorm = 0, outW = 0;

    bool load(const std::string& model, const std::string& metallib, uint32_t maxCtx, uint32_t dbgTokens);
    // Runs one token at position `pos`; returns the greedy next token. If dbgSlot >= 0 the per-layer residual and block outputs are recorded.
    int step(int token, uint32_t pos, int dbgSlot = -1, bool wantLogits = true);
    void resetState();
    static constexpr uint32_t kMaxChunk = 1024;
    // Batched prefill of M (<= kMaxChunk) tokens starting at position pos0 (exact fp16 path). Returns the greedy next token (or -1 if !wantLogits).
    int prefill(const int* toks, uint32_t M, uint32_t pos0, int dbgBase = -1, bool wantLogits = true, uint32_t nLogits = 1);
    // ---- MTP head (blk.64, Q8_0)
    bool hasMtp = false;
    bool saveHidden = false;   // step(): also produce the plain output-normalised hidden state (hidden())
    const float* hidden() const { return (const float*)bufH->contents(); }
    // Runs R (<= 2) MTP rows: row r pairs hidden state hin[r] with token toks[r] at position pos0 + r. Returns the greedy token predicted after the last row (logits in logits()).
    int mtpRows(uint32_t R, const int* toks, const float* hin, uint32_t pos0, bool wantLogits = true);
    bool int8Mode = false;   // path B: int8 activations x int8 trits, per-group scale epilogue (quality-gated)
    const float* logitsAll() const { return (const float*)bufLogitsAll->contents(); }   // logits of the last nLogits tokens of the previous prefill
    const float* logits() const { return (const float*)bufLogits->contents(); }
    const float* dbgResidual(int layer, int slot) const { return (const float*)dbgRes->contents() + ((size_t)layer * dbgTokens + slot) * bonsai::kHidden; }
    const float* dbgBlock(int layer, int slot) const { return (const float*)dbgBlk->contents() + ((size_t)layer * dbgTokens + slot) * bonsai::kHidden; }

private:
    MTL::Buffer *wbuf = nullptr;
    MTL::Buffer *bufXs[2], *bufXp, *bufH, *bufQkv, *bufConv, *bufZ, *bufAlpha, *bufBeta, *bufGdnOut, *bufTmp, *bufPM, *bufPL, *bufPA, *bufQfull, *bufK, *bufV, *bufQ, *bufGate, *bufAttnO, *bufG, *bufU, *bufLogits, *bufTok, *bufArg;
    MTL::Buffer *sg5120, *sg6144, *sg17408, *gsAll, *gdnState, *convState, *kcache, *vcache, *dbgRes, *dbgBlk;
    uint32_t nGdn = 0, nAttn = 0;
    struct MtpW { uint64_t ehProj, attnNorm, postNorm, wq, wk, wv, wo, qNorm, kNorm, ffnGate, ffnUp, ffnDown, enorm, hnorm, headNorm; } mtp{};
    MTL::Buffer *mK, *mV, *mTok, *mE, *mHin, *mCat, *mX, *mH, *mQfull, *mKp, *mVp, *mQ, *mGate, *mAttnO, *mTmp, *mY, *mG, *mU, *mPM, *mPL, *mPA;
    MTL::Buffer *pXq, *pSx, *pStage8, *pStageSW, *bufLogitsAll;
    // prefill buffers (rows padded to kMaxChunk)
    MTL::Buffer *pTok, *pX[2], *pXp, *pHh, *pQkv, *pZ, *pConv, *pAB, *pGb, *pQk, *pRaw, *wabAll, *pGdnOut, *pTmp, *pQfull, *pK, *pV, *pQ, *pGate, *pAttnO, *pG, *pU, *pStage, *pPM, *pPL, *pPA;
    void loadMtp();
    void dGemm(MTL::ComputeCommandEncoder* e, MTL::Buffer* A, MTL::Buffer* B, uint64_t bOff, MTL::Buffer* C, uint32_t Mpad, uint32_t K, uint32_t N);
    void dExpandGemm(MTL::ComputeCommandEncoder* e, uint64_t wOff, MTL::Buffer* A, MTL::Buffer* C, uint32_t Mpad, uint32_t K, uint32_t N);

    void dGemvPtq1(MTL::ComputeCommandEncoder* e, uint64_t off, MTL::Buffer* x, MTL::Buffer* y, uint32_t K, uint32_t N, const char* label = "gemv");
    void dGemvBf16(MTL::ComputeCommandEncoder* e, uint64_t off, MTL::Buffer* x, MTL::Buffer* y, uint32_t K, uint32_t N);
    void dRmsFwht(MTL::ComputeCommandEncoder* e, MTL::Buffer* src, MTL::Buffer* src2, MTL::Buffer* gs, uint64_t gsOff, MTL::Buffer* out, uint32_t W, uint32_t mode);
    void dAdd(MTL::ComputeCommandEncoder* e, MTL::Buffer* x, MTL::Buffer* y, uint32_t n);
    void dCopy(MTL::ComputeCommandEncoder* e, MTL::Buffer* src, MTL::Buffer* dst, uint64_t dstOff, uint32_t n);
};
