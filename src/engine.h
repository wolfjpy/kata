// Reference-quality decode engine (milestone M1/M2): correctness first, one token per step.
#pragma once
#include "gguf.h"
#include "gpu.h"
#include "kata_m6.h"
#include <string>
#include <vector>
#include <cstring>
#include <cstdlib>

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
    LayerW layers[kata::kLayers];
    uint64_t tokEmbd = 0, outNorm = 0, outW = 0;

    bool load(const std::string& model, const std::string& metallib, uint32_t maxCtx, uint32_t dbgTokens);
    // Runs one token at position `pos`; returns the greedy next token. If dbgSlot >= 0 the per-layer residual and block outputs are recorded.
    int step(int token, uint32_t pos, int dbgSlot = -1, bool wantLogits = true);
    void resetState();
    void warmup();   // dummy forward of every kind; safe to run on a background thread before the first real call
    static constexpr uint32_t kMaxChunk = 1024;
    // Batched prefill of M (<= kMaxChunk) tokens starting at position pos0 (exact fp16 path). Returns the greedy next token (or -1 if !wantLogits).
    int prefill(const int* toks, uint32_t M, uint32_t pos0, int dbgBase = -1, bool wantLogits = true, uint32_t nLogits = 1);
    // ---- MTP head (blk.64, Q8_0)
    bool hasMtp = false;
    bool saveHidden = false;   // step(): also produce the plain output-normalised hidden state (hidden())
    // the plain output-normalised hidden of the last fed position (bufH stores fp16), converted on read
    const float* hidden() const { const _Float16* h = (const _Float16*)bufH->contents(); float* o = (float*)hConv->contents(); for (int i = 0; i < 5120; i++) o[i] = (float)h[i]; return o; }
    // Runs R (<= 2) MTP rows: row r pairs hidden state hin[r] with token toks[r] at position pos0 + r. Returns the greedy token predicted after the last row (logits in logits()).
    int mtpRows(uint32_t R, const int* toks, const float* hin, uint32_t pos0, bool wantLogits = true);
    // the chained draft's input for the next step: output_norm of the MTP block's own residual (same convention as the trunk's h)
    void mtpHidden(float* dst) const { const _Float16* h = (const _Float16*)mHz->contents(); for (int i = 0; i < 5120; i++) dst[i] = (float)h[i]; }
    // Feeds n tokens at positions [pos0, pos0+n) (stepwise for short runs, batched prefill otherwise), keeps the MTP K/V cache current when mtpTrack is set, returns the greedy next token.
    bool mtpTrack = false;
    int feed(const int* toks, uint32_t n, uint32_t pos0, bool wantLogits = true);
    const float* hPrev() const { return (const float*)hPrevBuf->contents(); }
    const _Float16* hPrevH() const { return (const _Float16*)hPrevF16->contents(); }
    void setHPrevH(const _Float16* h) { memcpy(hPrevF16->contents(), h, 5120 * 2); }                       // hidden state of the last fed position
    void setHPrev(const float* h) { memcpy(hPrevBuf->contents(), h, kHiddenBytes); }
    // Speculative verification: runs 2 tokens (positions pos, pos+1) through the main model in one pass. Both GDN states (after token 0 and after token 1) are kept; commit2() picks one.
    void verify2(const int* toks, uint32_t pos, int* argmaxOut);
    void commit2(bool acceptBoth);
    const float* verifyLogits(int row) const { return (const float*)bufLogitsAll->contents() + (size_t)row * kata::kVocab; }
    void verifyHidden(int row, float* dst) const;   // post-final-norm hidden state of verified token `row`
    // verification GEMV path: 0 = wide matrix-unit kernel (T <= 8), 2 or 4 = ALU LUT+half2 kernel for exactly that many token rows
    uint32_t aluT = 0;
    uint32_t gemmGroup = getenv("KATA_GM") ? (uint32_t)atoi(getenv("KATA_GM")) : 32u;   // GEMM tile order: groups of g consecutive M tiles share a B strip (0 = N-fast, much slower)
    // ---- raw access to the persistent state for the prefix cache (src/ckpt.cpp)
    uint8_t* gdnPtr() const { return (uint8_t*)gdnState->contents(); }
    size_t gdnBytes() const { return gdnState->length(); }
    uint8_t* convPtr() const { return (uint8_t*)convState->contents(); }
    size_t convBytes() const { return convState->length(); }
    // KV planes of the main attention layers: fp16 -> {K, V}; int8 -> {K, V, K scales, V scales} (one fp16 scale per position and KV head)
    bool kvInt8 = false;     // --kv int8 / KATA_KV=int8 (lossy; set before load)
    uint32_t kvPlanes() const { return kvInt8 ? 4u : 2u; }
    size_t kvPlaneRow(uint32_t plane) const { return plane < 2 ? (size_t)kata::kKvDim * (kvInt8 ? 1 : 2) : (size_t)4 * 2; }
    uint8_t* kvPlanePtr(uint32_t attnLayer, uint32_t plane) const {
        MTL::Buffer* b = plane == 0 ? kcache : plane == 1 ? vcache : plane == 2 ? kscale : vscale; return (uint8_t*)b->contents() + (size_t)attnLayer * maxCtx * kvPlaneRow(plane); }
    // (the single MTP layer keeps an fp16 cache in every mode)
    uint8_t* mtpKvPtr(bool value) const { return (uint8_t*)(value ? mV : mK)->contents(); }
    uint8_t* hPrevPtr() const { return (uint8_t*)hPrevBuf->contents(); }
    float* logitsMut() const { return (float*)bufLogits->contents(); }
    uint32_t numAttn() const { return nAttn; }
    uint64_t modelFingerprint() const;      // model file size + hash of its first 64 KiB
    std::string numericsTag() const;        // everything that changes the numbers a prefill produces (int8 path, exclusions, ...)
    // int8 prefill (lossy): which matrices use the int8 x int8 path (the others stay exact fp16). Bit per name: qkv z ssmout wq wk wv wo gate up down.
    uint32_t i8Mask = 0;
    static uint32_t i8Bit(const char* name);
    bool i8(const char* name) const { return int8Mode && (i8Mask & i8Bit(name)) != 0; }
    // "all" (every matrix), "ffn" (FFN gate+up only: ~47% of the FLOPs, the error-optimal subset), or a comma list; "" or "none" switches int8 off.
    bool setI8(const std::string& spec);
    std::string i8Names() const;
    // ---- wide verification (M26): T (up to kMaxVerify) tokens through the trunk in one pass, keeping a GDN snapshot per accepted-prefix length
    static constexpr uint32_t kMaxVerify = 8;
    int verifyWide(const int* toks, uint32_t T, uint32_t pos, int* argmaxOut, MTL::Buffer* taps = nullptr, const int32_t* tapLayers = nullptr, uint32_t nTaps = 0);   // returns the greedy argmax of row T-1 (all T rows' logits stay in verifyLogits(row))
    void commitWide(uint32_t accept);                                            // keep the snapshot after `accept` tokens and make the MTP cache current
    const _Float16* verifyHiddenRow(uint32_t row) const { return (const _Float16*)pHh->contents() + (size_t)row * kata::kHidden; }
    // ---- thin wrappers the DFlash2 draft needs from the trunk (its embedding, its shared lm_head, its folded output norm)
    MTL::Buffer* weightsBuffer() const { return wbuf; }
    uint64_t offTokEmbd() const { return tokEmbd; }
    uint64_t offOutW() const { return outW; }
    uint64_t offOutNormGs() const { return (uint64_t)(kata::kLayers * 2) * kata::kHidden * 4; }
    MTL::Buffer* gsBuffer() const { return gsAll; }
    MTL::Buffer* bufLogitsAllPublic() const { return bufLogitsAll; }
    MTL::Buffer* sign5120() const { return sg5120; }
    void xWideGemv(MTL::ComputeCommandEncoder* e, uint64_t off, MTL::Buffer* xh, MTL::Buffer* y, uint32_t K, uint32_t N, uint32_t T) { dWide(e, off, xh, y, K, N, T); }
    MTL::Buffer* embedBuffer() { return bufXp; }        // fp16 staging the draft also uses for its own embeddings
    void xEmbed(MTL::ComputeCommandEncoder* e, MTL::Buffer* toks, uint32_t T, MTL::Buffer* outFloat);
    void xEmbedBlock(const int* toks, uint32_t T);   // diagnostic hook: embeds T explicit tokens into bufXp
    void xFinalNorm(MTL::ComputeCommandEncoder* e, MTL::Buffer* x, uint32_t T, MTL::Buffer* outXp);   // outNorm + sign + FWHT into fp16 rows
    bool int8Mode = false;   // path B: int8 activations x int8 trits, per-group scale epilogue (quality-gated)
    const float* logitsAll() const { return (const float*)bufLogitsAll->contents(); }   // logits of the last nLogits tokens of the previous prefill
    const float* logits() const { return (const float*)bufLogits->contents(); }
    const float* dbgResidual(int layer, int slot) const { return (const float*)dbgRes->contents() + ((size_t)layer * dbgTokens + slot) * kata::kHidden; }
    const float* dbgBlock(int layer, int slot) const { return (const float*)dbgBlk->contents() + ((size_t)layer * dbgTokens + slot) * kata::kHidden; }

private:
    MTL::Buffer *wbuf = nullptr;
    MTL::Buffer *bufXs[2], *bufXp, *bufH, *bufQkv, *bufConv, *bufZ, *bufAlpha, *bufBeta, *bufGdnOut, *bufTmp, *bufPM, *bufPL, *bufPA, *bufQfull, *bufK, *bufV, *bufQ, *bufGate, *bufAttnO, *bufG, *bufU, *bufLogits, *bufTok, *bufArg;
    MTL::Buffer *sg5120, *sg6144, *sg17408, *gsAll, *gdnState, *convState, *kcache, *vcache, *dbgRes, *dbgBlk;
    uint32_t nGdn = 0, nAttn = 0;
    uint32_t mtpLastR = 1;
    struct MtpW { uint64_t ehProj, attnNorm, postNorm, wq, wk, wv, wo, qNorm, kNorm, ffnGate, ffnUp, ffnDown, enorm, hnorm, headNorm; } mtp{};
    MTL::Buffer *mHz = nullptr;
    MTL::Buffer *mK, *mV, *mTok, *mE, *mHin, *mCat, *mX, *mH, *mQfull, *mKp, *mVp, *mQ, *mGate, *mAttnO, *mTmp, *mY, *mG, *mU, *mPM, *mPL, *mPA;
    MTL::Buffer *pXq, *pSx, *pStage8, *pStageSW, *bufLogitsAll;
    // prefill buffers (rows padded to kMaxChunk)
    MTL::Buffer *hPrevF16 = nullptr;
    MTL::Buffer *pTok, *pX[2], *pXp, *pHh, *pQkv, *pZ, *pConv, *pAB, *pGb, *pQk, *pRaw, *wabAll, *pGdnOut, *pTmp, *pQfull, *pK, *pV, *pQ, *pQh, *pGate, *pAttnO, *pG, *pU, *pStage, *pPM, *pPL, *pPA;
    static constexpr size_t kHiddenBytes = 5120 * 4;
    MTL::Buffer *hConv = nullptr;
    MTL::Buffer *kmean = nullptr;   // int8 KV: per (position, KV head) mean for the asymmetric key quantisation
    MTL::Buffer *gdnSet[3] = {}, *convSet[3] = {}, *pHid = nullptr, *hPrevBuf = nullptr; int stCur = 0;
    MTL::Buffer *vTok = nullptr, *vQkv = nullptr, *vRec = nullptr; int vBase = 0;   // snapshots of the verify window (snap[0] = the input state) and the window's tokens
    // per-layer replay records of a verify window: conv (T x 10240), qk (T x 4096) and gates (T x 96)
    static constexpr uint32_t kRecRow = 10240 + 4096 + 96;
    void dWide(MTL::ComputeCommandEncoder* e, uint64_t wOff, MTL::Buffer* xh, MTL::Buffer* y, uint32_t K, uint32_t N, uint32_t MT);

    void encMtpKv(MTL::ComputeCommandEncoder* e, uint32_t rowStart, uint32_t M, uint32_t pos0);
    MTL::Buffer *ilvBuf = nullptr;
    MTL::Buffer *decQp = nullptr, *decAcc = nullptr, *decM = nullptr, *decL = nullptr;   // flash-decoding scratch: packed Q, per-split partials
    void dDecAttn(MTL::ComputeCommandEncoder* e, MTL::Buffer* q, MTL::Buffer* kc, uint64_t kvOff, MTL::Buffer* vc, MTL::Buffer* gate, MTL::Buffer* out, uint32_t posBase, uint32_t T, uint32_t firstPos, MTL::Buffer* ksc = nullptr, MTL::Buffer* vsc = nullptr, uint64_t scOff = 0, MTL::Buffer* kmn = nullptr);
    MTL::Buffer *kscale = nullptr, *vscale = nullptr, *kvScrK = nullptr, *kvScrV = nullptr;   // int8 KV: scales and the fp16 scratch rows the prep kernels write before quantisation
    uint64_t kvLayerOff(uint32_t ai) const { return (uint64_t)ai * maxCtx * kata::kKvDim * (kvInt8 ? 1u : 2u); }
    uint64_t scLayerOff(uint32_t ai) const { return (uint64_t)ai * maxCtx * 4 * 2; }
    void dKvQuant(MTL::ComputeCommandEncoder* e, uint32_t T, uint32_t pos0, uint32_t ai);
    void loadMtp();
    void dGemm(MTL::ComputeCommandEncoder* e, MTL::Buffer* A, MTL::Buffer* B, uint64_t bOff, MTL::Buffer* C, uint32_t Mpad, uint32_t K, uint32_t N);
    void dExpandGemm(MTL::ComputeCommandEncoder* e, uint64_t wOff, MTL::Buffer* A, MTL::Buffer* C, uint32_t Mpad, uint32_t K, uint32_t N, const char* name);

    void dGemvPtq1(MTL::ComputeCommandEncoder* e, uint64_t off, MTL::Buffer* x, MTL::Buffer* y, uint32_t K, uint32_t N, const char* label = "gemv");
    void dGemvBf16(MTL::ComputeCommandEncoder* e, uint64_t off, MTL::Buffer* x, MTL::Buffer* y, uint32_t K, uint32_t N);
    void dRmsFwht(MTL::ComputeCommandEncoder* e, MTL::Buffer* src, MTL::Buffer* src2, MTL::Buffer* gs, uint64_t gsOff, MTL::Buffer* out, uint32_t W, uint32_t mode);
    void dAdd(MTL::ComputeCommandEncoder* e, MTL::Buffer* x, MTL::Buffer* y, uint32_t n);
    void dCopy(MTL::ComputeCommandEncoder* e, MTL::Buffer* src, MTL::Buffer* dst, uint64_t dstOff, uint32_t n);
};
