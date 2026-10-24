#include "engine.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <algorithm>

using namespace bonsai;

static std::string blk(const char* fmt, int il) { char b[128]; snprintf(b, sizeof b, fmt, il); return b; }

bool Engine::load(const std::string& model, const std::string& metallib, uint32_t maxContext, uint32_t dbgTok) {
    maxCtx = maxContext; dbgTokens = dbgTok;
    if (!gg.open(model.c_str())) return false;
    if (!gpu.init(metallib.c_str())) return false;
    // ---- verify the GGUF against the compile-time config
    auto need = [&](const char* k, uint64_t v) { auto it = gg.u32.find(k); if (it == gg.u32.end() || it->second != v) { fprintf(stderr, "config mismatch: %s expected %llu got %llu\n", k, (unsigned long long)v, it == gg.u32.end() ? 0ull : (unsigned long long)it->second); exit(1); } };
    need("qwen35.block_count", kAllBlocks); need("qwen35.embedding_length", kHidden); need("qwen35.feed_forward_length", kFfn);
    need("qwen35.attention.head_count", kHeads); need("qwen35.attention.head_count_kv", kKvHeads); need("qwen35.ssm.conv_kernel", kGdnConvK);
    need("qwen35.ssm.time_step_rank", kGdnVHeads); need("qwen35.ssm.group_count", kGdnKHeads); need("qwen35.ssm.state_size", kGdnKDim); need("qwen35.full_attention_interval", kFullAttnEvery);
    need("prism.hadamard.block_size", kHadBlock);
    // ---- weights: wrap the whole mapped file as one buffer (zero copy)
    wbuf = gpu.wrap(gg.map, gg.mapLen);
    tokEmbd = gg.tensor("token_embd.weight").fileOffset; outNorm = gg.tensor("output_norm.weight").fileOffset; outW = gg.tensor("output.weight").fileOffset;
    for (uint32_t il = 0; il < kLayers; il++) {
        LayerW& L = layers[il]; L.isAttn = layerIsAttention(il);
        auto T = [&](const char* f) { return gg.tensor(blk(f, il)).fileOffset; };
        L.attnNorm = T("blk.%d.attn_norm.weight"); L.postNorm = T("blk.%d.post_attention_norm.weight");
        L.ffnGate = T("blk.%d.ffn_gate.weight"); L.ffnUp = T("blk.%d.ffn_up.weight"); L.ffnDown = T("blk.%d.ffn_down.weight");
        if (L.isAttn) { L.wq = T("blk.%d.attn_q.weight"); L.wk = T("blk.%d.attn_k.weight"); L.wv = T("blk.%d.attn_v.weight"); L.qNorm = T("blk.%d.attn_q_norm.weight"); L.kNorm = T("blk.%d.attn_k_norm.weight"); L.wo = T("blk.%d.attn_output.weight"); L.stateIdx = (int)nAttn++; }
        else { L.qkv = T("blk.%d.attn_qkv.weight"); L.gate = T("blk.%d.attn_gate.weight"); L.alpha = T("blk.%d.ssm_alpha.weight"); L.beta = T("blk.%d.ssm_beta.weight"); L.conv = T("blk.%d.ssm_conv1d.weight");
               L.ssmA = T("blk.%d.ssm_a"); L.dt = T("blk.%d.ssm_dt.bias"); L.ssmNorm = T("blk.%d.ssm_norm.weight"); L.ssmOut = T("blk.%d.ssm_out.weight"); L.stateIdx = (int)nGdn++; }
    }
    // ---- Hadamard sign vectors (shared by all layers), gamma (.) sign folding
    std::vector<int32_t> widths = gg.signedInts("prism.hadamard.sign_widths"), vals = gg.signedInts("prism.hadamard.sign_values");
    if (widths.size() != 3 || widths[0] != 5120 || widths[1] != 6144 || widths[2] != 17408) { fprintf(stderr, "unexpected hadamard sign widths\n"); exit(1); }
    auto mk = [&](size_t off, size_t n) { MTL::Buffer* b = gpu.alloc(n * 4); float* p = (float*)b->contents(); for (size_t i = 0; i < n; i++) p[i] = (float)vals[off + i]; return b; };
    sg5120 = mk(0, 5120); sg6144 = mk(5120, 6144); sg17408 = mk(5120 + 6144, 17408);
    const float* s5120 = (const float*)sg5120->contents();
    gsAll = gpu.alloc((size_t)(kLayers * 2 + 2) * kHidden * 4); float* gs = (float*)gsAll->contents();
    auto fold = [&](uint64_t off, float* dst) { const float* g = (const float*)gg.bytes(off); for (uint32_t i = 0; i < kHidden; i++) dst[i] = g[i] * s5120[i]; };
    for (uint32_t il = 0; il < kLayers; il++) { fold(layers[il].attnNorm, gs + (size_t)(il * 2) * kHidden); fold(layers[il].postNorm, gs + (size_t)(il * 2 + 1) * kHidden); }
    fold(outNorm, gs + (size_t)(kLayers * 2) * kHidden);
    // ---- activations, state, caches
    bufXs[0] = gpu.alloc(5120 * 4); bufXs[1] = gpu.alloc(5120 * 4); bufPM = gpu.alloc(24 * 32 * 4); bufPL = gpu.alloc(24 * 32 * 4); bufPA = gpu.alloc((size_t)24 * 32 * 256 * 4); bufXp = gpu.alloc(17408 * 4); bufH = gpu.alloc(5120 * 4); bufQkv = gpu.alloc(kGdnQkvDim * 4); bufConv = gpu.alloc(kGdnQkvDim * 4);
    bufZ = gpu.alloc(6144 * 4); bufAlpha = gpu.alloc(64 * 4); bufBeta = gpu.alloc(64 * 4); bufGdnOut = gpu.alloc(6144 * 4); bufTmp = gpu.alloc(5120 * 4);
    bufQfull = gpu.alloc(kQProjOut * 4); bufK = gpu.alloc(kKvDim * 4); bufV = gpu.alloc(kKvDim * 4); bufQ = gpu.alloc(kAttnOutIn * 4); bufGate = gpu.alloc(kAttnOutIn * 4); bufAttnO = gpu.alloc(kAttnOutIn * 4);
    bufG = gpu.alloc(17408 * 4); bufU = gpu.alloc(17408 * 4); bufLogits = gpu.alloc((size_t)kVocab * 4); bufTok = gpu.alloc(16); bufArg = gpu.alloc(16);
    gdnState = gpu.alloc((size_t)nGdn * kGdnVHeads * 128 * 128 * 4); convState = gpu.alloc((size_t)nGdn * 3 * kGdnQkvDim * 4);
    kcache = gpu.alloc((size_t)nAttn * maxCtx * kKvDim * 2); vcache = gpu.alloc((size_t)nAttn * maxCtx * kKvDim * 2);
    {   // alpha|beta (bf16 [48][5120] each) -> one fp16 tile-major matrix with 128 output columns (alpha 0..47, beta 48..95, zero pad) per GDN layer
        wabAll = gpu.alloc((size_t)nGdn * 2 * 5120 * 64 * 2); _Float16* dst = (_Float16*)wabAll->contents();
        for (uint32_t il = 0; il < kLayers; il++) { const LayerW& L = layers[il]; if (L.isAttn) continue;
            const uint16_t* a = (const uint16_t*)gg.bytes(L.alpha); const uint16_t* b = (const uint16_t*)gg.bytes(L.beta); _Float16* d = dst + (size_t)L.stateIdx * 2 * 5120 * 64;
            for (uint32_t r = 0; r < 96; r++) { const uint16_t* row = r < 48 ? a + (size_t)r * 5120 : b + (size_t)(r - 48) * 5120;
                for (uint32_t k = 0; k < 5120; k++) { uint32_t bits = (uint32_t)row[k] << 16; float f; memcpy(&f, &bits, 4); d[((size_t)(r / 64) * 5120 + k) * 64 + (r % 64)] = (_Float16)f; } } }
    }
    pXq = gpu.alloc((size_t)kMaxChunk * 17408); pSx = gpu.alloc((size_t)kMaxChunk * 4); pStage8 = gpu.alloc((size_t)17408 * 5120); pStageSW = gpu.alloc(4 << 20); bufLogitsAll = gpu.alloc((size_t)128 * kVocab * 4);
    {   // batched prefill buffers
        const size_t R = kMaxChunk;
        pTok = gpu.alloc(R * 4); pX[0] = gpu.alloc(R * 5120 * 4); pX[1] = gpu.alloc(R * 5120 * 4); pXp = gpu.alloc(R * 17408 * 2); pHh = gpu.alloc(R * 5120 * 2);
        pQkv = gpu.alloc(R * kGdnQkvDim * 4); pZ = gpu.alloc(R * 6144 * 4); pConv = gpu.alloc(R * kGdnQkvDim * 4); pAB = gpu.alloc(R * 128 * 4); pGb = gpu.alloc(R * 96 * 4); pQk = gpu.alloc(R * 4096 * 4); pRaw = gpu.alloc(R * 6144 * 4);
        pGdnOut = gpu.alloc(R * 6144 * 4); pTmp = gpu.alloc(R * 5120 * 4); pQfull = gpu.alloc(R * kQProjOut * 4); pK = gpu.alloc(R * kKvDim * 4); pV = gpu.alloc(R * kKvDim * 4);
        pQ = gpu.alloc(R * 6144 * 4); pGate = gpu.alloc(R * 6144 * 4); pAttnO = gpu.alloc(R * 6144 * 4); pG = gpu.alloc(R * 17408 * 4); pU = gpu.alloc(R * 17408 * 4);
        pStage = gpu.alloc((size_t)17408 * 5120 * 2); pPM = gpu.alloc(256 * 24 * 8 * 4); pPL = gpu.alloc(256 * 24 * 8 * 4); pPA = gpu.alloc((size_t)256 * 24 * 8 * 256 * 4);
    }
    if (dbgTokens) { dbgRes = gpu.alloc((size_t)kLayers * dbgTokens * kHidden * 4); dbgBlk = gpu.alloc((size_t)kLayers * dbgTokens * kHidden * 4); }
    loadMtp();
    return true;
}

void Engine::resetState() {
    memset(gdnState->contents(), 0, gdnState->length()); memset(convState->contents(), 0, convState->length());
    memset(kcache->contents(), 0, kcache->length()); memset(vcache->contents(), 0, vcache->length());
}

void Engine::dGemvPtq1(MTL::ComputeCommandEncoder* e, uint64_t off, MTL::Buffer* x, MTL::Buffer* y, uint32_t K, uint32_t N, const char* label) {
    if (sk(label)) return;
    e->setComputePipelineState(gpu.pipe("k_gemv_v1_nr4_sg4")); e->setBuffer(wbuf, off, 0); e->setBuffer(x, 0, 1); e->setBuffer(y, 0, 2); e->setBytes(&K, 4, 3); e->setBytes(&N, 4, 4);
    e->dispatchThreadgroups(MTL::Size((N + 15) / 16, 1, 1), MTL::Size(128, 1, 1));
}
void Engine::dGemvBf16(MTL::ComputeCommandEncoder* e, uint64_t off, MTL::Buffer* x, MTL::Buffer* y, uint32_t K, uint32_t N) {
    e->setComputePipelineState(gpu.pipe("k_gemv_bf16")); e->setBuffer(wbuf, off, 0); e->setBuffer(x, 0, 1); e->setBuffer(y, 0, 2); e->setBytes(&K, 4, 3); e->setBytes(&N, 4, 4);
    e->dispatchThreadgroups(MTL::Size((N + 7) / 8, 1, 1), MTL::Size(256, 1, 1));
}
void Engine::dRmsFwht(MTL::ComputeCommandEncoder* e, MTL::Buffer* src, MTL::Buffer* src2, MTL::Buffer* gs, uint64_t gsOff, MTL::Buffer* out, uint32_t W, uint32_t mode) {
    if (sk(mode == 2 ? "fwht_gdn" : mode == 3 ? "fwht_attn" : "fwht_ffn")) return;
    e->setComputePipelineState(gpu.pipe("k_rms_fwht")); e->setBuffer(src, 0, 0); e->setBuffer(src2 ? src2 : src, 0, 1); e->setBuffer(gs, gsOff, 2); e->setBuffer(out, 0, 3); e->setBytes(&W, 4, 4); e->setBytes(&mode, 4, 5);
    e->dispatchThreadgroups(MTL::Size(W / 1024, 1, 1), MTL::Size(256, 1, 1));
}
void Engine::dAdd(MTL::ComputeCommandEncoder* e, MTL::Buffer* x, MTL::Buffer* y, uint32_t n) {
    e->setComputePipelineState(gpu.pipe("k_add")); e->setBuffer(x, 0, 0); e->setBuffer(y, 0, 1); e->setBytes(&n, 4, 2); e->dispatchThreads(MTL::Size(n, 1, 1), MTL::Size(256, 1, 1));
}
void Engine::dCopy(MTL::ComputeCommandEncoder* e, MTL::Buffer* src, MTL::Buffer* dst, uint64_t dstOff, uint32_t n) {
    e->setComputePipelineState(gpu.pipe("k_copy_f32")); e->setBuffer(src, 0, 0); e->setBuffer(dst, dstOff, 1); e->setBytes(&n, 4, 2); e->dispatchThreads(MTL::Size(n, 1, 1), MTL::Size(256, 1, 1));
}

int Engine::step(int token, uint32_t pos, int dbgSlot, bool wantLogits) {
    NS::AutoreleasePool* pool = NS::AutoreleasePool::alloc()->init();
    *(int*)bufTok->contents() = token;
    MTL::CommandBuffer* cb = gpu.queue->commandBuffer(); MTL::ComputeCommandEncoder* e = cb->computeCommandEncoder();
    const uint32_t H = kHidden;
    int c = 0;   // index of the live residual buffer
    // embedding: x = sign (.) H z
    { e->setComputePipelineState(gpu.pipe("k_embed")); e->setBuffer(wbuf, tokEmbd, 0); e->setBuffer(bufTok, 0, 1); e->setBuffer(sg5120, 0, 2); e->setBuffer(bufXs[0], 0, 3); e->dispatchThreadgroups(MTL::Size(5, 1, 1), MTL::Size(256, 1, 1)); }
    // fused: xnew = x (+ y); out = H(gs (.) rmsnorm(xnew)); optional plain normalised copy h
    auto fusedNorm = [&](MTL::Buffer* x, MTL::Buffer* y, uint64_t gsOff, MTL::Buffer* xnew, uint64_t gammaOff, bool wantH) {
        if (sk("fused_norm")) return;
        uint32_t flags = (y ? 1u : 0u) | (wantH ? 2u : 0u);
        e->setComputePipelineState(gpu.pipe("k_add_rms_fwht")); e->setBuffer(x, 0, 0); e->setBuffer(y ? y : x, 0, 1); e->setBuffer(gsAll, gsOff, 2); e->setBuffer(bufXp, 0, 3);
        e->setBuffer(xnew, 0, 4); e->setBuffer(wbuf, gammaOff, 5); e->setBuffer(bufH, 0, 6); e->setBytes(&flags, 4, 7);
        e->dispatchThreadgroups(MTL::Size(5, 1, 1), MTL::Size(256, 1, 1));
    };
    fusedNorm(bufXs[c], nullptr, 0, bufXs[c ^ 1], layers[0].attnNorm, !layers[0].isAttn); c ^= 1;
    for (uint32_t il = 0; il < kLayers; il++) {
        const LayerW& L = layers[il];
        if (!L.isAttn) {
            const uint32_t si = (uint32_t)L.stateIdx;
            dGemvPtq1(e, L.qkv, bufXp, bufQkv, H, kGdnQkvDim, "gemv_qkv");
            dGemvPtq1(e, L.gate, bufXp, bufZ, H, kGdnZDim, "gemv_z");
            if (!sk("k_gdn_ab")) { e->setComputePipelineState(gpu.pipe("k_gdn_ab")); e->setBuffer(wbuf, L.alpha, 0); e->setBuffer(wbuf, L.beta, 1); e->setBuffer(bufH, 0, 2); e->setBuffer(bufAlpha, 0, 3); e->setBuffer(bufBeta, 0, 4);
              e->dispatchThreadgroups(MTL::Size(96, 1, 1), MTL::Size(128, 1, 1)); }
            if (!sk("k_gdn_conv")) { e->setComputePipelineState(gpu.pipe("k_gdn_conv")); e->setBuffer(bufQkv, 0, 0); e->setBuffer(convState, (uint64_t)si * 3 * kGdnQkvDim * 4, 1); e->setBuffer(wbuf, L.conv, 2); e->setBuffer(bufConv, 0, 3);
              e->dispatchThreads(MTL::Size(kGdnQkvDim, 1, 1), MTL::Size(256, 1, 1)); }
            if (!sk("k_gdn_step")) { e->setComputePipelineState(gpu.pipe("k_gdn_step")); e->setBuffer(bufConv, 0, 0); e->setBuffer(bufAlpha, 0, 1); e->setBuffer(bufBeta, 0, 2); e->setBuffer(wbuf, L.ssmA, 3); e->setBuffer(wbuf, L.dt, 4);
              e->setBuffer(gdnState, (uint64_t)si * kGdnVHeads * 128 * 128 * 4, 5); e->setBuffer(bufZ, 0, 6); e->setBuffer(wbuf, L.ssmNorm, 7); e->setBuffer(bufGdnOut, 0, 8);
              e->dispatchThreadgroups(MTL::Size(kGdnVHeads, 1, 1), MTL::Size(128, 1, 1)); }
            dRmsFwht(e, bufGdnOut, nullptr, sg6144, 0, bufXp, 6144, 2);
            dGemvPtq1(e, L.ssmOut, bufXp, bufTmp, 6144, H, "gemv_ssmout");
        } else {
            const uint32_t ai = (uint32_t)L.stateIdx; const uint64_t kvOff = (uint64_t)ai * maxCtx * kKvDim * 2;
            dGemvPtq1(e, L.wq, bufXp, bufQfull, H, kQProjOut, "gemv_wq");
            dGemvPtq1(e, L.wk, bufXp, bufK, H, kKvDim, "gemv_wk");
            dGemvPtq1(e, L.wv, bufXp, bufV, H, kKvDim, "gemv_wv");
            if (!sk("k_attn_prep")) { e->setComputePipelineState(gpu.pipe("k_attn_prep")); e->setBuffer(bufQfull, 0, 0); e->setBuffer(bufK, 0, 1); e->setBuffer(bufV, 0, 2); e->setBuffer(wbuf, L.qNorm, 3); e->setBuffer(wbuf, L.kNorm, 4);
              e->setBuffer(bufQ, 0, 5); e->setBuffer(bufGate, 0, 6); e->setBuffer(kcache, kvOff, 7); e->setBuffer(vcache, kvOff, 8); e->setBytes(&pos, 4, 9);
              e->dispatchThreadgroups(MTL::Size(32, 1, 1), MTL::Size(256, 1, 1)); }
            if (!sk("k_attn_split")) { uint32_t npos = pos + 1, S = std::min<uint32_t>(32, std::max<uint32_t>(1, npos / 64));
              e->setComputePipelineState(gpu.pipe("k_attn_split")); e->setBuffer(bufQ, 0, 0); e->setBuffer(kcache, kvOff, 1); e->setBuffer(vcache, kvOff, 2); e->setBuffer(bufPM, 0, 3); e->setBuffer(bufPL, 0, 4); e->setBuffer(bufPA, 0, 5); e->setBytes(&npos, 4, 6);
              e->dispatchThreadgroups(MTL::Size(S, kKvHeads, 1), MTL::Size(256, 1, 1));
              e->setComputePipelineState(gpu.pipe("k_attn_combine")); e->setBuffer(bufPM, 0, 0); e->setBuffer(bufPL, 0, 1); e->setBuffer(bufPA, 0, 2); e->setBuffer(bufGate, 0, 3); e->setBuffer(bufAttnO, 0, 4); e->setBytes(&S, 4, 5);
              e->dispatchThreadgroups(MTL::Size(kHeads, 1, 1), MTL::Size(256, 1, 1)); }
            dRmsFwht(e, bufAttnO, nullptr, sg6144, 0, bufXp, 6144, 3);
            dGemvPtq1(e, L.wo, bufXp, bufTmp, 6144, H, "gemv_wo");
        }
        if (dbgSlot >= 0) dCopy(e, bufTmp, dbgBlk, ((uint64_t)il * dbgTokens + dbgSlot) * H * 4, H);
        fusedNorm(bufXs[c], bufTmp, (uint64_t)(il * 2 + 1) * H * 4, bufXs[c ^ 1], layers[il].postNorm, false); c ^= 1;     // attention residual + post-attention norm
        dGemvPtq1(e, L.ffnGate, bufXp, bufG, H, kFfn, "gemv_gate");
        dGemvPtq1(e, L.ffnUp, bufXp, bufU, H, kFfn, "gemv_up");
        dRmsFwht(e, bufG, bufU, sg17408, 0, bufXp, kFfn, 1);
        dGemvPtq1(e, L.ffnDown, bufXp, bufTmp, kFfn, H, "gemv_down");
        if (il + 1 < kLayers) fusedNorm(bufXs[c], bufTmp, (uint64_t)((il + 1) * 2) * H * 4, bufXs[c ^ 1], layers[il + 1].attnNorm, !layers[il + 1].isAttn);   // FFN residual + next layer's input norm
        else fusedNorm(bufXs[c], bufTmp, (uint64_t)(kLayers * 2) * H * 4, bufXs[c ^ 1], outNorm, saveHidden);                                         // FFN residual + final norm
        c ^= 1;
        if (dbgSlot >= 0) dCopy(e, bufXs[c], dbgRes, ((uint64_t)il * dbgTokens + dbgSlot) * H * 4, H);
    }
    if (wantLogits) {
        dGemvPtq1(e, outW, bufXp, bufLogits, H, kVocab, "gemv_lmhead");
        uint32_t n = kVocab; e->setComputePipelineState(gpu.pipe("k_argmax")); e->setBuffer(bufLogits, 0, 0); e->setBuffer(bufArg, 0, 1); e->setBytes(&n, 4, 2); e->dispatchThreadgroups(MTL::Size(1, 1, 1), MTL::Size(1024, 1, 1));
    }
    e->endEncoding(); cb->commit(); cb->waitUntilCompleted();
    if (cb->status() != MTL::CommandBufferStatusCompleted) { fprintf(stderr, "GPU command buffer failed: %s\n", cb->error() ? cb->error()->localizedDescription()->utf8String() : "?"); exit(1); }
    int r = wantLogits ? *(int*)bufArg->contents() : -1; pool->release(); return r;
}


// ------------------------------------------------------------------------------------------------------------------------------
// Batched prefill (path A: exact fp16). Weights are expanded to fp16 on the GPU matrix by matrix, then multiplied by the matrix unit.
void Engine::dGemm(MTL::ComputeCommandEncoder* e, MTL::Buffer* A, MTL::Buffer* B, uint64_t bOff, MTL::Buffer* C, uint32_t Mpad, uint32_t K, uint32_t N) {
    if (sk("p_gemm")) return;
    uint32_t dims[4] = {Mpad, N, K, 0};
    e->setComputePipelineState(gpu.pipe("k_gemm_f16")); e->setBuffer(A, 0, 0); e->setBuffer(B, bOff, 1); e->setBuffer(C, 0, 2); e->setBytes(dims, 16, 3);
    e->dispatchThreadgroups(MTL::Size(N / 64, Mpad / 32, 1), MTL::Size(32, 1, 1));
}
void Engine::dExpandGemm(MTL::ComputeCommandEncoder* e, uint64_t wOff, MTL::Buffer* A, MTL::Buffer* C, uint32_t Mpad, uint32_t K, uint32_t N) {
    if (int8Mode) {
        if (!sk("p_expand")) { e->setComputePipelineState(gpu.pipe("k_expand_i8")); e->setBuffer(wbuf, wOff, 0); e->setBuffer(pStage8, 0, 1); e->setBuffer(pStageSW, 0, 2); e->setBytes(&K, 4, 3); e->dispatchThreadgroups(MTL::Size(K / 128, N / 64, 1), MTL::Size(256, 1, 1)); }
        if (!sk("p_gemm")) { uint32_t dims[4] = {Mpad, N, K, 0}; e->setComputePipelineState(gpu.pipe("k_gemm_i8")); e->setBuffer(pXq, 0, 0); e->setBuffer(pStage8, 0, 1); e->setBuffer(pStageSW, 0, 2); e->setBuffer(pSx, 0, 3); e->setBuffer(C, 0, 4); e->setBytes(dims, 16, 5); e->dispatchThreadgroups(MTL::Size(N / 64, Mpad / 64, 1), MTL::Size(128, 1, 1)); }
        return;
    }
    if (!sk("p_expand")) { e->setComputePipelineState(gpu.pipe("k_expand_ptq1")); e->setBuffer(wbuf, wOff, 0); e->setBuffer(pStage, 0, 1); e->setBytes(&K, 4, 2);
        e->dispatchThreadgroups(MTL::Size(K / 128, N / 64, 1), MTL::Size(256, 1, 1)); }
    dGemm(e, A, pStage, 0, C, Mpad, K, N);
}

int Engine::prefill(const int* toks, uint32_t M, uint32_t pos0, int dbgBase, bool wantLogits, uint32_t nLogits) {
    if (M == 0 || M > kMaxChunk) { fprintf(stderr, "prefill: bad chunk size %u\n", M); exit(1); }
    NS::AutoreleasePool* pool = NS::AutoreleasePool::alloc()->init();
    memcpy(pTok->contents(), toks, M * 4);
    const uint32_t H = kHidden, Mpad32 = (M + 31) / 32 * 32, Mpad = int8Mode ? (M + 63) / 64 * 64 : Mpad32;
    MTL::CommandBuffer* cb = gpu.queue->commandBuffer(); MTL::ComputeCommandEncoder* e = cb->computeCommandEncoder();
    int c = 0;
    auto quant = [&](uint32_t K) { if (!int8Mode || sk("p_quant")) return; e->setComputePipelineState(gpu.pipe("k_quant_rows")); e->setBuffer(pXp, 0, 0); e->setBuffer(pXq, 0, 1); e->setBuffer(pSx, 0, 2); e->setBytes(&K, 4, 3); e->dispatchThreadgroups(MTL::Size(M, 1, 1), MTL::Size(256, 1, 1)); };
    if (!sk("p_embed")) { e->setComputePipelineState(gpu.pipe("k_embed_b")); e->setBuffer(wbuf, tokEmbd, 0); e->setBuffer(pTok, 0, 1); e->setBuffer(sg5120, 0, 2); e->setBuffer(pX[0], 0, 3); e->dispatchThreadgroups(MTL::Size(5, M, 1), MTL::Size(256, 1, 1)); }
    auto fusedNorm = [&](MTL::Buffer* x, MTL::Buffer* y, uint64_t gsOff, MTL::Buffer* xnew, uint64_t gammaOff, bool wantH) {
        if (sk("p_norm")) return;
        uint32_t flags = (y ? 1u : 0u) | (wantH ? 2u : 0u);
        e->setComputePipelineState(gpu.pipe("k_add_rms_fwht_t")); e->setBuffer(x, 0, 0); e->setBuffer(y ? y : x, 0, 1); e->setBuffer(gsAll, gsOff, 2); e->setBuffer(pXp, 0, 3);
        e->setBuffer(xnew, 0, 4); e->setBuffer(wbuf, gammaOff, 5); e->setBuffer(pHh, 0, 6); e->setBytes(&flags, 4, 7);
        e->dispatchThreadgroups(MTL::Size(M, 1, 1), MTL::Size(256, 1, 1));
        quant(5120);
    };
    auto fwhtB = [&](MTL::Buffer* src, MTL::Buffer* src2, MTL::Buffer* sgn, uint32_t W, uint32_t mode) {
        if (sk("p_fwht")) return;
        e->setComputePipelineState(gpu.pipe("k_rms_fwht_b")); e->setBuffer(src, 0, 0); e->setBuffer(src2 ? src2 : src, 0, 1); e->setBuffer(sgn, 0, 2); e->setBuffer(pXp, 0, 3); e->setBytes(&W, 4, 4); e->setBytes(&mode, 4, 5);
        e->dispatchThreadgroups(MTL::Size(W / 1024, M, 1), MTL::Size(256, 1, 1));
        quant(W);
    };
    fusedNorm(pX[c], nullptr, 0, pX[c ^ 1], layers[0].attnNorm, !layers[0].isAttn); c ^= 1;
    for (uint32_t il = 0; il < kLayers; il++) {
        const LayerW& L = layers[il];
        if (!L.isAttn) {
            const uint32_t si = (uint32_t)L.stateIdx; const uint64_t cOff = (uint64_t)si * 3 * kGdnQkvDim * 4;
            dExpandGemm(e, L.qkv, pXp, pQkv, Mpad, H, kGdnQkvDim);
            dExpandGemm(e, L.gate, pXp, pZ, Mpad, H, kGdnZDim);
            if (!sk("p_gdn_ab")) dGemm(e, pHh, wabAll, (uint64_t)si * 2 * 5120 * 64 * 2, pAB, Mpad32, H, 128);
            if (!sk("p_gdn_conv")) { e->setComputePipelineState(gpu.pipe("k_gdn_conv_t")); e->setBuffer(pQkv, 0, 0); e->setBuffer(convState, cOff, 1); e->setBuffer(wbuf, L.conv, 2); e->setBuffer(pConv, 0, 3); e->setBytes(&M, 4, 4);
              e->dispatchThreads(MTL::Size(kGdnQkvDim / 4, (M + 15) / 16, 1), MTL::Size(64, 1, 1));
              e->setComputePipelineState(gpu.pipe("k_gdn_conv_state")); e->setBuffer(pQkv, 0, 0); e->setBuffer(convState, cOff, 1); e->setBytes(&M, 4, 2);
              e->dispatchThreads(MTL::Size(kGdnQkvDim, 1, 1), MTL::Size(256, 1, 1)); }
            if (!sk("p_gdn_step")) {
                e->setComputePipelineState(gpu.pipe("k_gdn_qk_b")); e->setBuffer(pConv, 0, 0); e->setBuffer(pQk, 0, 1); e->dispatchThreadgroups(MTL::Size(M, 32, 1), MTL::Size(128, 1, 1));
                e->setComputePipelineState(gpu.pipe("k_gdn_gates_b")); e->setBuffer(pAB, 0, 0); e->setBuffer(wbuf, L.ssmA, 1); e->setBuffer(wbuf, L.dt, 2); e->setBuffer(pGb, 0, 3); e->setBytes(&M, 4, 4); e->dispatchThreads(MTL::Size(M * 48, 1, 1), MTL::Size(256, 1, 1));
                e->setComputePipelineState(gpu.pipe("k_gdn_rec_b2")); e->setBuffer(pConv, 0, 0); e->setBuffer(pQk, 0, 1); e->setBuffer(pGb, 0, 2); e->setBuffer(gdnState, (uint64_t)si * kGdnVHeads * 128 * 128 * 4, 3); e->setBuffer(pRaw, 0, 4); e->setBytes(&M, 4, 5);
                e->dispatchThreadgroups(MTL::Size(kGdnVHeads, 16, 1), MTL::Size(32, 1, 1));
                e->setComputePipelineState(gpu.pipe("k_gdn_gnorm_b")); e->setBuffer(pRaw, 0, 0); e->setBuffer(pZ, 0, 1); e->setBuffer(wbuf, L.ssmNorm, 2); e->setBuffer(pGdnOut, 0, 3); e->dispatchThreadgroups(MTL::Size(M, 48, 1), MTL::Size(128, 1, 1));
            }
            fwhtB(pGdnOut, nullptr, sg6144, 6144, 2);
            dExpandGemm(e, L.ssmOut, pXp, pTmp, Mpad, 6144, H);
        } else {
            const uint32_t ai = (uint32_t)L.stateIdx; const uint64_t kvOff = (uint64_t)ai * maxCtx * kKvDim * 2;
            dExpandGemm(e, L.wq, pXp, pQfull, Mpad, H, kQProjOut);
            dExpandGemm(e, L.wk, pXp, pK, Mpad, H, kKvDim);
            dExpandGemm(e, L.wv, pXp, pV, Mpad, H, kKvDim);
            if (!sk("p_attn_prep")) { e->setComputePipelineState(gpu.pipe("k_attn_prep_b")); e->setBuffer(pQfull, 0, 0); e->setBuffer(pK, 0, 1); e->setBuffer(pV, 0, 2); e->setBuffer(wbuf, L.qNorm, 3); e->setBuffer(wbuf, L.kNorm, 4);
              e->setBuffer(pQ, 0, 5); e->setBuffer(pGate, 0, 6); e->setBuffer(kcache, kvOff, 7); e->setBuffer(vcache, kvOff, 8); e->setBytes(&pos0, 4, 9); { uint32_t h00 = 0; e->setBytes(&h00, 4, 10); }
              e->dispatchThreadgroups(MTL::Size(32, M, 1), MTL::Size(256, 1, 1)); }
            if (!sk("p_attn")) for (uint32_t base = 0; base < M; base += 256) {
                const uint32_t mt = std::min<uint32_t>(256, M - base), posBase = pos0 + base, S = 8;
                e->setComputePipelineState(gpu.pipe("k_attn_split_b")); e->setBuffer(pQ, (uint64_t)base * 6144 * 4, 0); e->setBuffer(kcache, kvOff, 1); e->setBuffer(vcache, kvOff, 2); e->setBuffer(pPM, 0, 3); e->setBuffer(pPL, 0, 4); e->setBuffer(pPA, 0, 5); e->setBytes(&posBase, 4, 6); { uint32_t fp0 = 0; e->setBytes(&fp0, 4, 7); }
                e->dispatchThreadgroups(MTL::Size(S, kKvHeads, mt), MTL::Size(256, 1, 1));
                e->setComputePipelineState(gpu.pipe("k_attn_combine_b")); e->setBuffer(pPM, 0, 0); e->setBuffer(pPL, 0, 1); e->setBuffer(pPA, 0, 2); e->setBuffer(pGate, (uint64_t)base * 6144 * 4, 3); e->setBuffer(pAttnO, (uint64_t)base * 6144 * 4, 4); e->setBytes(&S, 4, 5);
                e->dispatchThreadgroups(MTL::Size(kHeads, mt, 1), MTL::Size(256, 1, 1));
            }
            fwhtB(pAttnO, nullptr, sg6144, 6144, 3);
            dExpandGemm(e, L.wo, pXp, pTmp, Mpad, 6144, H);
        }
        if (dbgBase >= 0) dCopy(e, pTmp, dbgBlk, ((uint64_t)il * dbgTokens + dbgBase) * H * 4, M * H);
        fusedNorm(pX[c], pTmp, (uint64_t)(il * 2 + 1) * H * 4, pX[c ^ 1], layers[il].postNorm, false); c ^= 1;
        dExpandGemm(e, L.ffnGate, pXp, pG, Mpad, H, kFfn);
        dExpandGemm(e, L.ffnUp, pXp, pU, Mpad, H, kFfn);
        fwhtB(pG, pU, sg17408, kFfn, 1);
        dExpandGemm(e, L.ffnDown, pXp, pTmp, Mpad, kFfn, H);
        if (il + 1 < kLayers) { fusedNorm(pX[c], pTmp, (uint64_t)((il + 1) * 2) * H * 4, pX[c ^ 1], layers[il + 1].attnNorm, !layers[il + 1].isAttn); }
        else { uint32_t n = M * H; e->setComputePipelineState(gpu.pipe("k_add2")); e->setBuffer(pX[c], 0, 0); e->setBuffer(pTmp, 0, 1); e->setBuffer(pX[c ^ 1], 0, 2); e->setBytes(&n, 4, 3); e->dispatchThreads(MTL::Size(n, 1, 1), MTL::Size(256, 1, 1)); }
        c ^= 1;
        if (dbgBase >= 0) dCopy(e, pX[c], dbgRes, ((uint64_t)il * dbgTokens + dbgBase) * H * 4, M * H);
    }
    if (wantLogits) {   // the last nLogits tokens -> final norm -> lm_head (decode-style kernels); slot j of bufLogitsAll holds the logits of token M-nl+j
        const uint32_t nl = std::min<uint32_t>(std::min<uint32_t>(nLogits ? nLogits : 1, M), 128);
        for (uint32_t j = 0; j < nl; j++) {
            uint32_t flags = 0; const uint64_t rowOff = (uint64_t)(M - nl + j) * H * 4;
            e->setComputePipelineState(gpu.pipe("k_add_rms_fwht")); e->setBuffer(pX[c], rowOff, 0); e->setBuffer(pX[c], rowOff, 1); e->setBuffer(gsAll, (uint64_t)(kLayers * 2) * H * 4, 2); e->setBuffer(bufXp, 0, 3);
            e->setBuffer(bufXs[0], 0, 4); e->setBuffer(wbuf, outNorm, 5); e->setBuffer(bufH, 0, 6); e->setBytes(&flags, 4, 7); e->dispatchThreadgroups(MTL::Size(5, 1, 1), MTL::Size(256, 1, 1));
            uint32_t K = H, N = kVocab; e->setComputePipelineState(gpu.pipe("k_gemv_v1_nr4_sg4")); e->setBuffer(wbuf, outW, 0); e->setBuffer(bufXp, 0, 1); e->setBuffer(bufLogitsAll, (uint64_t)j * kVocab * 4, 2); e->setBytes(&K, 4, 3); e->setBytes(&N, 4, 4);
            e->dispatchThreadgroups(MTL::Size((N + 15) / 16, 1, 1), MTL::Size(128, 1, 1));
        }
        uint32_t n = kVocab; e->setComputePipelineState(gpu.pipe("k_argmax")); e->setBuffer(bufLogitsAll, (uint64_t)(nl - 1) * kVocab * 4, 0); e->setBuffer(bufArg, 0, 1); e->setBytes(&n, 4, 2); e->dispatchThreadgroups(MTL::Size(1, 1, 1), MTL::Size(1024, 1, 1));
    }
    e->endEncoding(); cb->commit(); cb->waitUntilCompleted();
    if (cb->status() != MTL::CommandBufferStatusCompleted) { fprintf(stderr, "GPU command buffer failed: %s\n", cb->error() ? cb->error()->localizedDescription()->utf8String() : "?"); exit(1); }
    if (wantLogits) { const uint32_t nl = std::min<uint32_t>(std::min<uint32_t>(nLogits ? nLogits : 1, M), 128); memcpy(bufLogits->contents(), (const float*)bufLogitsAll->contents() + (size_t)(nl - 1) * kVocab, (size_t)kVocab * 4); }
    int r = wantLogits ? *(int*)bufArg->contents() : -1; pool->release(); return r;
}

// ------------------------------------------------------------------------------------------------------------------------------
// MTP head: bind the Q8_0 block and allocate its cache and scratch buffers (called from load()).
void Engine::loadMtp() {
    auto has = [&](const char* n) { return gg.tensors.count(n) > 0; };
    if (!has("blk.64.nextn.eh_proj.weight")) { hasMtp = false; return; }
    hasMtp = true; auto T = [&](const char* n) { return gg.tensor(n).fileOffset; };
    mtp.ehProj = T("blk.64.nextn.eh_proj.weight"); mtp.enorm = T("blk.64.nextn.enorm.weight"); mtp.hnorm = T("blk.64.nextn.hnorm.weight"); mtp.headNorm = T("blk.64.nextn.shared_head_norm.weight");
    mtp.attnNorm = T("blk.64.attn_norm.weight"); mtp.postNorm = T("blk.64.post_attention_norm.weight"); mtp.wq = T("blk.64.attn_q.weight"); mtp.wk = T("blk.64.attn_k.weight"); mtp.wv = T("blk.64.attn_v.weight");
    mtp.wo = T("blk.64.attn_output.weight"); mtp.qNorm = T("blk.64.attn_q_norm.weight"); mtp.kNorm = T("blk.64.attn_k_norm.weight"); mtp.ffnGate = T("blk.64.ffn_gate.weight"); mtp.ffnUp = T("blk.64.ffn_up.weight"); mtp.ffnDown = T("blk.64.ffn_down.weight");
    const size_t R = 4;
    mK = gpu.alloc((size_t)maxCtx * kKvDim * 2); mV = gpu.alloc((size_t)maxCtx * kKvDim * 2); mTok = gpu.alloc(R * 4); mE = gpu.alloc(R * 5120 * 4); mHin = gpu.alloc(R * 5120 * 4); mCat = gpu.alloc(R * 10240 * 4);
    mX = gpu.alloc(R * 5120 * 4); mH = gpu.alloc(R * 5120 * 4); mQfull = gpu.alloc(R * kQProjOut * 4); mKp = gpu.alloc(R * kKvDim * 4); mVp = gpu.alloc(R * kKvDim * 4); mQ = gpu.alloc(R * 6144 * 4); mGate = gpu.alloc(R * 6144 * 4);
    mAttnO = gpu.alloc(R * 6144 * 4); mTmp = gpu.alloc(R * 5120 * 4); mY = gpu.alloc(R * 5120 * 4); mG = gpu.alloc(R * 17408 * 4); mU = gpu.alloc(R * 17408 * 4); mPM = gpu.alloc(R * 24 * 8 * 4); mPL = gpu.alloc(R * 24 * 8 * 4); mPA = gpu.alloc(R * 24 * 8 * 256 * 4);
    // shared head norm folded with the sign vector (index 2*kLayers + 1 of gsAll)
    float* gs = (float*)gsAll->contents() + (size_t)(kLayers * 2 + 1) * kHidden; const float* s5120 = (const float*)sg5120->contents(); const float* g = (const float*)gg.bytes(mtp.headNorm);
    for (uint32_t i = 0; i < kHidden; i++) gs[i] = g[i] * s5120[i];
}

int Engine::mtpRows(uint32_t R, const int* toks, const float* hin, uint32_t pos0, bool wantLogits) {
    if (!hasMtp || R == 0 || R > 2) { fprintf(stderr, "mtpRows: bad call\n"); exit(1); }
    NS::AutoreleasePool* pool = NS::AutoreleasePool::alloc()->init();
    memcpy(mTok->contents(), toks, R * 4); memcpy(mHin->contents(), hin, (size_t)R * 5120 * 4);
    const uint32_t H = kHidden; const std::string tn = R == 1 ? "k_gemv_q8_t1" : "k_gemv_q8_t2";
    MTL::CommandBuffer* cb = gpu.queue->commandBuffer(); MTL::ComputeCommandEncoder* e = cb->computeCommandEncoder();
    auto gq = [&](uint64_t off, MTL::Buffer* x, MTL::Buffer* y, uint32_t K, uint32_t N) { e->setComputePipelineState(gpu.pipe(tn.c_str())); e->setBuffer(wbuf, off, 0); e->setBuffer(x, 0, 1); e->setBuffer(y, 0, 2); e->setBytes(&K, 4, 3); e->setBytes(&N, 4, 4); e->dispatchThreadgroups(MTL::Size((N + 7) / 8, 1, 1), MTL::Size(256, 1, 1)); };
    auto rows = [&](MTL::Buffer* src, uint64_t wOff, MTL::Buffer* out) { uint32_t w = H; e->setComputePipelineState(gpu.pipe("k_rmsnorm_rows")); e->setBuffer(src, 0, 0); e->setBuffer(wbuf, wOff, 1); e->setBuffer(out, 0, 2); e->setBytes(&w, 4, 3); e->dispatchThreadgroups(MTL::Size(R, 1, 1), MTL::Size(256, 1, 1)); };
    auto add = [&](MTL::Buffer* a, MTL::Buffer* b, MTL::Buffer* out, uint32_t n) { e->setComputePipelineState(gpu.pipe("k_add3")); e->setBuffer(a, 0, 0); e->setBuffer(b, 0, 1); e->setBuffer(out, 0, 2); e->setBytes(&n, 4, 3); e->dispatchThreads(MTL::Size(n, 1, 1), MTL::Size(256, 1, 1)); };
    { e->setComputePipelineState(gpu.pipe("k_embed_b")); e->setBuffer(wbuf, tokEmbd, 0); e->setBuffer(mTok, 0, 1); e->setBuffer(sg5120, 0, 2); e->setBuffer(mE, 0, 3); e->dispatchThreadgroups(MTL::Size(5, R, 1), MTL::Size(256, 1, 1)); }
    { e->setComputePipelineState(gpu.pipe("k_mtp_cat")); e->setBuffer(mE, 0, 0); e->setBuffer(mHin, 0, 1); e->setBuffer(wbuf, mtp.enorm, 2); e->setBuffer(wbuf, mtp.hnorm, 3); e->setBuffer(mCat, 0, 4); e->dispatchThreadgroups(MTL::Size(R, 1, 1), MTL::Size(256, 1, 1)); }
    gq(mtp.ehProj, mCat, mX, 10240, H);
    rows(mX, mtp.attnNorm, mH);
    gq(mtp.wq, mH, mQfull, H, kQProjOut); gq(mtp.wk, mH, mKp, H, kKvDim); gq(mtp.wv, mH, mVp, H, kKvDim);
    { uint32_t h00 = 0; e->setComputePipelineState(gpu.pipe("k_attn_prep_b")); e->setBuffer(mQfull, 0, 0); e->setBuffer(mKp, 0, 1); e->setBuffer(mVp, 0, 2); e->setBuffer(wbuf, mtp.qNorm, 3); e->setBuffer(wbuf, mtp.kNorm, 4);
      e->setBuffer(mQ, 0, 5); e->setBuffer(mGate, 0, 6); e->setBuffer(mK, 0, 7); e->setBuffer(mV, 0, 8); e->setBytes(&pos0, 4, 9); e->setBytes(&h00, 4, 10); e->dispatchThreadgroups(MTL::Size(32, R, 1), MTL::Size(256, 1, 1)); }
    { uint32_t S = std::min<uint32_t>(8, std::max<uint32_t>(1, (pos0 + R) / 64)), first = 1;
      e->setComputePipelineState(gpu.pipe("k_attn_split_b")); e->setBuffer(mQ, 0, 0); e->setBuffer(mK, 0, 1); e->setBuffer(mV, 0, 2); e->setBuffer(mPM, 0, 3); e->setBuffer(mPL, 0, 4); e->setBuffer(mPA, 0, 5); e->setBytes(&pos0, 4, 6); e->setBytes(&first, 4, 7);
      e->dispatchThreadgroups(MTL::Size(S, kKvHeads, R), MTL::Size(256, 1, 1));
      e->setComputePipelineState(gpu.pipe("k_attn_combine_b")); e->setBuffer(mPM, 0, 0); e->setBuffer(mPL, 0, 1); e->setBuffer(mPA, 0, 2); e->setBuffer(mGate, 0, 3); e->setBuffer(mAttnO, 0, 4); e->setBytes(&S, 4, 5);
      e->dispatchThreadgroups(MTL::Size(kHeads, R, 1), MTL::Size(256, 1, 1)); }
    gq(mtp.wo, mAttnO, mTmp, 6144, H);
    add(mTmp, mX, mY, R * H);                         // attention residual (inpSA = eh_proj output)
    rows(mY, mtp.postNorm, mH);
    gq(mtp.ffnGate, mH, mG, H, kFfn); gq(mtp.ffnUp, mH, mU, H, kFfn);
    { uint32_t n = R * kFfn; e->setComputePipelineState(gpu.pipe("k_swiglu")); e->setBuffer(mG, 0, 0); e->setBuffer(mU, 0, 1); e->setBuffer(mG, 0, 2); e->setBytes(&n, 4, 3); e->dispatchThreads(MTL::Size(n, 1, 1), MTL::Size(256, 1, 1)); }
    gq(mtp.ffnDown, mG, mTmp, kFfn, H);
    add(mTmp, mY, mX, R * H);                         // FFN residual -> block output (mX reused)
    if (wantLogits) {
        uint32_t flags = 0; const uint64_t rowOff = (uint64_t)(R - 1) * H * 4;
        e->setComputePipelineState(gpu.pipe("k_add_rms_fwht")); e->setBuffer(mX, rowOff, 0); e->setBuffer(mX, rowOff, 1); e->setBuffer(gsAll, (uint64_t)(kLayers * 2 + 1) * H * 4, 2); e->setBuffer(bufXp, 0, 3);
        e->setBuffer(bufXs[0], 0, 4); e->setBuffer(wbuf, mtp.headNorm, 5); e->setBuffer(bufH, 0, 6); e->setBytes(&flags, 4, 7); e->dispatchThreadgroups(MTL::Size(5, 1, 1), MTL::Size(256, 1, 1));
        dGemvPtq1(e, outW, bufXp, bufLogits, H, kVocab, "gemv_lmhead");
        uint32_t n = kVocab; e->setComputePipelineState(gpu.pipe("k_argmax")); e->setBuffer(bufLogits, 0, 0); e->setBuffer(bufArg, 0, 1); e->setBytes(&n, 4, 2); e->dispatchThreadgroups(MTL::Size(1, 1, 1), MTL::Size(1024, 1, 1));
    }
    e->endEncoding(); cb->commit(); cb->waitUntilCompleted();
    if (cb->status() != MTL::CommandBufferStatusCompleted) { fprintf(stderr, "GPU command buffer failed (mtp): %s\n", cb->error() ? cb->error()->localizedDescription()->utf8String() : "?"); exit(1); }
    int r = wantLogits ? *(int*)bufArg->contents() : -1; pool->release(); return r;
}
