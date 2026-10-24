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
    gsAll = gpu.alloc((size_t)(kLayers * 2 + 1) * kHidden * 4); float* gs = (float*)gsAll->contents();
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
    if (dbgTokens) { dbgRes = gpu.alloc((size_t)kLayers * dbgTokens * kHidden * 4); dbgBlk = gpu.alloc((size_t)kLayers * dbgTokens * kHidden * 4); }
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
        else fusedNorm(bufXs[c], bufTmp, (uint64_t)(kLayers * 2) * H * 4, bufXs[c ^ 1], layers[il].postNorm, false);                                         // FFN residual + final norm
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
