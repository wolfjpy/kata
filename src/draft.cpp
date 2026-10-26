// DFlash2 draft: weight binding, ring buffers and the draft's own copy of the shared target tensors.
#include "draft.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>

using namespace bonsai;

static std::string blkName(const char* fmt, int il) { char b[96]; snprintf(b, sizeof b, fmt, il); return b; }

uint32_t Draft::signOffFor(uint32_t W) const {
    const int idx = (W == 4096u) ? 0 : (W == 5120u) ? 1 : (W == 17408u) ? 2 : 3;
    if ((W == 4096u && signOff[0] == 0xffffffffu)) { fprintf(stderr, "draft: no sign vector for width %u\n", W); exit(1); }
    return signOff[idx];
}
void Draft::fwhtIn(Engine& eng, MTL::ComputeCommandEncoder* e, MTL::Buffer* inF, MTL::Buffer* outH, uint32_t W, uint32_t T) {
    uint32_t mode = 0, off = signOffFor(W);
    e->setComputePipelineState(eng.gpu.pipe("k_rms_fwht_b")); e->setBuffer(inF, 0, 0); e->setBuffer(inF, 0, 1); e->setBuffer(signs, (uint64_t)off * 4, 2);
    e->setBuffer(outH, 0, 3); e->setBytes(&W, 4, 4); e->setBytes(&mode, 4, 5); e->dispatchThreadgroups(MTL::Size(W / 1024, T, 1), MTL::Size(256, 1, 1));
}

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
    // ternary drafters carry their own Hadamard basis: sign_widths lists the widths (4096/5120/17408/25600) and sign_values their concatenated signs
    std::vector<int32_t> widths = gg.signedInts("prism.hadamard.sign_widths"), vals = gg.signedInts("prism.hadamard.sign_values");
    if (!widths.empty() && widths.size() == 4) {
        rotated = true; uint32_t total = 0;
        for (int i = 0; i < 4; i++) { signOff[i] = total; total += (uint32_t)widths[(size_t)i]; }
        if ((uint32_t)vals.size() != total) { fprintf(stderr, "draft: sign_values %zu != %u\n", vals.size(), total); exit(1); }
        signs = eng.gpu.alloc((size_t)total * 4); float* sp = (float*)signs->contents(); for (uint32_t i = 0; i < total; i++) sp[i] = (float)vals[i];
        ones = eng.gpu.alloc(25600 * 4); float* op = (float*)ones->contents(); for (uint32_t i = 0; i < 25600; i++) op[i] = 1.0f;
        printf("[draft] rotated basis: sign widths %d/%d/%d/%d\n", widths[0], widths[1], widths[2], widths[3]);
    } else { for (int i = 0; i < 4; i++) signOff[i] = 0; }
    for (auto& kv : gg.tensors) bytes += kv.second.nbytes;
    // the draft's weight buffer: one zero-copy wrap of its own mapping, like the trunk
    wbuf = eng.gpu.wrap(gg.map, gg.mapLen);
    // ring K/V (one ring per layer) and the block scratch
    const size_t ringBytes = (size_t)kLayers * kRing * kKvHeads * kHeadDim * 2;
    kbuf = eng.gpu.alloc(ringBytes); vbuf = eng.gpu.alloc(ringBytes);
    catH = eng.gpu.alloc((size_t)kNMax * 25600 * 2); catF = eng.gpu.alloc((size_t)kNMax * 25600 * 4); cf = eng.gpu.alloc((size_t)kNMax * kH * 4); cH = eng.gpu.alloc((size_t)kNMax * kH * 2);
    fcf = eng.gpu.alloc((size_t)kNMax * kH * 4); kvf = eng.gpu.alloc((size_t)kNMax * kKvHeads * kHeadDim * 4);
    nrmH = eng.gpu.alloc(kBlock * kH * 2); nrmf = eng.gpu.alloc(kBlock * kH * 4); aoutH = eng.gpu.alloc(kBlock * kHeads * kHeadDim * 2);
    xA = eng.gpu.alloc(kBlock * kH * 4); xB = eng.gpu.alloc(kBlock * kH * 4); tmpA = eng.gpu.alloc(kBlock * kH * 4);
    fout2 = eng.gpu.alloc(kBlock * kFfn * 4); ffnH = eng.gpu.alloc(kBlock * kFfn * 2); tmpB = eng.gpu.alloc(kBlock * kH * 4); gU = eng.gpu.alloc(kBlock * kFfn * 4); kvf2 = eng.gpu.alloc((size_t)kNMax * kKvHeads * kHeadDim * 4);
    x = eng.gpu.alloc(kBlock * kH * 4); nrm = eng.gpu.alloc(kBlock * kH * 4);
    q = eng.gpu.alloc(kBlock * kHeads * kHeadDim * 4); k = eng.gpu.alloc(kBlock * kKvHeads * kHeadDim * 4);
    v = eng.gpu.alloc(kBlock * kKvHeads * kHeadDim * 4); aout = eng.gpu.alloc(kBlock * kHeads * kHeadDim * 4);
    tmp = eng.gpu.alloc(kBlock * kH * 4); dyn = eng.gpu.alloc(kBlock * kConvProj * 4);
    attnOut0 = eng.gpu.alloc(kBlock * kHeads * kHeadDim * 4); qOut0 = eng.gpu.alloc(kBlock * kHeads * kHeadDim * 4); convOut0 = eng.gpu.alloc(kBlock * kH * 4); convIn0 = eng.gpu.alloc(kBlock * kH * 4); convDyn0 = eng.gpu.alloc(kBlock * kConvProj * 4);
    fout = eng.gpu.alloc(kBlock * kFfn * 4); probs = eng.gpu.alloc(kBlock * kH * 2);
    tapsBuf = eng.gpu.alloc((size_t)kTargets * kNMax * kH * 2); tokbuf = eng.gpu.alloc(kBlock * 4);
    ok = true;
    printf("[draft] %u layers, %llu MB of weights, ring %u positions, %zu MB of K/V\n", kLayers, (unsigned long long)(bytes >> 20), kRing,
           (ringBytes * 2) >> 20);
    return true;
}

// c_t = rmsnorm(fc(concat taps), encNorm); K/V per layer = wk/wv c_t, then k_norm + rope into the ring.
void Draft::materialise(Engine& eng, const void* tapsFp16, uint32_t n) {
    if (n == 0) return;
    if (n > kBlock) { fprintf(stderr, "draft: materialise n=%u > %u\n", n, kBlock); exit(1); }
    const uint16_t* src = (const uint16_t*)tapsFp16; const uint32_t H = kH;
    MTL::CommandBuffer* cb = eng.gpu.queue->commandBuffer(); MTL::ComputeCommandEncoder* e = cb->computeCommandEncoder();
    if (rotated) {   // the fc input is the concatenated taps in the rotated basis: build it as float, transform, then use it as the matmul operand
        { _Float16* srcH = (_Float16*)src; float* dstF = (float*)catF->contents();
          for (uint32_t i = 0; i < n; i++) for (uint32_t t = 0; t < kTargets; t++)
              for (uint32_t k = 0; k < H; k++) dstF[(size_t)i * kTargets * H + t * H + k] = (float)srcH[((size_t)t * n + i) * H + k]; }
        fwhtIn(eng, e, catF, catH, kTargets * H, n);
    } else {
        for (uint32_t i = 0; i < n; i++) for (uint32_t t = 0; t < kTargets; t++)   // tap-major input -> row-major [n][5*H]
            memcpy((uint8_t*)catH->contents() + ((size_t)i * kTargets * H + t * H) * 2, src + ((size_t)t * n + i) * H, H * 2);
    }
    auto mm = [&](uint64_t off, MTL::Buffer* a, MTL::Buffer* y, uint32_t K, uint32_t N, bool) {
        const int qt = qtypeOf(off);                               // the decoder comes from the tensor's own quant type
        static const char* names[5] = {"k_draft_mm_m8_n16_g4_q0", "k_draft_mm_m8_n16_g4_q1", "k_draft_mm_m8_n16_g4_q2", "k_draft_mm_m8_n16_g4_q3", "k_draft_mm_m8_n16_g4_q4"};
        e->setComputePipelineState(eng.gpu.pipe(names[qt]));
        e->setBuffer(wbuf, off, 0); e->setBuffer(a, 0, 1); e->setBuffer(y, 0, 2); e->setBytes(&K, 4, 3); e->setBytes(&N, 4, 4);
        e->dispatchThreadgroups(MTL::Size((N + 15) / 16, 1, 1), MTL::Size(128, 1, 1));
    };
    auto rmsH = [&](MTL::Buffer* in, uint64_t wOff, MTL::Buffer* out) { uint32_t W = H; e->setComputePipelineState(eng.gpu.pipe("k_rmsnorm_rows_h"));
        e->setBuffer(in, 0, 0); e->setBuffer(wbuf, wOff, 1); e->setBuffer(out, 0, 2); e->setBytes(&W, 4, 3); e->dispatchThreadgroups(MTL::Size(n, 1, 1), MTL::Size(256, 1, 1)); };
    const bool dumpFc = getenv("BONSAI_DUMP_FC") != nullptr;
    mm(fc, catH, fcf, kTargets * H, H, false);                       // fc is Q4_K (25600 -> 5120)
    if (rotated) {   // c_t must be transformed in float: rmsnorm to a float buffer, then FWHT into the fp16 operand
        uint32_t W2 = H; e->setComputePipelineState(eng.gpu.pipe("k_rmsnorm_rows")); e->setBuffer(fcf, 0, 0); e->setBuffer(wbuf, encNorm, 1); e->setBuffer(cf, 0, 2); e->setBytes(&W2, 4, 3);
        e->dispatchThreadgroups(MTL::Size(n, 1, 1), MTL::Size(256, 1, 1));
        fwhtIn(eng, e, cf, cH, H, n);
    } else rmsH(fcf, encNorm, cH);
    MTL::Buffer* dFcf = nullptr; MTL::Buffer* dCh = nullptr;
    if (dumpFc) { dFcf = eng.gpu.alloc((size_t)n * H * 4); dCh = eng.gpu.alloc((size_t)n * H * 2);
        uint32_t n1 = n * H; e->setComputePipelineState(eng.gpu.pipe("k_copy_f32")); e->setBuffer(fcf, 0, 0); e->setBuffer(dFcf, 0, 1); e->setBytes(&n1, 4, 2); e->dispatchThreads(MTL::Size(n1, 1, 1), MTL::Size(256, 1, 1)); }
    const uint32_t HKV = kKvHeads, ring = kRing;
    if (rotated) { for (uint32_t il = 0; il < kLayers; il++) { mm(L[il].k, cH, kvf, H, HKV * kHeadDim, false); mm(L[il].v, cH, kvf2, H, HKV * kHeadDim, false);
        e->setComputePipelineState(eng.gpu.pipe("k_draft_ctx_kv")); e->setBuffer(kvf, 0, 0); e->setBuffer(kvf2, 0, 1); e->setBuffer(wbuf, L[il].kNorm, 2);
        e->setBuffer(kbuf, (uint64_t)il * kRing * HKV * kHeadDim * 2, 3); e->setBuffer(vbuf, (uint64_t)il * kRing * HKV * kHeadDim * 2, 4);
        e->setBytes(&frontier, 4, 5); e->setBytes(&HKV, 4, 6); e->setBytes(&ring, 4, 7); e->dispatchThreadgroups(MTL::Size(2 * HKV, n, 1), MTL::Size(128, 1, 1)); } }
    else for (uint32_t il = 0; il < kLayers; il++) {
        mm(L[il].k, cH, kvf, H, HKV * kHeadDim, false);
        mm(L[il].v, cH, kvf2, H, HKV * kHeadDim, false);
        const uint32_t nog = 0;
        e->setComputePipelineState(eng.gpu.pipe("k_draft_ctx_kv")); e->setBuffer(kvf, 0, 0); e->setBuffer(kvf2, 0, 1); e->setBuffer(wbuf, L[il].kNorm, 2);
        e->setBuffer(kbuf, (uint64_t)il * kRing * HKV * kHeadDim * 2, 3); e->setBuffer(vbuf, (uint64_t)il * kRing * HKV * kHeadDim * 2, 4);
        e->setBytes(&frontier, 4, 5); e->setBytes(&HKV, 4, 6); e->setBytes(&ring, 4, 7);
        e->dispatchThreadgroups(MTL::Size(2 * HKV, n, 1), MTL::Size(128, 1, 1)); (void)nog;
    }
    e->endEncoding(); cb->commit(); cb->waitUntilCompleted();
    if (dumpFc) {   // after the GPU finished: valid snapshots of fc's output and of c_t
        FILE* fa = fopen("/tmp/d_cat.bin", "wb"); fwrite(catH->contents(), 1, (size_t)n * kTargets * H * 2, fa); fclose(fa);
        FILE* fb = fopen("/tmp/d_fcf.bin", "wb"); fwrite(dFcf->contents(), 1, (size_t)n * H * 4, fb); fclose(fb);
        FILE* fc2 = fopen("/tmp/d_ch.bin", "wb"); fwrite(dCh->contents(), 1, (size_t)n * H * 2, fc2); fclose(fc2);
        FILE* fd = fopen("/tmp/d_fcrows.bin", "wb"); fwrite((const uint8_t*)gg.bytes(fc), 1, (size_t)8 * 100 * 144, fd); fclose(fd);
        FILE* fe = fopen("/tmp/d_ringk.bin", "wb"); fwrite((const uint8_t*)kbuf->contents(), 1, (size_t)n * 1024 * 2 * kLayers, fe); fclose(fe); }
    frontier += n;
}

// One masked forward at position pos: [anchor, MASK x 7]; writes 8 x vocab logits (from the target's shared PTQ1_0 head).
void Draft::forwardBlock(Engine& eng, int32_t anchor, uint32_t pos, MTL::Buffer* logitsOut) {
    int32_t tk[kBlock]; tk[0] = anchor; for (uint32_t i = 1; i < kBlock; i++) tk[i] = kMaskTok;
    forwardBlockToks(eng, tk, pos, logitsOut, kBlock);
}
void Draft::forwardBlockTruth(Engine& eng, const int* toks, uint32_t pos, MTL::Buffer* logitsOut, uint32_t T) {
    int32_t tk[kBlock]; for (uint32_t i = 0; i < T; i++) tk[i] = toks[i];
    forwardBlockToks(eng, tk, pos, logitsOut, T);
}
void Draft::forwardBlockToks(Engine& eng, const int32_t* toksIn, uint32_t pos, MTL::Buffer* logitsOut, uint32_t T) {
    const uint32_t H = kH, HKV = kKvHeads;
    { int32_t* tk = (int32_t*)tokbuf->contents(); for (uint32_t i = 0; i < T; i++) tk[i] = toksIn[i]; }
    MTL::CommandBuffer* cb = eng.gpu.queue->commandBuffer(); MTL::ComputeCommandEncoder* e = cb->computeCommandEncoder();
    auto mm = [&](uint64_t off, MTL::Buffer* a, MTL::Buffer* y, uint32_t K, uint32_t N, bool) {
        const int qt = qtypeOf(off);                               // the decoder comes from the tensor's own quant type
        static const char* names[5] = {"k_draft_mm_m8_n16_g4_q0", "k_draft_mm_m8_n16_g4_q1", "k_draft_mm_m8_n16_g4_q2", "k_draft_mm_m8_n16_g4_q3", "k_draft_mm_m8_n16_g4_q4"};
        e->setComputePipelineState(eng.gpu.pipe(names[qt]));
        e->setBuffer(wbuf, off, 0); e->setBuffer(a, 0, 1); e->setBuffer(y, 0, 2); e->setBytes(&K, 4, 3); e->setBytes(&N, 4, 4);
        e->dispatchThreadgroups(MTL::Size((N + 15) / 16, 1, 1), MTL::Size(128, 1, 1));
    };
    auto f2h = [&](MTL::Buffer* in, MTL::Buffer* out, uint32_t n) { e->setComputePipelineState(eng.gpu.pipe("k_f2h")); e->setBuffer(in, 0, 0); e->setBuffer(out, 0, 1); e->setBytes(&n, 4, 2); e->dispatchThreads(MTL::Size(n, 1, 1), MTL::Size(256, 1, 1)); };
    // operand for a rotated-basis matrix: FWHT(sign . x) with the drafter's own signs, else a plain fp16 cast
    auto toMat = [&](MTL::Buffer* src, MTL::Buffer* dst, uint32_t W) { if (rotated) fwhtIn(eng, e, src, dst, W, T); else f2h(src, dst, T * W); };
    auto rmsF = [&](MTL::Buffer* in, uint64_t wOff, MTL::Buffer* outf) { uint32_t W = H; e->setComputePipelineState(eng.gpu.pipe("k_rmsnorm_rows"));
        e->setBuffer(in, 0, 0); e->setBuffer(wbuf, wOff, 1); e->setBuffer(outf, 0, 2); e->setBytes(&W, 4, 3); e->dispatchThreadgroups(MTL::Size(T, 1, 1), MTL::Size(256, 1, 1)); };
    auto conv = [&](MTL::Buffer* in, MTL::Buffer* dynBuf, uint64_t baseOff, MTL::Buffer* out, uint32_t width, uint32_t side) {
        e->setComputePipelineState(eng.gpu.pipe("k_draft_conv")); e->setBuffer(in, 0, 0); e->setBuffer(dynBuf, 0, 1); e->setBuffer(wbuf, baseOff, 2); e->setBuffer(out, 0, 3);
        e->setBytes(&T, 4, 4); e->setBytes(&width, 4, 5); e->setBytes(&side, 4, 6); e->dispatchThreads(MTL::Size(width, T, 1), MTL::Size(128, 1, 1));
    };
    auto add = [&](MTL::Buffer* a, MTL::Buffer* b, MTL::Buffer* o, uint32_t n) { e->setComputePipelineState(eng.gpu.pipe("k_add3")); e->setBuffer(a, 0, 0); e->setBuffer(b, 0, 1); e->setBuffer(o, 0, 2); e->setBytes(&n, 4, 3); e->dispatchThreads(MTL::Size(n, 1, 1), MTL::Size(256, 1, 1)); };
    const bool dbg = getenv("BONSAI_DBG_STAGE") != nullptr;
    auto chk = [&](const char* name, MTL::Buffer* b, uint32_t nf) {
        if (!dbg) return;
        e->endEncoding(); cb->commit(); cb->waitUntilCompleted();
        const float* p = (const float*)b->contents(); double sa = 0; int nan = 0;
        for (uint32_t i = 0; i < nf; i++) { const float v = p[i]; if (!(v == v)) nan++; else sa += std::fabs(v); }
        fprintf(stderr, "  [%s] sum|.|=%.3f nan=%d first=%.4f %.4f\n", name, sa, nan, p[0], p[1]);
        cb = eng.gpu.queue->commandBuffer(); e = cb->computeCommandEncoder();
    };
    eng.xEmbed(e, tokbuf, T, xA);                                    // plain embedding rows [8][5120]
    chk("embed", xA, T * H);
    if (dbg) {   // per-row embedding magnitude: the mask rows all share one table row
        e->endEncoding(); cb->commit(); cb->waitUntilCompleted();
        const float* px = (const float*)xA->contents(); fprintf(stderr, "  [embed rows]");
        for (uint32_t r = 0; r < T; r++) { double s2 = 0; for (uint32_t i = 0; i < H; i++) s2 += std::fabs(px[r * H + i]); fprintf(stderr, " %.1f", s2); }
        fprintf(stderr, "\n"); cb = eng.gpu.queue->commandBuffer(); e = cb->computeCommandEncoder(); }
    for (uint32_t il = 0; il < kLayers; il++) {
        const LayerW& w = L[il];
        rmsF(xA, w.attnNorm, nrmf); if (il == 0) chk("rms attnNorm", nrmf, T * H); toMat(nrmf, nrmH, H);
        mm(w.attnCp, nrmH, dyn, H, kConvProj, false);                 // dynamic coefficients from the normalised input
        if (il == 0) chk("dyn (conv proj)", dyn, T * kConvProj);
        conv(nrmf, dyn, w.attnCb, tmpA, H, 0);                        // side 0 (input side)
        if (rotated) toMat(tmpA, nrmH, H);
        if (dumpLayer0 && il == 0) { const uint32_t n1 = T * H, n2 = T * kConvProj;
            e->setComputePipelineState(eng.gpu.pipe("k_copy_f32")); e->setBuffer(tmpA, 0, 0); e->setBuffer(convOut0, 0, 1); e->setBytes(&n1, 4, 2); e->dispatchThreads(MTL::Size(n1, 1, 1), MTL::Size(256, 1, 1));
            e->setBuffer(nrmf, 0, 0); e->setBuffer(convIn0, 0, 1); e->dispatchThreads(MTL::Size(n1, 1, 1), MTL::Size(256, 1, 1));
            e->setBuffer(dyn, 0, 0); e->setBuffer(convDyn0, 0, 1); e->setBytes(&n2, 4, 2); e->dispatchThreads(MTL::Size(n2, 1, 1), MTL::Size(256, 1, 1)); }
        if (il == 0) chk("conv0 attn", tmpA, T * H);
        if (!rotated) toMat(tmpA, nrmH, H);
        mm(w.q, nrmH, q, H, kHeads * kHeadDim, false);
        mm(w.k, nrmH, k, H, HKV * kHeadDim, false);
        mm(w.v, nrmH, v, H, HKV * kHeadDim, false);
        { const uint32_t hp = kHeads, hkv = HKV;
          e->setComputePipelineState(eng.gpu.pipe("k_draft_qk")); e->setBuffer(q, 0, 0); e->setBuffer(k, 0, 1); e->setBuffer(v, 0, 2);
          e->setBuffer(wbuf, w.qNorm, 3); e->setBuffer(wbuf, w.kNorm, 4); e->setBuffer(aout, 0, 5);
          e->setBuffer(kbuf, (uint64_t)il * kRing * HKV * kHeadDim * 2, 6); e->setBuffer(vbuf, (uint64_t)il * kRing * HKV * kHeadDim * 2, 7);
          e->setBytes(&pos, 4, 8); e->setBytes(&hp, 4, 9); e->setBytes(&hkv, 4, 10);
          e->dispatchThreadgroups(MTL::Size(kHeads + 2 * HKV, T, 1), MTL::Size(128, 1, 1)); }
        { const uint32_t win = kWindow, ring = kRing;
          e->setComputePipelineState(eng.gpu.pipe("k_draft_attn")); e->setBuffer(aout, 0, 0); e->setBuffer(kbuf, (uint64_t)il * kRing * HKV * kHeadDim * 2, 1);
          e->setBuffer(vbuf, (uint64_t)il * kRing * HKV * kHeadDim * 2, 2); e->setBuffer(tmpA, 0, 3);
          e->setBytes(&pos, 4, 4); e->setBytes(&T, 4, 5); const uint32_t hp = kHeads, hkv = HKV; e->setBytes(&hp, 4, 6); e->setBytes(&hkv, 4, 7);
          e->setBytes(&win, 4, 8); e->setBytes(&ring, 4, 9);
          e->dispatchThreadgroups(MTL::Size(kHeads, T, 1), MTL::Size(128, 1, 1)); }
        if (il == 0) chk("attn out", tmpA, T * kHeads * kHeadDim);
        if (dumpLayer0 && il == 0) { const uint32_t nc = T * kHeads * kHeadDim;   // GPU-side copies: the host cannot read before the command buffer runs
            e->setComputePipelineState(eng.gpu.pipe("k_copy_f32")); e->setBuffer(tmpA, 0, 0); e->setBuffer(attnOut0, 0, 1); e->setBytes(&nc, 4, 2); e->dispatchThreads(MTL::Size(nc, 1, 1), MTL::Size(256, 1, 1));
            e->setBuffer(aout, 0, 0); e->setBuffer(qOut0, 0, 1); e->dispatchThreads(MTL::Size(nc, 1, 1), MTL::Size(256, 1, 1)); }
        toMat(tmpA, aoutH, kHeads * kHeadDim);
        mm(w.o, aoutH, tmpA, kHeads * kHeadDim, H, false);
        if (il == 0) chk("conv1 in (wo out)", tmpA, T * H);            // attention output projection
        conv(tmpA, dyn, w.attnCb, xB, H, 1);                          // side 1 (output side), same delta
        add(xB, xA, tmpA, T * H);                                     // ffn_inp = cur + inpL
        rmsF(tmpA, w.ffnNorm, nrmf); if (il == 0) chk("ffn norm", nrmf, T * H); toMat(nrmf, nrmH, H);
        mm(w.ffnCp, nrmH, dyn, H, kConvProj, false); if (il == 0) chk("ffn dyn", dyn, T * kConvProj);
        conv(nrmf, dyn, w.ffnCb, xB, H, 0); if (il == 0) chk("ffn conv0", xB, T * H);
        toMat(xB, nrmH, H);
        mm(w.gate, nrmH, fout, H, kFfn, false); if (il == 0) chk("gate out", fout, T * kFfn);
        mm(w.up, nrmH, gU, H, kFfn, false);
        { uint32_t n = T * kFfn; e->setComputePipelineState(eng.gpu.pipe("k_swiglu")); e->setBuffer(fout, 0, 0); e->setBuffer(gU, 0, 1); e->setBuffer(fout2, 0, 2); e->setBytes(&n, 4, 3); e->dispatchThreads(MTL::Size(n, 1, 1), MTL::Size(256, 1, 1)); }
        toMat(fout2, ffnH, kFfn);
        (void)0;
        if (il == 0) chk("swiglu", fout2, T * kFfn);
        mm(w.down, ffnH, xB, kFfn, H, false);
        if (il == 0) chk("down out", xB, T * H);
        conv(xB, dyn, w.ffnCb, tmpB, H, 1);                           // output side into its own buffer: tmpA still holds ffn_inp
        add(tmpB, tmpA, xB, T * H);                                   // x = conv(ffn_out) + ffn_inp
        // swap the two residual buffers for the next layer
        MTL::Buffer* t = xA; xA = xB; xB = t;
        chk("x after layer", xA, T * H);
    }
    // final RMSNorm with the draft's own output_norm, then the Hadamard basis of the shared PTQ1_0 head (sign vector only, no trunk gamma)
    rmsF(xA, outNorm, nrmf);
    { uint32_t W = H, mode = 0;
      e->setComputePipelineState(eng.gpu.pipe("k_rms_fwht_b")); e->setBuffer(nrmf, 0, 0); e->setBuffer(nrmf, 0, 1); e->setBuffer(eng.sign5120(), 0, 2); e->setBuffer(probs, 0, 3);
      e->setBytes(&W, 4, 4); e->setBytes(&mode, 4, 5); e->dispatchThreadgroups(MTL::Size(W / 1024, T, 1), MTL::Size(256, 1, 1)); }
    eng.xWideGemv(e, eng.offOutW(), probs, logitsOut, H, (uint32_t)kVocab, T);
    e->endEncoding(); cb->commit(); cb->waitUntilCompleted();
}
