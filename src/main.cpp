// bonsai: decode engine CLI (token-id interface with reference comparison).
#include "engine.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

static std::vector<float> readF32(const std::string& path) {
    FILE* f = fopen(path.c_str(), "rb"); if (!f) return {};
    fseek(f, 0, SEEK_END); long n = ftell(f); fseek(f, 0, SEEK_SET); std::vector<float> v(n / 4); fread(v.data(), 4, v.size(), f); fclose(f); return v;
}
// KL(ref || mine) per position and top-1 agreement; `ref` and `mine` are [n][nv]
static void klStats(const char* label, const float* ref, const float* mine, size_t n, size_t nv) {
    double sumKl = 0, maxKl = 0; size_t agree = 0, bad = 0;
    for (size_t t = 0; t < n; t++) {
        const float* r = ref + t * nv; const float* m = mine + t * nv; double mr = -1e30, mm = -1e30; size_t br = 0, bm = 0;
        for (size_t k = 0; k < nv; k++) { if (r[k] > mr) { mr = r[k]; br = k; } if (m[k] > mm) { mm = m[k]; bm = k; } }
        double zr = 0, zm = 0; for (size_t k = 0; k < nv; k++) { zr += std::exp((double)r[k] - mr); zm += std::exp((double)m[k] - mm); }
        double lzr = std::log(zr) + mr, lzm = std::log(zm) + mm, kl = 0;
        for (size_t k = 0; k < nv; k++) { double lr = (double)r[k] - lzr; double p = std::exp(lr); if (p > 1e-12) kl += p * (lr - ((double)m[k] - lzm)); }
        if (!(kl == kl)) { bad++; continue; }
        sumKl += kl; maxKl = std::max(maxKl, kl); agree += (br == bm);
    }
    printf("%-34s positions %zu | mean KL %.3e | max KL %.3e | top-1 agreement %zu/%zu (%.2f%%)%s\n", label, n, sumKl / std::max<size_t>(1, n - bad), maxKl, agree, n, 100.0 * agree / n, bad ? " (some NaN)" : "");
}
#include "cli.h"
int main(int argc, char** argv) {
    if (argc > 1 && std::string(argv[1]) == "tokenize") return cmdTokenize(argc, argv);
    if (argc > 1 && std::string(argv[1]) == "mtp-accept") return cmdMtpAccept(argc, argv);
    if (argc > 1 && std::string(argv[1]) == "cache") return cmdCache(argc, argv);
    if (argc > 1 && std::string(argv[1]) == "kl-eval") return cmdKlEval(argc, argv);
    if (argc > 1 && std::string(argv[1]) == "verify-check") return cmdVerifyCheck(argc, argv);
    if (argc > 1 && std::string(argv[1]) == "mtp-eval") return cmdMtpEval(argc, argv);
    if (argc > 1 && std::string(argv[1]) == "run") return cmdRun(argc, argv, false);
    if (argc > 1 && std::string(argv[1]) == "chat") return cmdRun(argc, argv, true);
    std::string model, metallib = "build/bonsai.metallib", ids, refDir, refLogits, dumpLogits, cmpLogits, skipArg; int gen = 0, verifyN = 0, repeat = 1, nlogits = 0; uint32_t ctx = 2048; bool usePrefill = false, int8 = false; std::string i8spec;
    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        if (a == "--model") model = argv[++i]; else if (a == "--metallib") metallib = argv[++i]; else if (a == "--ids") ids = argv[++i]; else if (a == "--gen") gen = atoi(argv[++i]); else if (a == "--verify") verifyN = atoi(argv[++i]);
        else if (a == "--ref-dir") refDir = argv[++i]; else if (a == "--ref-logits") refLogits = argv[++i]; else if (a == "--ctx") ctx = atoi(argv[++i]); else if (a == "--skip") skipArg = argv[++i];
        else if (a == "--prefill") usePrefill = true; else if (a == "--repeat") repeat = atoi(argv[++i]); else if (a == "--int8") int8 = true; else if (a == "--i8") i8spec = argv[++i]; else if (a == "--nlogits") nlogits = atoi(argv[++i]);
        else if (a == "--dump-logits") dumpLogits = argv[++i]; else if (a == "--cmp-logits") cmpLogits = argv[++i];
    }
    if (model.empty() || ids.empty()) { fprintf(stderr, "usage: bonsai --model M.gguf --ids 1,2,3 [--prefill] [--int8] [--nlogits N] [--gen N] [--ref-dir D] [--ref-logits F] [--dump-logits F] [--cmp-logits F] [--ctx N] [--skip a,b]\n"); return 1; }
    std::vector<int> toks; { size_t p = 0; while (p < ids.size()) { size_t q = ids.find(',', p); if (q == std::string::npos) q = ids.size(); toks.push_back(atoi(ids.substr(p, q - p).c_str())); p = q + 1; } }
    const size_t ntok = toks.size(), nv = bonsai::kVocab;
    const bool wantCompare = !refLogits.empty() || !dumpLogits.empty() || !cmpLogits.empty();
    const uint32_t nl = wantCompare ? (nlogits > 0 ? (uint32_t)std::min<size_t>(nlogits, std::min<size_t>(ntok, 128)) : (uint32_t)std::min<size_t>(ntok, 128)) : 1;   // positions we keep logits for: the last nl
    const uint32_t dbg = refDir.empty() ? 0 : (uint32_t)ntok;
    Engine eng; if (!eng.load(model, metallib, ctx, dbg)) return 1; eng.setI8(i8spec.empty() ? (int8 ? "all" : "") : i8spec); int8 = eng.int8Mode;
    { size_t p = 0; while (p < skipArg.size()) { size_t q = skipArg.find(',', p); if (q == std::string::npos) q = skipArg.size(); eng.skip.push_back(skipArg.substr(p, q - p)); p = q + 1; } }
    std::vector<float> mine((size_t)nl * nv, 0.0f);   // logits of the last nl prompt positions
    int next = 0;
    if (usePrefill) {
        for (int r = 1; r < repeat; r++) { eng.resetState(); auto s0 = std::chrono::steady_clock::now(); eng.prefill(toks.data(), (uint32_t)ntok, 0, -1, true, 1); double d = std::chrono::duration<double>(std::chrono::steady_clock::now() - s0).count(); printf("warm-up prefill %d: %.3f s = %.1f tok/s\n", r, d, ntok / d); }
        if (repeat > 1) eng.resetState();
        auto s0 = std::chrono::steady_clock::now();
        next = eng.prefill(toks.data(), (uint32_t)ntok, 0, dbg ? 0 : -1, true, nl);
        double d = std::chrono::duration<double>(std::chrono::steady_clock::now() - s0).count();
        printf("prefill (%s) of %zu tokens: %.3f s = %.1f tok/s; next token = %d\n", int8 ? "int8 path" : "fp16 exact path", ntok, d, ntok / d, next);
        if (wantCompare) memcpy(mine.data(), eng.logitsAll(), (size_t)nl * nv * 4);
    } else {
        auto t0 = std::chrono::steady_clock::now();
        for (size_t i = 0; i < ntok; i++) {
            const bool need = wantCompare ? (i + nl >= ntok) : (i + 1 == ntok);
            next = eng.step(toks[i], (uint32_t)i, dbg ? (int)i : -1, need);
            if (wantCompare && need) memcpy(mine.data() + (i + nl - ntok) * nv, eng.logits(), nv * 4);
        }
        printf("stepwise prompt of %zu tokens processed in %.2f s; next token = %d\n", ntok, std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count(), next);
    }
    if (!refDir.empty()) {
        printf("%-4s %-6s %-14s %-14s %-14s\n", "layer", "type", "block-out relL2", "residual relL2", "residual maxabs");
        for (uint32_t il = 0; il < bonsai::kLayers; il++) {
            const bool at = bonsai::layerIsAttention(il); char nm[64];
            snprintf(nm, sizeof nm, "%s/l_out-%u.f32", refDir.c_str(), il); std::vector<float> ref = readF32(nm);
            snprintf(nm, sizeof nm, at ? "%s/attn_output-%u.f32" : "%s/linear_attn_out-%u.f32", refDir.c_str(), il); std::vector<float> refB = readF32(nm);
            double worstR = 0, worstM = 0, worstB = 0;
            for (size_t t = 0; t < ntok; t++) {
                const float* a = eng.dbgResidual(il, (int)t); double n2 = 0, d2 = 0, mx = 0;
                for (uint32_t k = 0; k < bonsai::kHidden; k++) { double r = ref.empty() ? 0 : ref[t * bonsai::kHidden + k]; double dd = a[k] - r; n2 += r * r; d2 += dd * dd; mx = std::max(mx, std::fabs(dd)); }
                worstR = std::max(worstR, std::sqrt(d2 / (n2 + 1e-30))); worstM = std::max(worstM, mx);
                if (!refB.empty()) { const float* b = eng.dbgBlock(il, (int)t); double bn = 0, bd = 0; for (uint32_t k = 0; k < bonsai::kHidden; k++) { double r = refB[t * bonsai::kHidden + k]; double dd = b[k] - r; bn += r * r; bd += dd * dd; } worstB = std::max(worstB, std::sqrt(bd / (bn + 1e-30))); }
            }
            printf("%-4u %-6s %-14.3e %-14.3e %-14.3e\n", il, at ? "attn" : "gdn", worstB, worstR, worstM);
        }
    }
    if (!dumpLogits.empty()) { FILE* f = fopen(dumpLogits.c_str(), "wb"); fwrite(mine.data(), 4, mine.size(), f); fclose(f); printf("wrote %s (%u positions)\n", dumpLogits.c_str(), nl); }
    if (!refLogits.empty()) { std::vector<float> ref = readF32(refLogits); size_t nref = ref.size() / nv; if (nref >= nl) klStats("vs llama.cpp reference", ref.data() + (nref - nl) * nv, mine.data(), nl, nv); }
    if (!cmpLogits.empty()) { std::vector<float> ref = readF32(cmpLogits); size_t nref = ref.size() / nv; if (nref >= nl) klStats("vs saved logits", ref.data() + (nref - nl) * nv, mine.data(), nl, nv); }
    if (gen > 0) {
        auto g0 = std::chrono::steady_clock::now();
        for (int g = 0; g < gen; g++) { printf("gen[%d] = %d\n", g, next); next = eng.step(next, (uint32_t)(ntok + g)); }
        double gdt = std::chrono::duration<double>(std::chrono::steady_clock::now() - g0).count(); printf("decode: %d tokens in %.3f s = %.2f tok/s (%.1f ms/token)\n", gen, gdt, gen / gdt, gdt / gen * 1e3);
    }
    if (verifyN > 0) {   // timing of the 2-token verification pass (state is not advanced meaningfully; results only valid without --skip)
        const int vt[2] = {next, next}; int am[2];
        eng.verify2(vt, (uint32_t)ntok, am);
        auto v0 = std::chrono::steady_clock::now();
        for (int i = 0; i < verifyN; i++) { eng.verify2(vt, (uint32_t)ntok, am); eng.commit2(false); }
        double vdt = std::chrono::duration<double>(std::chrono::steady_clock::now() - v0).count(); printf("verify2: %d passes in %.3f s = %.2f ms/pass\n", verifyN, vdt, vdt / verifyN * 1e3);
    }
    return 0;
}
