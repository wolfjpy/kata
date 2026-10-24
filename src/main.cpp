// bonsai: decode engine CLI (milestone M2: token-id interface with reference comparison).
#include "engine.h"
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
int main(int argc, char** argv) {
    std::string model, metallib = "build/bonsai.metallib", ids, refDir, refLogits; int gen = 0; uint32_t ctx = 2048;
    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        if (a == "--model") model = argv[++i]; else if (a == "--metallib") metallib = argv[++i]; else if (a == "--ids") ids = argv[++i]; else if (a == "--gen") gen = atoi(argv[++i]);
        else if (a == "--ref-dir") refDir = argv[++i]; else if (a == "--ref-logits") refLogits = argv[++i]; else if (a == "--ctx") ctx = atoi(argv[++i]);
    }
    if (model.empty() || ids.empty()) { fprintf(stderr, "usage: bonsai --model M.gguf --ids 1,2,3 [--gen N] [--ref-dir D] [--ref-logits F] [--ctx N]\n"); return 1; }
    std::vector<int> toks; { size_t p = 0; while (p < ids.size()) { size_t q = ids.find(',', p); if (q == std::string::npos) q = ids.size(); toks.push_back(atoi(ids.substr(p, q - p).c_str())); p = q + 1; } }
    const uint32_t dbg = refDir.empty() ? 0 : (uint32_t)toks.size();
    Engine eng; if (!eng.load(model, metallib, ctx, dbg)) return 1;
    std::vector<std::vector<float>> myLogits;
    auto t0 = std::chrono::steady_clock::now(); int next = 0;
    for (size_t i = 0; i < toks.size(); i++) {
        next = eng.step(toks[i], (uint32_t)i, dbg ? (int)i : -1);
        if (!refLogits.empty()) myLogits.emplace_back(eng.logits(), eng.logits() + bonsai::kVocab);
    }
    double dt = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    printf("prompt of %zu tokens processed in %.2f s; next token = %d\n", toks.size(), dt, next);
    if (!refDir.empty()) {
        printf("%-4s %-6s %-14s %-14s %-14s\n", "layer", "type", "block-out relL2", "residual relL2", "residual maxabs");
        for (uint32_t il = 0; il < bonsai::kLayers; il++) {
            const bool at = bonsai::layerIsAttention(il); char nm[64];
            snprintf(nm, sizeof nm, "%s/l_out-%u.f32", refDir.c_str(), il); std::vector<float> ref = readF32(nm);
            snprintf(nm, sizeof nm, at ? "%s/attn_output-%u.f32" : "%s/linear_attn_out-%u.f32", refDir.c_str(), il); std::vector<float> refB = readF32(nm);
            double worstR = 0, worstM = 0, worstB = 0;
            for (size_t t = 0; t < toks.size(); t++) {
                const float* a = eng.dbgResidual(il, (int)t); double n2 = 0, d2 = 0, mx = 0;
                for (uint32_t k = 0; k < bonsai::kHidden; k++) { double r = ref.empty() ? 0 : ref[t * bonsai::kHidden + k]; double d = a[k] - r; n2 += r * r; d2 += d * d; mx = std::max(mx, std::fabs(d)); }
                worstR = std::max(worstR, std::sqrt(d2 / (n2 + 1e-30))); worstM = std::max(worstM, mx);
                if (!refB.empty()) { const float* b = eng.dbgBlock(il, (int)t); double bn = 0, bd = 0; for (uint32_t k = 0; k < bonsai::kHidden; k++) { double r = refB[t * bonsai::kHidden + k]; double d = b[k] - r; bn += r * r; bd += d * d; } worstB = std::max(worstB, std::sqrt(bd / (bn + 1e-30))); }
            }
            printf("%-4u %-6s %-14.3e %-14.3e %-14.3e\n", il, at ? "attn" : "gdn", worstB, worstR, worstM);
        }
    }
    if (!refLogits.empty()) {
        std::vector<float> ref = readF32(refLogits); const size_t nv = bonsai::kVocab; int agree = 0; double maxd = 0;
        for (size_t t = 0; t < toks.size(); t++) {
            const float* r = ref.data() + t * nv; const std::vector<float>& m = myLogits[t]; size_t br = 0, bm = 0;
            for (size_t k = 0; k < nv; k++) { if (r[k] > r[br]) br = k; if (m[k] > m[bm]) bm = k; maxd = std::max(maxd, (double)std::fabs(m[k] - r[k])); }
            agree += (br == bm); printf("pos %zu: ref top1 %zu (%.4f) mine top1 %zu (%.4f)\n", t, br, r[br], bm, m[bm]);
        }
        printf("top-1 agreement %d/%zu, max |logit diff| %.4f\n", agree, toks.size(), maxd);
    }
    for (int g = 0; g < gen; g++) { printf("gen[%d] = %d\n", g, next); next = eng.step(next, (uint32_t)(toks.size() + g)); }
    return 0;
}
