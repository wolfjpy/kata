#include "spec.h"
#include <chrono>

using Clock = std::chrono::steady_clock;
static double since(Clock::time_point t);
static double since(Clock::time_point t) { return std::chrono::duration<double>(Clock::now() - t).count(); }

SpecResult specGenerate(Engine& eng, uint32_t pos, uint32_t ctx, int maxTokens, const SamplerParams& sp, std::mt19937_64& rng, const std::function<bool(int)>& isEog,
                        const std::function<void(int)>& emit, SpecStats* stats) {
    const int H = 5120; const int V = (int)bonsai::kVocab;
    SpecStats dummy; SpecStats& st = stats ? *stats : dummy;
    auto sample = [&](const float* lg, int greedyTok) { auto t0 = Clock::now(); int r = sp.temp <= 0.0f ? greedyTok : sampleToken(lg, V, sp, rng); st.sampleSec += since(t0); return r; };
    uint32_t n = pos;                                    // position of `next`
    std::vector<float> hPrev(eng.hPrev(), eng.hPrev() + H);
    std::vector<float> pendH(H); int pendTok = -1; bool hasPend = false;   // MTP row (pendTok, pendH) at position n-1 that has not been computed yet
    std::vector<float> h0(H), h1(H);
    int next; { int g = 0; const float* lg = eng.logits(); for (int i = 1; i < V; i++) if (lg[i] > lg[g]) g = i; next = sample(lg, g); }
    int emitted = 0; SpecResult res;
    while (true) {
        emit(next); emitted++; st.tokens++;
        if (isEog(next) || emitted >= maxTokens || n + 3 >= ctx) { res.unfed = next; if (hasPend) { eng.mtpRows(1, &pendTok, pendH.data(), n - 1, false); hasPend = false; } eng.setHPrev(hPrev.data()); res.pos = n; return res; }
        // ---- draft: bring the MTP cache up to position n and predict the token after `next`
        auto t0 = Clock::now();
        int rt[2]; const float* rh[2]; uint32_t R = 0;
        if (hasPend) { rt[R] = pendTok; rh[R] = pendH.data(); R++; }
        rt[R] = next; rh[R] = hPrev.data(); R++;
        std::vector<float> hin((size_t)R * H); for (uint32_t r = 0; r < R; r++) memcpy(hin.data() + (size_t)r * H, rh[r], H * 4);
        const int d = eng.mtpRows(R, rt, hin.data(), n - (R - 1), true);
        hasPend = false; st.draftSec += since(t0);
        // ---- verify [next, d] at positions n, n+1
        t0 = Clock::now(); int vt[2] = {next, d}, am[2]; eng.verify2(vt, n, am); st.verifySec += since(t0);
        st.cycles++;
        eng.verifyHidden(0, h0.data()); eng.verifyHidden(1, h1.data());
        const int s0 = sample(eng.verifyLogits(0), am[0]);
        if (s0 == d && !isEog(d)) {
            eng.commit2(true); st.accepted++;
            emit(d); emitted++; st.tokens++;
            if (emitted >= maxTokens) { res.unfed = -1; pendTok = d; memcpy(pendH.data(), h0.data(), H * 4); hasPend = true; n += 2; eng.mtpRows(1, &pendTok, pendH.data(), n - 1, false); hasPend = false; eng.setHPrev(h1.data()); res.pos = n; return res; }
            const int s1 = sample(eng.verifyLogits(1), am[1]);
            pendTok = d; memcpy(pendH.data(), h0.data(), H * 4); hasPend = true;
            hPrev = h1; next = s1; n += 2;
        } else {
            eng.commit2(false);
            hPrev = h0; next = s0; n += 1;
            if (s0 == d) { /* d is an end token: s0 == d is emitted by the loop top */ }
        }
    }
}

// ------------------------------------------------------------------------------------------------------------------------------
// MTP chain drafting with adaptive depth. Depth d (1 or 3) drafts d tokens by chaining the MTP head (each step's output hidden feeds the next),
// then verifies them with T = d+1 rows through the ALU verification path. The policy keeps an online estimate of the per-position acceptance
// and of the round cost for each depth and picks the depth with the best expected tokens per millisecond.
namespace {
struct DepthStat { double p[3] = {0.6, 0.5, 0.4}; double ms = 70.0; int n = 0; };
}
SpecResult specGenerateChain(Engine& eng, uint32_t pos, uint32_t ctx, int maxTokens, const SamplerParams& sp, std::mt19937_64& rng, const std::function<bool(int)>& isEog,
                             const std::function<void(int)>& emit, SpecStats* stats, bool adaptive) {
    SpecStats dummy; SpecStats& st = stats ? *stats : dummy;
    const int H = 5120, V = (int)bonsai::kVocab;
    auto sampleFrom = [&](const float* lg, int row) { const float* l = lg + (size_t)row * V; int g = 0; for (int i = 1; i < V; i++) if (l[i] > l[g]) g = i;
        if (sp.temp <= 0.0f) return g; return sampleToken(l, V, sp, rng); };
    uint32_t n = pos;
    std::vector<float> hPrev(eng.hPrev(), eng.hPrev() + H), hStep(H);
    int next; { int g = 0; const float* lg = eng.logits(); for (int i = 1; i < V; i++) if (lg[i] > lg[g]) g = i; next = sampleFrom(lg, 0); (void)g; }
    DepthStat ds[2];   // [0] = depth 1 (T=2), [1] = depth 3 (T=4)
    ds[0].ms = 67.0; ds[1].ms = 113.0;   // measured round costs, so the policy starts with honest numbers
    int emitted = 0; SpecResult res; res.pos = n; res.unfed = -1;
    std::vector<int> win(5);
    bool firstRound = true;
    while (true) {
        if (firstRound) { emit(next); emitted++; st.tokens++; firstRound = false; }   // the anchor is emitted once, before the first round
        if (isEog(next) || emitted >= maxTokens || n + 5 >= ctx) { res.unfed = next; eng.setHPrev(hPrev.data()); res.pos = n; return res; }
        // ---- pick the depth
        int slot = getenv("BONSAI_CHAIN_D1") ? 0 : 1;
        const bool force3 = getenv("BONSAI_CHAIN_D3") != nullptr, force1 = getenv("BONSAI_CHAIN_D1") != nullptr;
        if (adaptive && !force3 && !force1 && ds[0].n > 2 && ds[1].n > 2) {
            auto e1 = 1.0 + ds[0].p[0];
            auto e4 = 1.0 + ds[1].p[0] * (1.0 + ds[1].p[1] * (1.0 + ds[1].p[2]));
            slot = (e4 / ds[1].ms > e1 / ds[0].ms) ? 1 : 0;
        }
        const uint32_t d = slot ? 3u : 1u, T = d + 1u;
        eng.aluT = T;   // 2 or 4 token rows on the ALU verification path
        const bool fastDepth1 = (d == 1u);
        auto t0 = Clock::now();
        win[0] = next;
        for (uint32_t j = 0; j < d; j++) {
            const int tok = win[(size_t)j];
            win[(size_t)j + 1] = eng.mtpRows(1, &tok, j == 0 ? hPrev.data() : hStep.data(), n + j, true);
            eng.mtpHidden(hStep.data());
        }
        st.draftSec += since(t0);
        int am[8]; t0 = Clock::now();
        if (fastDepth1) { int vt[2] = {win[0], win[0]}; eng.verify2(vt, n, am); }
        else eng.verifyWide(win.data(), T, n, am);
        st.verifySec += since(t0);
        st.cycles++;
        // exact speculative sampling: sample the target distribution per row and accept a draft only when the sample equals it
        int matched = 0; int nextTok = -1;
        const float* rowLg = fastDepth1 ? eng.logits() : nullptr;
        for (int j = 0; j < (int)d; j++) {
            const int s_j = sampleFrom(fastDepth1 ? rowLg : eng.verifyLogits(j), j);
            if (s_j == win[(size_t)j + 1]) matched++; else { nextTok = s_j; break; }
        }
        if (nextTok < 0) nextTok = fastDepth1 ? sampleFrom(eng.logits(), 0) : sampleFrom(eng.verifyLogits((int)d), (int)d);
        const uint32_t F = (uint32_t)matched + 1;
        if (getenv("BONSAI_DBG_CHAIN")) { fprintf(stderr, "  [r] n=%u win", n); for (uint32_t q = 0; q <= d; q++) fprintf(stderr, " %d", win[q]);
            fprintf(stderr, " | am"); for (uint32_t q = 0; q < T; q++) fprintf(stderr, " %d", am[q]); fprintf(stderr, " | matched %d next %d F %u\n", matched, nextTok, F); }
        if (fastDepth1) eng.commit2(false); else eng.commitWide(F);
        DepthStat& D = ds[slot]; D.n++;
        for (int j = 0; j < (int)d; j++) if (j <= matched) D.p[j] = 0.8 * D.p[j] + 0.2 * (j < matched ? 1.0 : 0.0);
        D.ms = 0.8 * D.ms + 0.2 * (1e3 * (since(t0) + 0.0));
        st.accepted += (uint64_t)matched;
        for (uint32_t j = 1; j < F && emitted < maxTokens; j++) { emit(win[(size_t)j]); emitted++; st.tokens++; }   // the accepted drafts
        if (emitted < maxTokens) { emit(nextTok); emitted++; st.tokens++; }                                          // the correction / bonus token
        { std::vector<float> hh(H); if (fastDepth1) memcpy(hh.data(), eng.hidden(), (size_t)H * 4); else eng.verifyHidden(F - 1, hh.data());
          if (getenv("BONSAI_DBG_CHAIN")) { double s1 = 0, s0 = 0; for (int q = 0; q < H; q++) { s1 += std::fabs(hh[(size_t)q]); s0 += std::fabs(hPrev[(size_t)q]); }
              fprintf(stderr, "  [h] src=%s row=%u new sum|h|=%.1f prev sum|h|=%.1f first %.4f vs %.4f\n", fastDepth1 ? "bufH" : "pHh", fastDepth1 ? 0u : F - 1, s1, s0, hh[0], hPrev[0]); }
          hPrev = hh; }
        next = nextTok; n += F;
        if (getenv("BONSAI_DBG_CHAIN") && ds[1].n > 0) fprintf(stderr, "  [chain] d3: p1 %.3f p2 %.3f p3 %.3f | d1: p1 %.3f (n %d/%d)\n", ds[1].p[0], ds[1].p[1], ds[1].p[2], ds[0].p[0], ds[1].n, ds[0].n);
        if (emitted >= maxTokens) { res.unfed = -1; res.pos = n; return res; }
    }
}
