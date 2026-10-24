#include "spec.h"
#include <chrono>

using Clock = std::chrono::steady_clock;
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
