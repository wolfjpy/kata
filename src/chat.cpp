#include "chat.h"
#include <algorithm>
#include <cmath>
#include <numeric>

static std::string trim(const std::string& s) {
    size_t b = 0, e = s.size(); auto ws = [](unsigned char c) { return c == ' ' || c == '\t' || c == '\n' || c == '\r' || c == '\v' || c == '\f'; };
    while (b < e && ws(s[b])) b++; while (e > b && ws(s[e - 1])) e--; return s.substr(b, e - b);
}
std::string reasoningInstructions(const ChatOpts& o) {
    if (!o.thinking) return "";
    if (o.effort == "xhigh") return "Reasoning effort is set to xhigh. Please think carefully through the task, validate key assumptions, consider plausible alternatives, and prioritize correctness, consistency, and clarity in the final answer.";
    if (o.effort == "low") return "Reasoning effort is set to low. Keep your thinking brief and focused, moving directly to the conclusion without unnecessary elaboration.";
    return "";
}
std::string renderChat(const std::vector<Msg>& msgs, const ChatOpts& o) {
    std::string out; const std::string ri = reasoningInstructions(o);
    if (!msgs.empty() && msgs[0].role == "system") {
        std::string c = trim(msgs[0].content);
        if (!c.empty()) out += "<|im_start|>system\n" + (ri.empty() ? "" : ri + "\n\n") + c + "<|im_end|>\n";
        else if (!ri.empty()) out += "<|im_start|>system\n" + ri + "<|im_end|>\n";
    } else if (!ri.empty()) out += "<|im_start|>system\n" + ri + "<|im_end|>\n";
    for (size_t i = 0; i < msgs.size(); i++) {
        const Msg& m = msgs[i]; std::string c = trim(m.content);
        if (m.role == "system") continue;
        else if (m.role == "user") out += "<|im_start|>user\n" + c + "<|im_end|>\n";
        else if (m.role == "assistant") out += "<|im_start|>assistant\n<think>\n" + trim(m.reasoning) + "\n</think>\n\n" + c + "<|im_end|>\n";
    }
    if (o.addGenerationPrompt) { out += "<|im_start|>assistant\n"; out += o.thinking ? "<think>\n" : "<think>\n\n</think>\n\n"; }
    return out;
}

// llama.cpp's default chain: top-k -> top-p -> min-p -> temperature -> sample
// The distribution is returned unnormalised (w, with z = sum) so that sampleToken keeps its exact arithmetic.
SparseDist samplerDistribution(const float* logits, int n, const SamplerParams& p) {
    const int K = std::min(p.topK > 0 ? p.topK : n, n);
    std::vector<int> idx;
    if (K <= 64) {   // one pass over the logits keeping the K best (ties: the lower index wins, as in the full sort); about 0.2 ms for the 248K vocabulary
        idx.reserve((size_t)K + 1); float thr = -INFINITY;   // thr = the K-th best logit once K are held; anything not above it cannot enter
        for (int i = 0; i < n; i++) {
            const float v = logits[i];
            if (!(v > thr)) { if ((int)idx.size() == K) continue; }
            size_t pos = idx.size(); while (pos > 0 && logits[idx[pos - 1]] < v) pos--;
            idx.insert(idx.begin() + (long)pos, i); if ((int)idx.size() > K) idx.pop_back();
            if ((int)idx.size() == K) thr = logits[idx.back()];
        }
    } else {
        idx.resize((size_t)n); std::iota(idx.begin(), idx.end(), 0);
        std::partial_sort(idx.begin(), idx.begin() + K, idx.end(), [&](int a, int b) { return logits[a] > logits[b] || (logits[a] == logits[b] && a < b); });
        idx.resize((size_t)K);
    }
    std::vector<double> pr(K); double mx = logits[idx[0]], z = 0; for (int i = 0; i < K; i++) { pr[i] = std::exp((double)logits[idx[i]] - mx); z += pr[i]; } for (auto& v : pr) v /= z;
    int keep = K;
    if (p.topP < 1.0f) { double cum = 0; keep = 0; for (int i = 0; i < K; i++) { cum += pr[i]; keep = i + 1; if (cum >= p.topP) break; } }
    if (p.minP > 0.0f) { int k2 = 0; for (int i = 0; i < keep; i++) if (pr[i] >= (double)p.minP * pr[0]) k2 = i + 1; else break; keep = std::max(1, k2); }
    SparseDist d; d.ids.assign(idx.begin(), idx.begin() + keep); d.w.resize((size_t)keep); d.z = 0;
    for (int i = 0; i < keep; i++) { d.w[(size_t)i] = std::exp(((double)logits[idx[(size_t)i]] - mx) / p.temp); d.z += d.w[(size_t)i]; }
    return d;
}
double SparseDist::prob(int token) const { for (size_t i = 0; i < ids.size(); i++) if (ids[i] == token) return w[i] / z; return 0.0; }
int sampleFromDist(const SparseDist& d, std::mt19937_64& rng) {
    std::uniform_real_distribution<double> U(0.0, d.z); double r = U(rng), acc = 0;
    for (size_t i = 0; i < d.ids.size(); i++) { acc += d.w[i]; if (r <= acc) return d.ids[i]; }
    return d.ids.back();
}
int sampleToken(const float* logits, int n, const SamplerParams& p, std::mt19937_64& rng) {
    if (p.temp <= 0.0f) { int b = 0; for (int i = 1; i < n; i++) if (logits[i] > logits[b]) b = i; return b; }
    return sampleFromDist(samplerDistribution(logits, n, p), rng);
}
