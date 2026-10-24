#include "cli.h"
#include "gguf.h"
#include "tokenizer.h"
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

static std::string readFile(const char* p) { FILE* f = fopen(p, "rb"); if (!f) { perror(p); exit(1); } std::string s; char buf[65536]; size_t n; while ((n = fread(buf, 1, sizeof buf, f)) > 0) s.append(buf, n); fclose(f); return s; }

int cmdTokenize(int argc, char** argv) {
    std::string model, text; bool haveText = false, noSpecial = false, show = false;
    for (int i = 2; i < argc; i++) { std::string a = argv[i]; if (a == "--model") model = argv[++i]; else if (a == "--text") { text = argv[++i]; haveText = true; } else if (a == "--file") { text = readFile(argv[++i]); haveText = true; } else if (a == "--no-special") noSpecial = true; else if (a == "--show") show = true; }
    if (model.empty() || !haveText) { fprintf(stderr, "usage: bonsai tokenize --model M.gguf (--text T | --file F) [--no-special]\n"); return 1; }
    Gguf gg; if (!gg.open(model.c_str())) return 1; Tokenizer tk; if (!tk.load(gg)) return 1;
    std::vector<int32_t> ids = tk.encode(text, !noSpecial);
    if (show) { for (size_t i = 0; i < ids.size(); i++) { std::string p = tk.piece(ids[i]); printf("%zu: %d \"", i, ids[i]); for (unsigned char c : p) { if (c == 10) printf("\\n"); else if (c == 13) printf("\\r"); else if (c == 9) printf("\\t"); else if (c < 32 || c >= 127) printf("\\x%02x", c); else putchar(c); } printf("\"\n"); } return 0; }
    printf("["); for (size_t i = 0; i < ids.size(); i++) printf("%s%d", i ? ", " : "", ids[i]); printf("]\n");
    return 0;
}

// ------------------------------------------------------------------------------------------------------------------------------
#include "chat.h"
#include "engine.h"
#include "spec.h"
#include <chrono>
#include <iostream>

namespace {
struct Session {
    Engine eng; Tokenizer tok; Gguf* gg = nullptr; uint32_t pos = 0; uint32_t ctx = 0; bool fast = false;
    double prefillSec = 0; uint32_t prefillTokens = 0;
    // Feeds tokens at positions [pos, pos+n); returns the greedy token after the last one (logits stay in eng.logits()).
    int feed(const std::vector<int32_t>& ids) {
        auto t0 = std::chrono::steady_clock::now();
        int last = eng.feed(ids.data(), (uint32_t)ids.size(), pos, true); pos += (uint32_t)ids.size();
        prefillSec += std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count(); prefillTokens += (uint32_t)ids.size(); return last;
    }
};
}

int cmdRun(int argc, char** argv, bool interactive) {
    std::string model, prompt, system, effort = "xhigh", metallib = "build/bonsai.metallib"; bool haveFile = false, raw = false, noThink = false, stats = true, noSpec = false; int maxTokens = 512; uint32_t ctx = 8192;
    SamplerParams sp; sp.seed = (uint64_t)std::chrono::steady_clock::now().time_since_epoch().count(); bool fast = false;
    for (int i = 2; i < argc; i++) {
        std::string a = argv[i]; auto nx = [&]() { return std::string(argv[++i]); };
        if (a == "--model") model = nx(); else if (a == "--prompt") prompt = nx(); else if (a == "--file") { prompt = readFile(argv[++i]); haveFile = true; } else if (a == "--system") system = nx(); else if (a == "--raw") raw = true;
        else if (a == "--no-think") noThink = true; else if (a == "--effort") effort = nx(); else if (a == "--max-tokens") maxTokens = atoi(nx().c_str()); else if (a == "--ctx") ctx = (uint32_t)atoi(nx().c_str());
        else if (a == "--temp") sp.temp = (float)atof(nx().c_str()); else if (a == "--top-k") sp.topK = atoi(nx().c_str()); else if (a == "--top-p") sp.topP = (float)atof(nx().c_str()); else if (a == "--min-p") sp.minP = (float)atof(nx().c_str());
        else if (a == "--seed") sp.seed = (uint64_t)atoll(nx().c_str()); else if (a == "--fast") fast = true; else if (a == "--exact") fast = false; else if (a == "--metallib") metallib = nx(); else if (a == "--quiet") stats = false; else if (a == "--no-spec") noSpec = true;
    }
    if (model.empty() || (!interactive && prompt.empty() && !haveFile)) { fprintf(stderr, "usage: bonsai run  --model M.gguf --prompt P [--system S] [--raw] [--no-think] [--effort xhigh|medium|low] [--max-tokens N] [--temp T --top-k K --top-p P --min-p P --seed S] [--fast] [--ctx N]\n       bonsai chat --model M.gguf [same options]\n"); return 1; }
    Session S; S.ctx = ctx; S.fast = fast;
    if (!S.eng.load(model, metallib, ctx, 0)) return 1; S.eng.int8Mode = fast; const bool spec = S.eng.hasMtp && !noSpec; S.eng.mtpTrack = spec;
    SpecStats sst;
    if (!S.tok.load(S.eng.gg)) return 1;
    ChatOpts co; co.thinking = !noThink; co.effort = effort; std::mt19937_64 rng(sp.seed);
    std::vector<Msg> history; Detok dt(&S.tok); int nOut = 0; double decSec = 0;
    int unfed = -1;   // last generated token that was not fed to the model (-1 if none)
    auto generate = [&](int first, std::vector<int32_t>& gen) {
        (void)first; unfed = -1;
        if (spec) {   // MTP speculative decoding (exact: the output distribution is unchanged)
            auto t0 = std::chrono::steady_clock::now(); size_t before = gen.size();
            SpecResult r = specGenerate(S.eng, S.pos, S.ctx, maxTokens, sp, rng, [&](int t) { return S.tok.isEog(t); }, [&](int t) {
                gen.push_back(t); if (S.tok.isEog(t)) return; std::string s = dt.push(t); if (!s.empty()) { fputs(s.c_str(), stdout); fflush(stdout); } }, &sst);
            S.pos = r.pos; unfed = r.unfed; nOut += (int)(gen.size() - before);
            decSec += std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
            std::string rest = dt.flush(); if (!rest.empty()) fputs(rest.c_str(), stdout); fputc('\n', stdout); fflush(stdout); return;
        }
        int tokenId = sampleToken(S.eng.logits(), (int)bonsai::kVocab, sp, rng);
        for (int n = 0; n < maxTokens; n++) {
            if (S.tok.isEog(tokenId)) { gen.push_back(tokenId); unfed = tokenId; break; }
            gen.push_back(tokenId); std::string s = dt.push(tokenId); if (!s.empty()) { fputs(s.c_str(), stdout); fflush(stdout); }
            if (S.pos + 1 >= S.ctx) { fprintf(stderr, "\n[context full]\n"); unfed = tokenId; break; }
            auto t0 = std::chrono::steady_clock::now(); int fedTok = tokenId; S.eng.feed(&fedTok, 1, S.pos++, true); decSec += std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count(); nOut++;
            tokenId = sampleToken(S.eng.logits(), (int)bonsai::kVocab, sp, rng);
        }
        std::string rest = dt.flush(); if (!rest.empty()) fputs(rest.c_str(), stdout); fputc('\n', stdout); fflush(stdout);
    };
    auto report = [&]() { if (stats && spec && sst.cycles) fprintf(stderr, "[mtp: %llu cycles, %.1f%% of drafts accepted, %.2f tokens/cycle | draft %.1f ms/cycle, verify %.1f ms/cycle, sampling %.1f ms/cycle]\n", (unsigned long long)sst.cycles, 100.0 * sst.accepted / sst.cycles, (double)(sst.cycles + sst.accepted) / sst.cycles, 1e3 * sst.draftSec / sst.cycles, 1e3 * sst.verifySec / sst.cycles, 1e3 * sst.sampleSec / sst.cycles);
        if (stats) fprintf(stderr, "[prefill %u tokens in %.2f s = %.1f tok/s | decode %d tokens in %.2f s = %.1f tok/s]\n", S.prefillTokens, S.prefillSec, S.prefillTokens / std::max(1e-9, S.prefillSec), nOut, decSec, nOut / std::max(1e-9, decSec)); };
    if (!interactive) {
        std::string text = raw ? prompt : renderChat({ {"system", system, ""}, {"user", prompt, ""} }, co);
        if (!raw && system.empty()) text = renderChat({ {"user", prompt, ""} }, co);
        std::vector<int32_t> ids = S.tok.encode(text, true);
        if (ids.size() + (size_t)maxTokens + 1 > ctx) { fprintf(stderr, "prompt (%zu tokens) + max-tokens does not fit in --ctx %u\n", ids.size(), ctx); return 1; }
        int first = S.feed(ids); std::vector<int32_t> gen; generate(first, gen); report(); return 0;
    }
    // interactive chat: keep the KV cache and the GDN state; each turn only prefills the new suffix
    fprintf(stderr, "bonsai chat (%s path). Empty line or /quit to exit, /reset to start over.\n", fast ? "int8 fast" : "exact");
    bool firstTurn = true; std::string line;
    while (true) {
        fputs("\n> ", stdout); fflush(stdout); if (!std::getline(std::cin, line)) break; if (line == "/quit" || line.empty()) break;
        if (line == "/reset") { S.eng.resetState(); S.pos = 0; firstTurn = true; fprintf(stderr, "[reset]\n"); continue; }
        std::string suffix;
        if (firstTurn) { std::vector<Msg> m; if (!system.empty()) m.push_back({"system", system, ""}); m.push_back({"user", line, ""}); suffix = renderChat(m, co); }
        else { suffix = "\n<|im_start|>user\n" + line + "<|im_end|>\n<|im_start|>assistant\n"; suffix += co.thinking ? "<think>\n" : "<think>\n\n</think>\n\n"; }
        firstTurn = false; std::vector<int32_t> ids = S.tok.encode(suffix, true);
        if (S.pos + ids.size() + (size_t)maxTokens + 1 > ctx) { fprintf(stderr, "[context would overflow --ctx %u; use /reset]\n", ctx); continue; }
        int first = S.feed(ids); std::vector<int32_t> gen; generate(first, gen);
        {   // close the assistant turn: feed the generated token that has not been fed yet (if any) and <|im_end|>
            std::vector<int32_t> tail; if (unfed >= 0 && !S.tok.isEog(unfed)) tail.push_back(unfed); tail.push_back(S.tok.imEnd);
            S.eng.feed(tail.data(), (uint32_t)tail.size(), S.pos, false); S.pos += (uint32_t)tail.size();
        }
        report();
    }
    return 0;
}

// ------------------------------------------------------------------------------------------------------------------------------
// bonsai mtp-eval: how often does the MTP head predict the main model's greedy continuation?
int cmdMtpEval(int argc, char** argv) {
    std::string model, ids, metallib = "build/bonsai.metallib"; int gen = 96;
    for (int i = 2; i < argc; i++) { std::string a = argv[i]; if (a == "--model") model = argv[++i]; else if (a == "--ids") ids = argv[++i]; else if (a == "--gen") gen = atoi(argv[++i]); else if (a == "--metallib") metallib = argv[++i]; }
    std::vector<int> X; { size_t p = 0; while (p < ids.size()) { size_t q = ids.find(',', p); if (q == std::string::npos) q = ids.size(); X.push_back(atoi(ids.substr(p, q - p).c_str())); p = q + 1; } }
    Engine eng; if (!eng.load(model, metallib, 4096, 0)) return 1; if (!eng.hasMtp) { fprintf(stderr, "model has no MTP block\n"); return 1; } eng.saveHidden = true;
    const size_t P = X.size(); std::vector<std::vector<float>> H;   // H[j] = plain output-normalised hidden state after processing position j
    int next = 0;
    for (size_t i = 0; i < P; i++) { next = eng.step(X[i], (uint32_t)i, -1, true); H.emplace_back(eng.hidden(), eng.hidden() + 5120); }
    for (int g = 0; g < gen; g++) { X.push_back(next); next = eng.step(next, (uint32_t)(X.size() - 1), -1, true); H.emplace_back(eng.hidden(), eng.hidden() + 5120); }
    // X has P + gen tokens whose hidden states are all in H (the final `next` is not part of X)
    size_t hitPrompt = 0, nPrompt = 0, hitGen = 0, nGen = 0;
    for (size_t j = 1; j + 1 < X.size(); j++) {
        int tok = X[j]; int pred = eng.mtpRows(1, &tok, H[j - 1].data(), (uint32_t)j, true);
        bool hit = pred == X[j + 1]; if (j + 1 < P) { nPrompt++; hitPrompt += hit; } else { nGen++; hitGen += hit; }
    }
    printf("MTP depth-1 top-1 match: on the given prompt %zu/%zu (%.1f%%), on the main model's greedy continuation %zu/%zu (%.1f%%)\n", hitPrompt, nPrompt, 100.0 * hitPrompt / std::max<size_t>(1, nPrompt), hitGen, nGen, 100.0 * hitGen / std::max<size_t>(1, nGen));
    return 0;
}

// ------------------------------------------------------------------------------------------------------------------------------
// bonsai verify-check: logits of the 2-token verification pass versus the exact single-token decode path, over many positions.
int cmdVerifyCheck(int argc, char** argv) {
    std::string model, ids, metallib = "build/bonsai.metallib";
    for (int i = 2; i < argc; i++) { std::string a = argv[i]; if (a == "--model") model = argv[++i]; else if (a == "--ids") ids = argv[++i]; else if (a == "--metallib") metallib = argv[++i]; }
    std::vector<int> X; { size_t p = 0; while (p < ids.size()) { size_t q = ids.find(',', p); if (q == std::string::npos) q = ids.size(); X.push_back(atoi(ids.substr(p, q - p).c_str())); p = q + 1; } }
    Engine eng; if (!eng.load(model, metallib, 4096, 0)) return 1; const int V = (int)bonsai::kVocab;
    double sumKl = 0, maxKl = 0, maxAbs = 0; size_t n = 0, agree = 0; double bSum[20] = {}, bMax[20] = {}; int bCnt[20] = {};
    auto cmp = [&](const float* ref, const float* got) {
        double mr = -1e30, mg = -1e30; int br = 0, bg = 0; for (int k = 0; k < V; k++) { if (ref[k] > mr) { mr = ref[k]; br = k; } if (got[k] > mg) { mg = got[k]; bg = k; } maxAbs = std::max(maxAbs, (double)std::fabs(ref[k] - got[k])); }
        double zr = 0, zg = 0; for (int k = 0; k < V; k++) { zr += std::exp((double)ref[k] - mr); zg += std::exp((double)got[k] - mg); }
        const double lr = std::log(zr) + mr, lg = std::log(zg) + mg; double kl = 0; for (int k = 0; k < V; k++) { double l = (double)ref[k] - lr, p = std::exp(l); if (p > 1e-12) kl += p * (l - ((double)got[k] - lg)); }
        { const size_t bk = std::min<size_t>(19, n / 100); bSum[bk] += kl; bMax[bk] = std::max(bMax[bk], kl); bCnt[bk]++; }
        sumKl += kl; maxKl = std::max(maxKl, kl); agree += (br == bg); n++;
    };
    std::vector<float> v0((size_t)V), v1((size_t)V);
    size_t j = 0; for (; j < 6 && j < X.size(); j++) eng.step(X[j], (uint32_t)j, -1, false);
    for (; j + 1 < X.size(); j += 2) {
        int vt[2] = {X[j], X[j + 1]}, am[2]; eng.verify2(vt, (uint32_t)j, am);
        memcpy(v0.data(), eng.verifyLogits(0), (size_t)V * 4); memcpy(v1.data(), eng.verifyLogits(1), (size_t)V * 4);
        eng.step(X[j], (uint32_t)j, -1, true); cmp(eng.logits(), v0.data());
        eng.step(X[j + 1], (uint32_t)(j + 1), -1, true); cmp(eng.logits(), v1.data());
    }
    for (int b = 0; b < 20; b++) if (bCnt[b]) printf("  checks %4d..%4d: mean KL %.2e  max KL %.2e\n", b * 100, b * 100 + bCnt[b] - 1, bSum[b] / bCnt[b], bMax[b]);
    printf("verify2 vs exact decode: %zu positions | mean KL %.3e | max KL %.3e | max |logit diff| %.4f | top-1 agreement %zu/%zu\n", n, sumKl / n, maxKl, maxAbs, agree, n);
    return 0;
}
