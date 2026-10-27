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
    if (model.empty() || !haveText) { fprintf(stderr, "usage: kata tokenize --model M.gguf (--text T | --file F) [--no-special]\n"); return 1; }
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
#include "ckpt.h"
#include "draft.h"
#include <chrono>
#include <thread>
#include <iostream>

namespace {
struct Session {
    Engine eng; Tokenizer tok; Gguf* gg = nullptr; uint32_t pos = 0; uint32_t ctx = 0; bool fast = false;
    double prefillSec = 0; uint32_t prefillTokens = 0;
    // Feeds tokens at positions [pos, pos+n); returns the greedy token after the last one (logits stay in eng.logits()).
    PrefixCache* cache = nullptr; int32_t userTok = -1; uint32_t restoredTokens = 0; double restoreSec = 0;
    int feedRange(const std::vector<int32_t>& ids, size_t a, size_t b) {
        auto t0 = std::chrono::steady_clock::now();
        int last = eng.feed(ids.data() + a, (uint32_t)(b - a), pos, true); pos += (uint32_t)(b - a);
        prefillSec += std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count(); prefillTokens += (uint32_t)(b - a); return last;
    }
    // Feeds the prompt. At position 0 the longest cached prefix is restored from disk first, and the checkpoints the cache plans (start of the last
    // user turn, end of the prompt) are written as the prompt is consumed. Later feeds (a chat turn continuing in this process) go straight to the engine.
    int feed(const std::vector<int32_t>& ids) {
        if (!(cache && cache->config().enabled && pos == 0 && ids.size() >= cache->config().minTokens)) return feedRange(ids, 0, ids.size());
        if (userTok < 0) { std::vector<int32_t> u = tok.encode("user", false); userTok = u.size() == 1 ? u[0] : -2; }
        size_t cur = 0; int last = -1;
        auto t0 = std::chrono::steady_clock::now(); int lt = -1; const uint32_t r = cache->restore(ids, &lt);
        if (r) { restoreSec = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count(); restoredTokens = r; cur = r; pos = r; fprintf(stderr, "[cache] restored %u of %zu prompt tokens in %.2f s\n", r, ids.size(), restoreSec); if (r == ids.size()) return lt; }
        for (uint32_t p : PrefixCache::plan(ids, tok.imStart, userTok, cache->config().minTokens)) { if (p <= cur) continue; last = feedRange(ids, cur, p); cur = p; cache->save(ids.data(), p); }
        if (cur < ids.size()) last = feedRange(ids, cur, ids.size());
        return last;
    }
};
}

int cmdRun(int argc, char** argv, bool interactive) {
    std::string model, prompt, system, effort = "xhigh", metallib = "build/kata.metallib"; bool haveFile = false, raw = false, noThink = false, stats = true, noSpec = false, chain = getenv("KATA_CHAIN") != nullptr; int maxTokens = 512; uint32_t ctx = 8192;
    CacheConfig cc; SamplerParams sp; sp.seed = (uint64_t)std::chrono::steady_clock::now().time_since_epoch().count(); std::string i8spec; bool i8explicit = false;   // int8 FFN prefill is opt-in: it does not stack with the default int8 KV cache inside the quality gate
    for (int i = 2; i < argc; i++) {
        std::string a = argv[i]; auto nx = [&]() { return std::string(argv[++i]); };
        if (a == "--model") model = nx(); else if (a == "--prompt") prompt = nx(); else if (a == "--file") { prompt = readFile(argv[++i]); haveFile = true; } else if (a == "--system") system = nx(); else if (a == "--raw") raw = true;
        else if (a == "--no-think") noThink = true; else if (a == "--effort") effort = nx(); else if (a == "--max-tokens") maxTokens = atoi(nx().c_str()); else if (a == "--ctx") ctx = (uint32_t)atoi(nx().c_str());
        else if (a == "--temp") sp.temp = (float)atof(nx().c_str()); else if (a == "--top-k") sp.topK = atoi(nx().c_str()); else if (a == "--top-p") sp.topP = (float)atof(nx().c_str()); else if (a == "--min-p") sp.minP = (float)atof(nx().c_str());
        else if (a == "--seed") sp.seed = (uint64_t)atoll(nx().c_str()); else if (a == "--fast") { i8spec = "all"; i8explicit = true; } else if (a == "--fast-ffn") { i8spec = "ffn"; i8explicit = true; } else if (a == "--i8") { i8spec = nx(); i8explicit = true; } else if (a == "--exact") { i8spec.clear(); setenv("KATA_KV", "fp16", 1); } else if (a == "--metallib") metallib = nx(); else if (a == "--quiet") stats = false; else if (a == "--no-spec") noSpec = true; else if (a == "--kv") setenv("KATA_KV", nx().c_str(), 1); else if (a == "--no-cache") cc.enabled = false; else if (a == "--cache-dir") cc.dir = nx(); else if (a == "--cache-gb") cc.budgetBytes = (uint64_t)(atof(nx().c_str()) * (1ull << 30)); else if (a == "--cache-min") cc.minTokens = (uint32_t)atoi(nx().c_str());
    }
    if (model.empty() || (!interactive && prompt.empty() && !haveFile)) { fprintf(stderr, "usage: kata run  --model M.gguf --prompt P [--system S] [--raw] [--no-think] [--effort xhigh|medium|low] [--max-tokens N] [--temp T --top-k K --top-p P --min-p P --seed S] [--fast] [--ctx N]\n       kata chat --model M.gguf [same options]\n"); return 1; }
    Session S; S.ctx = ctx; S.fast = !i8spec.empty();
    if (!S.eng.load(model, metallib, ctx, 0)) return 1; if (!S.eng.setI8(i8spec)) { fprintf(stderr, "--i8: unknown matrix name in \"%s\" (qkv z ssmout wq wk wv wo gate up down, all, ffn)\n", i8spec.c_str()); return 1; } const bool fast = S.eng.int8Mode; const bool spec = S.eng.hasMtp && !noSpec; S.eng.mtpTrack = spec;
    // the int8 KV cache is on by default (half the cache memory, +14% decode / +10% prefill at long context, top-1 99.71%);
    // the two int8 paths together measured 99.58% top-1, below the 99.7% gate, so the prefill stays exact unless asked for.
    // NOTE: the int8 KV cache is opt-in (--kv int8): the chunked prefill attention path measured mean KL 43.7 / max logit
    // diff 1.7e7 against fp16 KV, so it is not fit to be a default until that kernel is fixed.
    if (i8explicit && std::string(getenv("KATA_KV")) == "int8") {
        fprintf(stderr, "note: the int8 FFN prefill stacked on the int8 KV cache measured 99.58%% top-1 (gate 99.7%%); the prefill stays exact. Use --kv fp16 to combine them anyway.\n");
        i8spec.clear(); S.fast = false;
    }
    SpecStats sst;
    PrefixCache cache(S.eng, cc, spec); S.cache = &cache;
    if (!S.tok.load(S.eng.gg)) return 1;
    ChatOpts co; co.thinking = !noThink; co.effort = effort; std::mt19937_64 rng(sp.seed);
    std::vector<Msg> history; Detok dt(&S.tok); int nOut = 0; double decSec = 0;
    int unfed = -1;   // last generated token that was not fed to the model (-1 if none)
    auto generate = [&](int first, std::vector<int32_t>& gen) {
        (void)first; unfed = -1;
        if (spec) {   // MTP speculative decoding (exact: the output distribution is unchanged)
            auto t0 = std::chrono::steady_clock::now(); size_t before = gen.size();
            auto emitFn = [&](int t) { gen.push_back(t); if (S.tok.isEog(t)) return; std::string s = dt.push(t); if (!s.empty()) { fputs(s.c_str(), stdout); fflush(stdout); } };
            SpecResult r = chain ? specGenerateChain(S.eng, S.pos, S.ctx, maxTokens, sp, rng, [&](int t) { return S.tok.isEog(t); }, emitFn, &sst, true)
                            : specGenerate(S.eng, S.pos, S.ctx, maxTokens, sp, rng, [&](int t) { return S.tok.isEog(t); }, [&](int t) {
                gen.push_back(t); if (S.tok.isEog(t)) return; std::string s = dt.push(t); if (!s.empty()) { fputs(s.c_str(), stdout); fflush(stdout); } }, &sst);
            S.pos = r.pos; unfed = r.unfed; nOut += (int)(gen.size() - before);
            decSec += std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
            std::string rest = dt.flush(); if (!rest.empty()) fputs(rest.c_str(), stdout); fputc('\n', stdout); fflush(stdout); return;
        }
        int tokenId = sampleToken(S.eng.logits(), (int)kata::kVocab, sp, rng);
        for (int n = 0; n < maxTokens; n++) {
            if (S.tok.isEog(tokenId)) { gen.push_back(tokenId); unfed = tokenId; break; }
            gen.push_back(tokenId); std::string s = dt.push(tokenId); if (!s.empty()) { fputs(s.c_str(), stdout); fflush(stdout); }
            if (S.pos + 1 >= S.ctx) { fprintf(stderr, "\n[context full]\n"); unfed = tokenId; break; }
            auto t0 = std::chrono::steady_clock::now(); int fedTok = tokenId; S.eng.feed(&fedTok, 1, S.pos++, true); decSec += std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count(); nOut++;
            tokenId = sampleToken(S.eng.logits(), (int)kata::kVocab, sp, rng);
        }
        std::string rest = dt.flush(); if (!rest.empty()) fputs(rest.c_str(), stdout); fputc('\n', stdout); fflush(stdout);
    };
    auto report = [&]() { if (stats && spec && sst.cycles) fprintf(stderr, "[mtp: %llu cycles, %.1f%% of drafts accepted, %.2f tokens/cycle | draft %.1f ms/cycle, verify %.1f ms/cycle, sampling %.1f ms/cycle]\n", (unsigned long long)sst.cycles, 100.0 * sst.accepted / sst.cycles, (double)(sst.cycles + sst.accepted) / sst.cycles, 1e3 * sst.draftSec / sst.cycles, 1e3 * sst.verifySec / sst.cycles, 1e3 * sst.sampleSec / sst.cycles);
        if (stats && S.restoredTokens) fprintf(stderr, "[cache: %u prompt tokens restored from disk in %.2f s]\n", S.restoredTokens, S.restoreSec);
        if (stats) fprintf(stderr, "[prefill %u tokens in %.2f s = %.1f tok/s | decode %d tokens in %.2f s = %.1f tok/s]\n", S.prefillTokens, S.prefillSec, S.prefillTokens / std::max(1e-9, S.prefillSec), nOut, decSec, nOut / std::max(1e-9, decSec)); };
    if (!interactive) {
        std::string text = raw ? prompt : renderChat({ {"system", system, ""}, {"user", prompt, ""} }, co);
        if (!raw && system.empty()) text = renderChat({ {"user", prompt, ""} }, co);
        std::vector<int32_t> ids = S.tok.encode(text, true);
        if (ids.size() + (size_t)maxTokens + 1 > ctx) { fprintf(stderr, "prompt (%zu tokens) + max-tokens does not fit in --ctx %u\n", ids.size(), ctx); return 1; }
        int first = S.feed(ids); std::vector<int32_t> gen; generate(first, gen); report(); return 0;
    }
    // interactive chat: keep the KV cache and the GDN state; each turn only prefills the new suffix
    std::thread warm([&]() { S.eng.warmup(); });   // hides the ~3 s cold start behind the time the user spends typing
    struct Joiner { std::thread& t; ~Joiner() { if (t.joinable()) t.join(); } } joiner{warm};
    fprintf(stderr, "kata chat (%s). Empty line or /quit to exit, /reset to start over.\n", fast ? ("int8 prefill: " + S.eng.i8Names()).c_str() : "exact path");
    bool firstTurn = true; std::string line;
    while (true) {
        fputs("\n> ", stdout); fflush(stdout); if (!std::getline(std::cin, line)) break; if (line == "/quit" || line.empty()) break;
        if (warm.joinable()) warm.join();
        if (line == "/reset") { cache.waitIdle(); S.eng.resetState(); S.pos = 0; firstTurn = true; fprintf(stderr, "[reset]\n"); continue; }
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
// kata mtp-eval: how often does the MTP head predict the main model's greedy continuation?
int cmdMtpEval(int argc, char** argv) {
    std::string model, ids, metallib = "build/kata.metallib"; int gen = 96;
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
// kata verify-check: logits of the 2-token verification pass versus the exact single-token decode path, over many positions.
int cmdVerifyCheck(int argc, char** argv) {
    std::string model, ids, metallib = "build/kata.metallib";
    for (int i = 2; i < argc; i++) { std::string a = argv[i]; if (a == "--model") model = argv[++i]; else if (a == "--ids") ids = argv[++i]; else if (a == "--metallib") metallib = argv[++i]; }
    std::vector<int> X; { size_t p = 0; while (p < ids.size()) { size_t q = ids.find(',', p); if (q == std::string::npos) q = ids.size(); X.push_back(atoi(ids.substr(p, q - p).c_str())); p = q + 1; } }
    Engine eng; if (!eng.load(model, metallib, 4096, 0)) return 1; const int V = (int)kata::kVocab;
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

// ------------------------------------------------------------------------------------------------------------------------------
// kata kl-eval: quality of the int8 prefill path against the exact fp16 path on a text corpus. Both paths prefill the corpus chunk by chunk (each with its own state);
// at the last `keep` positions of every chunk the full-vocabulary distributions are compared (the first pass spills its logits to a temp file).
int cmdKlEval(int argc, char** argv) {
    std::string model, file, metallib = "build/kata.metallib", spill = "/tmp/kata_kl_spill.bin", saveRef, cmpRef, i8spec = "all"; bool i8given = false; int ntok = 16384, keep = 64, chunk = 1024;
    for (int i = 2; i < argc; i++) { std::string a = argv[i]; if (a == "--model") model = argv[++i]; else if (a == "--file") file = argv[++i]; else if (a == "--tokens") ntok = atoi(argv[++i]); else if (a == "--keep") keep = atoi(argv[++i]); else if (a == "--i8") { i8spec = argv[++i]; i8given = true; } else if (a == "--fast-ffn") { i8spec = "ffn"; i8given = true; } else if (a == "--save-ref") saveRef = argv[++i]; else if (a == "--ref") cmpRef = argv[++i]; else if (a == "--chunk") chunk = atoi(argv[++i]); else if (a == "--metallib") metallib = argv[++i]; }
    if (model.empty() || file.empty()) { fprintf(stderr, "usage: kata kl-eval --model M.gguf --file corpus.txt [--tokens N] [--keep K] [--chunk C]\n       KATA_KL_TRACE=1 prints the per-position KL trace and a KL-by-depth table; KATA_KL_DBG=1 prints outliers\n"); return 1; }
    Engine eng; if (!eng.load(model, metallib, (uint32_t)ntok + 64, 0)) return 1; Tokenizer tok; if (!tok.load(eng.gg)) return 1;
    std::vector<int32_t> ids = tok.encode(readFile(file.c_str()), false); if ((int)ids.size() > ntok) ids.resize((size_t)ntok);
    const int V = (int)kata::kVocab; const int n = (int)ids.size(); printf("corpus: %d tokens, chunk %d, comparing the last %d positions of each chunk\n", n, chunk, keep);
    const bool single = !saveRef.empty() || !cmpRef.empty();   // --save-ref / --ref: one pass with the configuration of this process (e.g. KATA_KV=int8) instead of exact-vs-int8 in one process
    FILE* sp = fopen(single ? (saveRef.empty() ? cmpRef.c_str() : saveRef.c_str()) : spill.c_str(), single && saveRef.empty() ? "rb" : "wb+"); if (!sp) { perror("spill"); return 1; }
    int gbase = 0;
    auto pass = [&](bool int8, auto&& onLogits) {
        eng.setI8(int8 ? i8spec : ""); eng.resetState();
        for (int pos = 0; pos < n; pos += chunk) { const int m = std::min(chunk, n - pos); const int nl = std::min(keep, m); gbase = pos + (m - nl); std::vector<int> t(ids.begin() + pos, ids.begin() + pos + m); eng.prefill(t.data(), (uint32_t)m, (uint32_t)pos, -1, true, (uint32_t)nl); onLogits(eng.logitsAll(), nl); }
    };
    if (!saveRef.empty()) { pass(i8given, [&](const float* lg, int nl) { fwrite(lg, 4, (size_t)nl * V, sp); }); fclose(sp); printf("saved the reference logits of this configuration to %s\n", saveRef.c_str()); return 0; }
    if (!single) { pass(false, [&](const float* lg, int nl) { fwrite(lg, 4, (size_t)nl * V, sp); }); fflush(sp); fseek(sp, 0, SEEK_SET); }
    double sumKl = 0, maxKl = 0, sumAbs = 0; size_t cnt = 0, agree = 0; std::vector<float> ref((size_t)keep * V);
    const bool trace = getenv("KATA_KL_TRACE") != nullptr;   // per-position KL trace + a KL-by-depth-bucket summary
    static const int bkEdges[6] = {1024, 4096, 8192, 16384, 32768, 1 << 30};
    double bkSum[6] = {0, 0, 0, 0, 0, 0}, bkMax[6] = {0, 0, 0, 0, 0, 0}; size_t bkN[6] = {0, 0, 0, 0, 0, 0}, bkBad[6] = {0, 0, 0, 0, 0, 0};
    pass(single ? i8given : true, [&](const float* lg, int nl) {
        if (fread(ref.data(), 4, (size_t)nl * V, sp) != (size_t)nl * V) { fprintf(stderr, "spill read failed\n"); exit(1); }
        for (int r = 0; r < nl; r++) {
            const float* a = ref.data() + (size_t)r * V; const float* b = lg + (size_t)r * V; double ma = -1e30, mb = -1e30; int ba = 0, bb = 0;
            for (int k = 0; k < V; k++) { if (a[k] > ma) { ma = a[k]; ba = k; } if (b[k] > mb) { mb = b[k]; bb = k; } }
            double za = 0, zb = 0; for (int k = 0; k < V; k++) { za += std::exp((double)a[k] - ma); zb += std::exp((double)b[k] - mb); }
            const double la = std::log(za) + ma, lb = std::log(zb) + mb; double kl = 0, ad = 0;
            for (int k = 0; k < V; k++) { const double l = (double)a[k] - la, p = std::exp(l); if (p > 1e-12) kl += p * (l - ((double)b[k] - lb)); ad = std::max(ad, (double)std::fabs(a[k] - b[k])); }
            { int bi = 0; while (bi < 5 && gbase + r >= bkEdges[bi]) bi++; bkSum[bi] += kl; bkN[bi]++; bkMax[bi] = std::max(bkMax[bi], kl); bkBad[bi] += (ba != bb); }
            if (trace) printf("%d %.6e %.4g %.4g %s\n", gbase + r, kl, ma, mb, ba == bb ? "ok" : "DIFF");
            if (getenv("KATA_KL_DBG") && kl > 1.0) fprintf(stderr, "  [bad] pos %d (row %d of chunk) KL %.3e logitmax ref %.4g / int8 %.4g\n", gbase + r, r, kl, ma, mb);
            sumKl += kl; maxKl = std::max(maxKl, kl); sumAbs = std::max(sumAbs, ad); cnt++; agree += (ba == bb);
        }
        printf("\r  compared %zu positions, running mean KL %.3e, top-1 %.2f%%", cnt, sumKl / cnt, 100.0 * agree / cnt); fflush(stdout);
    });
    if (trace) { printf("\n  KL by depth:\n    %-12s %8s %10s %10s %10s\n", "depth", "positions", "mean KL", "max KL", "top-1 miss");
        for (int i = 0; i < 6; i++) if (bkN[i]) printf("    %-12s %8zu %10.3e %10.3e %8zu (%.2f%%)\n",
            i == 0 ? "0-1K" : i == 1 ? "1K-4K" : i == 2 ? "4K-8K" : i == 3 ? "8K-16K" : i == 4 ? "16K-32K" : ">=32K",
            bkN[i], bkSum[i] / (double)bkN[i], bkMax[i], bkBad[i], 100.0 * (double)bkBad[i] / (double)bkN[i]); }
    if (const char* dv = getenv("KATA_KV_DUMP")) {   // dump the first 64 KB of layer 0's K and V planes, to diff two cache modes
        FILE* f = fopen(dv, "wb"); for (uint32_t pl : {0u, 1u}) fwrite(eng.kvPlanePtr(0, pl), 1, 65536, f); fclose(f);
        fprintf(stderr, "[dump] wrote %s (planes 0,1 of layer 0)\n", dv); }
    printf("\n%s: %zu positions | mean KL %.3e nat | max KL %.3e | max |logit diff| %.3f | top-1 agreement %zu/%zu (%.2f%%)\n", single ? "this configuration vs the reference file" : "int8 vs exact fp16", cnt, sumKl / cnt, maxKl, sumAbs, agree, cnt, 100.0 * agree / cnt);
    fclose(sp); if (!single) remove(spill.c_str()); return 0;
}

// ------------------------------------------------------------------------------------------------------------------------------
// kata cache [--cache-dir D] [--clear]: list or delete the persistent prefix cache
int cmdCache(int argc, char** argv) {
    std::string dir; bool clr = false;
    for (int i = 2; i < argc; i++) { std::string a = argv[i]; if (a == "--cache-dir") dir = argv[++i]; else if (a == "--clear") clr = true; }
    return listCacheDir(dir.empty() ? defaultCacheDir() : dir, clr);
}

// ------------------------------------------------------------------------------------------------------------------------------
// kata mtp-accept: offline estimate of the speculative acceptance rate in sampling mode. The model generates with the real sampler;
// at every position the MTP head's distribution q and the target's distribution p (both after top-k/top-p/min-p/temperature) are compared:
//   deterministic draft (argmax q):   acceptance = p(argmax q)
//   sampled draft (d ~ q, accept with min(1, p/q), residual resampling on rejection):   acceptance = sum_v min(p(v), q(v))
int cmdMtpAccept(int argc, char** argv) {
    std::string model, metallib = "build/kata.metallib"; int gen = 200; SamplerParams sp; sp.seed = 7;
    for (int i = 2; i < argc; i++) { std::string a = argv[i]; if (a == "--model") model = argv[++i]; else if (a == "--gen") gen = atoi(argv[++i]); else if (a == "--temp") sp.temp = (float)atof(argv[++i]); else if (a == "--top-k") sp.topK = atoi(argv[++i]); else if (a == "--top-p") sp.topP = (float)atof(argv[++i]); else if (a == "--min-p") sp.minP = (float)atof(argv[++i]); else if (a == "--seed") sp.seed = (uint64_t)atoll(argv[++i]); else if (a == "--metallib") metallib = argv[++i]; }
    Engine eng; if (!eng.load(model, metallib, 4096, 0)) return 1; if (!eng.hasMtp) { fprintf(stderr, "model has no MTP block\n"); return 1; } eng.mtpTrack = true; eng.saveHidden = true;
    Tokenizer tok; if (!tok.load(eng.gg)) return 1; const int V = (int)kata::kVocab; std::mt19937_64 rng(sp.seed);
    struct Case { const char* name; const char* prompt; bool think; } cases[] = {
        {"prose (no thinking)", "Write a short story about a lighthouse keeper who discovers a message in a bottle.", false},
        {"code (no thinking)", "Write a Python function that returns all prime numbers below n using a sieve, then explain its complexity.", false},
        {"reasoning (thinking)", "A train leaves the station at 3 pm travelling at 60 mph. A second train leaves the same station at 4 pm in the same direction at 80 mph. At what time does the second train catch up? Show your reasoning.", true},
        {"explanation (no thinking)", "Explain how a transformer language model generates text, step by step, for a curious teenager.", false} };
    double totDet = 0, totSamp = 0, totTop1 = 0; size_t totN = 0;
    for (auto& c : cases) {
        eng.resetState(); ChatOpts co; co.thinking = c.think; std::vector<int32_t> ids = tok.encode(renderChat({{"user", c.prompt, ""}}, co), true);
        int g = eng.feed(ids.data(), (uint32_t)ids.size(), 0, true); (void)g; uint32_t pos = (uint32_t)ids.size();
        std::vector<float> pl((size_t)V), ql((size_t)V); std::vector<float> hPrev(eng.hPrev(), eng.hPrev() + 5120);
        int x = sampleToken(eng.logits(), V, sp, rng); double sDet = 0, sSamp = 0, sTop1 = 0; int n = 0;
        for (int t = 0; t < gen; t++) {
            eng.mtpRows(1, &x, hPrev.data(), pos, true); memcpy(ql.data(), eng.logits(), (size_t)V * 4);          // q: MTP prediction for position pos+1 from (H[pos-1], x)
            eng.step(x, pos, -1, true); memcpy(pl.data(), eng.logits(), (size_t)V * 4); memcpy(hPrev.data(), eng.hidden(), 5120 * 4);   // p: the target's distribution for the same position
            const SparseDist P = samplerDistribution(pl.data(), V, sp), Q = samplerDistribution(ql.data(), V, sp);
            const int a = Q.ids[0];    // ids are sorted by logit, so ids[0] is the argmax of q
            sDet += P.prob(a); double ov = 0; for (size_t i = 0; i < Q.ids.size(); i++) ov += std::min(P.prob(Q.ids[i]), Q.w[i] / Q.z); sSamp += ov; int pa = P.ids[0]; sTop1 += (pa == a); n++;
            x = sampleFromDist(P, rng); pos++;
        }
        printf("%-28s  det-argmax draft: %5.1f%%   sampled draft: %5.1f%%   (top-1 agreement %4.1f%%, %d positions)\n", c.name, 100 * sDet / n, 100 * sSamp / n, 100 * sTop1 / n, n);
        totDet += sDet; totSamp += sSamp; totTop1 += sTop1; totN += (size_t)n;
    }
    printf("%-28s  det-argmax draft: %5.1f%%   sampled draft: %5.1f%%   (top-1 agreement %4.1f%%)   => tokens/cycle %.2f -> %.2f\n", "ALL", 100 * totDet / totN, 100 * totSamp / totN, 100 * totTop1 / totN, 1 + totDet / totN, 1 + totSamp / totN);
    return 0;
}

// ------------------------------------------------------------------------------------------------------------------------------
// kata dflash-eval: measures the DFlash2 drafter's acceptance and the round cost, end to end (prompt and continuation both go through
// the wide verification path so the draft's context is materialised from the target's residuals).
int cmdDflashEval(int argc, char** argv) {
    std::string model, draftPath, ids, metallib = "build/kata.metallib"; int gen = 48;
    for (int i = 2; i < argc; i++) { std::string a = argv[i]; if (a == "--model") model = argv[++i]; else if (a == "--draft") draftPath = argv[++i];
        else if (a == "--ids") ids = argv[++i]; else if (a == "--gen") gen = atoi(argv[++i]); else if (a == "--metallib") metallib = argv[++i]; }
    std::vector<int> X; { size_t p = 0; while (p < ids.size()) { size_t q = ids.find(',', p); if (q == std::string::npos) q = ids.size(); X.push_back(atoi(ids.substr(p, q - p).c_str())); p = q + 1; } }
    if (model.empty() || draftPath.empty() || X.empty()) { fprintf(stderr, "usage: kata dflash-eval --model M --draft D.gguf --ids 1,2,3 [--gen N]\n"); return 1; }
    Engine eng; if (!eng.load(model, metallib, 8192, 0)) return 1;
    Draft dr; if (!dr.load(eng, draftPath)) return 1; eng.mtpTrack = false;
    if (getenv("KATA_TAP_SHIFT")) { const int sh = atoi(getenv("KATA_TAP_SHIFT")); for (int i = 0; i < 5; i++) dr.targetLayers[i] += sh; }
    const int V = (int)kata::kVocab, T = (int)Draft::kBlock;
    MTL::Buffer* taps = eng.gpu.alloc(5 * 8 * 5120 * 2); MTL::Buffer* logits = eng.bufLogitsAllPublic();
    int am[8]; uint32_t pos = 0; int anchor = -1;
    auto argmaxOf = [&](int row) { const float* lg = (const float*)logits->contents() + (size_t)row * V; int b = 0; for (int k = 1; k < V; k++) if (lg[k] > lg[b]) b = k; return b; };
    auto t0 = std::chrono::steady_clock::now();
    while (pos < (uint32_t)X.size()) {   // prompt: 8-token verification windows, then materialise the draft context
        const uint32_t n = std::min<uint32_t>(T, (uint32_t)X.size() - pos);
        eng.verifyWide(X.data() + pos, n, pos, am, taps, dr.targetLayers, 5);
        eng.commitWide(n);
        if (getenv("KATA_POISON_CTX")) { memset(dr.kbuf->contents(), 0x3c, dr.kbuf->length()); memset(dr.vbuf->contents(), 0x3c, dr.vbuf->length()); }
        else if (!getenv("KATA_NO_CTX")) { dr.materialise(eng, taps->contents(), n);
            if (getenv("KATA_DBG_CTX")) { const _Float16* tp = (const _Float16*)taps->contents(); double st = 0; int nt = 0;
                for (int i = 0; i < 5 * (int)n * 5120; i++) { double v = std::fabs((double)tp[i]); st += v; if (v != 0) nt++; }
                const _Float16* ch = (const _Float16*)dr.cH->contents(); double sc2 = 0; int nc = 0;
                for (int i = 0; i < (int)n * 5120; i++) { double v = std::fabs((double)ch[i]); sc2 += v; if (v != 0) nc++; }
                const _Float16* kr = (const _Float16*)dr.kbuf->contents(); double sk = 0; int nk = 0;
                for (int i = 0; i < 4096 * 1024; i++) { double v = std::fabs((double)kr[i]); sk += v; if (v != 0) nk++; }
                fprintf(stderr, "[ctx] taps sum|.|=%.1f nonzero=%d/%d | cH sum|.|=%.1f nonzero=%d | ringL0 sum|.|=%.1f nonzero=%d\n", st, nt, 5 * (int)n * 5120, sc2, nc, sk, nk); } }
        pos += n; anchor = am[n - 1];
    }
    double promptSec = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    printf("prompt %zu tokens via 8-row windows in %.2f s; anchor %d at pos %u\n", X.size(), promptSec, anchor, pos);
    uint64_t rounds = 0, accepted = 0, produced = 0; double roundSec = 0, draftSec = 0;
    int d[8];
    if (getenv("KATA_DF_TRUTH")) {   // diagnostic: run the block with the true continuation in rows 1..7 instead of masks
        std::vector<int> truth; int a = anchor; uint32_t p2 = pos;
        for (int i = 0; i < T; i++) { truth.push_back(a); a = eng.step(a, p2, -1, true); p2++; }
        int amT[8]; eng.resetState();
        for (uint32_t q2 = 0; q2 < pos; ) { const uint32_t n2 = std::min<uint32_t>(T, pos - q2); eng.verifyWide(X.data() + q2, n2, q2, amT, taps, dr.targetLayers, 5); eng.commitWide(n2); if (!getenv("KATA_NO_CTX")) dr.materialise(eng, taps->contents(), n2); q2 += n2; }
        int tk[8]; for (int i = 0; i < T; i++) tk[i] = truth[(size_t)i];
        eng.xEmbedBlock(tk, (uint32_t)T);
        fprintf(stderr, "[truth] block inputs:"); for (int i = 0; i < T; i++) fprintf(stderr, " %d", tk[i]); fprintf(stderr, "\n");
        dr.forwardBlockTruth(eng, tk, pos, logits, T);
        fprintf(stderr, "[truth] row argmax:"); for (int i = 0; i < T; i++) fprintf(stderr, " %d(exp %d)", argmaxOf(i), i + 1 < T ? truth[(size_t)i + 1] : -1); fprintf(stderr, "\n");
        return 0;
    }
    for (int g = 0; g < gen; rounds++) {
        auto dt0 = std::chrono::steady_clock::now();
        dr.forwardBlock(eng, anchor, pos, logits);                       // proposes rows 1..7
        draftSec += std::chrono::duration<double>(std::chrono::steady_clock::now() - dt0).count();
        if (rounds == 0) {   // self-consistency: row 0 sees the anchor at its own position, so it should predict the anchor itself
            const float* lg = (const float*)logits->contents(); double sa = 0; int nz = 0;
            for (int k = 0; k < V; k++) { sa += std::fabs(lg[k]); if (lg[k] != 0.0f) nz++; }
            int rank = 0; for (int k = 0; k < V; k++) if (lg[k] > lg[anchor]) rank++;
            int top5[5] = {0,0,0,0,0};
            for (int t5 = 0; t5 < 5; t5++) { int b = -1; for (int k = 0; k < V; k++) { bool used = false; for (int q5 = 0; q5 < t5; q5++) if (top5[q5] == k) used = true; if (!used && (b < 0 || lg[k] > lg[b])) b = k; } top5[t5] = b; }
            fprintf(stderr, "[draft] row0 top5:"); for (int t5 = 0; t5 < 5; t5++) fprintf(stderr, " %d(%.2f)", top5[t5], lg[top5[t5]]); fprintf(stderr, " | anchor %d logit %.2f | row1 top1 %d\n", anchor, lg[anchor], argmaxOf(1));
            fprintf(stderr, "[draft] row0 argmax %d (%s anchor %d, rank %d), sum|.|=%.1f nonzero=%d; row1 argmax %d\n",
                    argmaxOf(0), argmaxOf(0) == anchor ? "==" : "!=", anchor, rank, sa, nz, argmaxOf(1)); }
        if (rounds == 0 && getenv("KATA_DBG_DRAFT")) { const float* lg = (const float*)logits->contents(); double s0 = 0; int nz = 0;
            for (int k = 0; k < V; k++) { s0 += std::fabs(lg[k]); if (lg[k] != 0.0f) nz++; }
            fprintf(stderr, "[draft] row0 logits: %.4f %.4f %.4f ... sum|.| %.1f nonzero %d, argmax %d; row1 argmax %d\n", lg[0], lg[1], lg[2], s0, nz, argmaxOf(0), argmaxOf(1)); }
        if (getenv("KATA_DF_ARGMAX")) { for (int i = 0; i < T - 1; i++) d[i] = argmaxOf(i + 1); } else dr.propose(eng, logits, (uint32_t)T, d, anchor);
        int win[8]; win[0] = anchor; for (int i = 0; i < T - 1; i++) win[i + 1] = d[i];
        auto r0 = std::chrono::steady_clock::now();
        eng.verifyWide(win, (uint32_t)T, pos, am, taps, dr.targetLayers, 5);
        int match = 0; while (match < T - 1 && d[match] == am[match]) match++;
        const uint32_t F = (uint32_t)match + 1;                          // the anchor is confirmed
        eng.commitWide(F);
        if (getenv("KATA_POISON_CTX")) { memset(dr.kbuf->contents(), 0x3c, dr.kbuf->length()); memset(dr.vbuf->contents(), 0x3c, dr.vbuf->length()); }
        else if (!getenv("KATA_NO_CTX")) dr.materialise(eng, taps->contents(), F);
        roundSec += std::chrono::duration<double>(std::chrono::steady_clock::now() - r0).count();
        accepted += (uint64_t)match; produced += F; g += (int)F; pos += F; anchor = am[F - 1];
        if (rounds < 3 || getenv("KATA_DBG_DRAFT")) { printf("  round %llu: drafts", (unsigned long long)rounds); for (int i = 0; i < T - 1; i++) printf(" %d", d[i]); printf("\n            trunk  "); for (int i = 0; i < T - 1; i++) printf(" %d", am[i]); printf("\n            accept %d -> %u tokens\n", match, F); }
    }
    printf("DFlash2: %llu rounds, %llu tokens, %.2f tokens/round, %.1f%% drafts accepted, %.1f ms/round, %.1f tok/s\n",
           (unsigned long long)rounds, (unsigned long long)produced, (double)produced / rounds, 100.0 * accepted / (rounds * (T - 1)), 1e3 * roundSec / rounds,
           produced / (roundSec + 1e-9));
    printf("  draft forward: %.1f ms/round, verify+commit+materialise: %.1f ms/round\n", 1e3 * draftSec / rounds, 1e3 * roundSec / rounds);
    return 0;
}
