#include "cli.h"
#include "gguf.h"
#include "tokenizer.h"
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
#include <chrono>
#include <iostream>

namespace {
struct Session {
    Engine eng; Tokenizer tok; Gguf* gg = nullptr; uint32_t pos = 0; uint32_t ctx = 0; bool fast = false;
    double prefillSec = 0; uint32_t prefillTokens = 0;
    // Feeds tokens at positions [pos, pos+n); returns the greedy token after the last one (logits stay in eng.logits()).
    int feed(const std::vector<int32_t>& ids) {
        auto t0 = std::chrono::steady_clock::now(); int last = -1; size_t i = 0;
        if (ids.size() <= 24) { for (; i < ids.size(); i++) last = eng.step(ids[i], pos++, -1, true); }   // short suffixes: stepwise decode is cheaper than a full prefill pass
        else while (i < ids.size()) { uint32_t m = (uint32_t)std::min<size_t>(Engine::kMaxChunk, ids.size() - i); last = eng.prefill(ids.data() + i, m, pos, -1, true, 1); pos += m; i += m; }
        prefillSec += std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count(); prefillTokens += (uint32_t)ids.size(); return last;
    }
};
}

int cmdRun(int argc, char** argv, bool interactive) {
    std::string model, prompt, system, effort = "xhigh", metallib = "build/bonsai.metallib"; bool haveFile = false, raw = false, noThink = false, stats = true; int maxTokens = 512; uint32_t ctx = 8192;
    SamplerParams sp; sp.seed = (uint64_t)std::chrono::steady_clock::now().time_since_epoch().count(); bool fast = false;
    for (int i = 2; i < argc; i++) {
        std::string a = argv[i]; auto nx = [&]() { return std::string(argv[++i]); };
        if (a == "--model") model = nx(); else if (a == "--prompt") prompt = nx(); else if (a == "--file") { prompt = readFile(argv[++i]); haveFile = true; } else if (a == "--system") system = nx(); else if (a == "--raw") raw = true;
        else if (a == "--no-think") noThink = true; else if (a == "--effort") effort = nx(); else if (a == "--max-tokens") maxTokens = atoi(nx().c_str()); else if (a == "--ctx") ctx = (uint32_t)atoi(nx().c_str());
        else if (a == "--temp") sp.temp = (float)atof(nx().c_str()); else if (a == "--top-k") sp.topK = atoi(nx().c_str()); else if (a == "--top-p") sp.topP = (float)atof(nx().c_str()); else if (a == "--min-p") sp.minP = (float)atof(nx().c_str());
        else if (a == "--seed") sp.seed = (uint64_t)atoll(nx().c_str()); else if (a == "--fast") fast = true; else if (a == "--exact") fast = false; else if (a == "--metallib") metallib = nx(); else if (a == "--quiet") stats = false;
    }
    if (model.empty() || (!interactive && prompt.empty() && !haveFile)) { fprintf(stderr, "usage: bonsai run  --model M.gguf --prompt P [--system S] [--raw] [--no-think] [--effort xhigh|medium|low] [--max-tokens N] [--temp T --top-k K --top-p P --min-p P --seed S] [--fast] [--ctx N]\n       bonsai chat --model M.gguf [same options]\n"); return 1; }
    Session S; S.ctx = ctx; S.fast = fast;
    if (!S.eng.load(model, metallib, ctx, 0)) return 1; S.eng.int8Mode = fast;
    if (!S.tok.load(S.eng.gg)) return 1;
    ChatOpts co; co.thinking = !noThink; co.effort = effort; std::mt19937_64 rng(sp.seed);
    std::vector<Msg> history; Detok dt(&S.tok); int nOut = 0; double decSec = 0;
    auto generate = [&](int first, std::vector<int32_t>& gen) {
        // `first` is the greedy token from the prompt's last position, but we sample from the logits instead
        (void)first; int tokenId = sampleToken(S.eng.logits(), (int)bonsai::kVocab, sp, rng);
        for (int n = 0; n < maxTokens; n++) {
            if (S.tok.isEog(tokenId)) { gen.push_back(tokenId); break; }
            gen.push_back(tokenId); std::string s = dt.push(tokenId); if (!s.empty()) { fputs(s.c_str(), stdout); fflush(stdout); }
            if (S.pos + 1 >= S.ctx) { fprintf(stderr, "\n[context full]\n"); break; }
            auto t0 = std::chrono::steady_clock::now(); S.eng.step(tokenId, S.pos++, -1, true); decSec += std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count(); nOut++;
            tokenId = sampleToken(S.eng.logits(), (int)bonsai::kVocab, sp, rng);
        }
        std::string rest = dt.flush(); if (!rest.empty()) fputs(rest.c_str(), stdout); fputc('\n', stdout); fflush(stdout);
    };
    auto report = [&]() { if (stats) fprintf(stderr, "[prefill %u tokens in %.2f s = %.1f tok/s | decode %d tokens in %.2f s = %.1f tok/s]\n", S.prefillTokens, S.prefillSec, S.prefillTokens / std::max(1e-9, S.prefillSec), nOut, decSec, nOut / std::max(1e-9, decSec)); };
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
        if (!gen.empty() && S.tok.isEog(gen.back())) { /* the template appends "\n" after <|im_end|>; it is fed with the next turn */ }
        else { /* generation stopped early: close the assistant turn so the next suffix is well formed */ int32_t e = S.tok.imEnd; S.eng.step(e, S.pos++, -1, false); }
        report();
    }
    return 0;
}
