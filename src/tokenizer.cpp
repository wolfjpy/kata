#include "tokenizer.h"
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <queue>

static void appendUtf8(std::string& s, uint32_t cp) {
    if (cp < 0x80) s += (char)cp; else if (cp < 0x800) { s += (char)(0xC0 | (cp >> 6)); s += (char)(0x80 | (cp & 0x3F)); }
    else if (cp < 0x10000) { s += (char)(0xE0 | (cp >> 12)); s += (char)(0x80 | ((cp >> 6) & 0x3F)); s += (char)(0x80 | (cp & 0x3F)); }
    else { s += (char)(0xF0 | (cp >> 18)); s += (char)(0x80 | ((cp >> 12) & 0x3F)); s += (char)(0x80 | ((cp >> 6) & 0x3F)); s += (char)(0x80 | (cp & 0x3F)); }
}

bool Tokenizer::load(const Gguf& gg) {
    tokens = gg.strArray("tokenizer.ggml.tokens"); tokenType = gg.signedInts("tokenizer.ggml.token_type");
    for (size_t i = 0; i < tokens.size(); i++) tok2id.emplace(tokens[i], (int32_t)i);
    // GPT-2 byte <-> unicode mapping
    int n = 0; bool direct[256] = {};
    for (int b = 33; b <= 126; b++) direct[b] = true; for (int b = 161; b <= 172; b++) direct[b] = true; for (int b = 174; b <= 255; b++) direct[b] = true;
    for (int b = 0; b < 256; b++) { uint32_t cp = direct[b] ? (uint32_t)b : (uint32_t)(256 + n++); byteUni[b].clear(); appendUtf8(byteUni[b], cp); uniByte[cp] = (uint8_t)b; }
    for (int b = 0; b < 256; b++) { auto it = tok2id.find(byteUni[b]); if (it == tok2id.end()) { fprintf(stderr, "tokenizer: byte token %d missing\n", b); return false; } byteTok[b] = it->second; }
    std::vector<std::string> mg = gg.strArray("tokenizer.ggml.merges");
    merges.reserve(mg.size() * 2);
    for (size_t r = 0; r < mg.size(); r++) {
        size_t sp = mg[r].find(' '); if (sp == std::string::npos) continue;
        auto a = tok2id.find(mg[r].substr(0, sp)), b = tok2id.find(mg[r].substr(sp + 1)), m = tok2id.find(mg[r].substr(0, sp) + mg[r].substr(sp + 1));
        if (a == tok2id.end() || b == tok2id.end() || m == tok2id.end()) continue;
        merges.emplace(((uint64_t)(uint32_t)a->second << 32) | (uint32_t)b->second, std::make_pair((int32_t)r, m->second));
    }
    // literal added tokens: control (3) and user-defined (4) types
    for (size_t i = 0; i < tokens.size(); i++) if ((tokenType[i] == 3 || tokenType[i] == 4) && !tokens[i].empty()) specials.emplace_back(tokens[i], (int32_t)i);
    std::sort(specials.begin(), specials.end(), [](auto& a, auto& b) { return a.first.size() > b.first.size(); });
    auto find = [&](const char* s) { auto it = tok2id.find(s); return it == tok2id.end() ? -1 : it->second; };
    eosIm = find("<|im_end|>"); eosText = find("<|endoftext|>"); imStart = find("<|im_start|>"); imEnd = eosIm;
    return true;
}

void Tokenizer::bpe(const uint8_t* s, size_t n, std::vector<int32_t>& out) const {
    if (n == 0) return;
    std::vector<int32_t> id(n), prev(n), next(n); std::vector<char> alive(n, 1);
    for (size_t i = 0; i < n; i++) { id[i] = byteTok[s[i]]; prev[i] = (int32_t)i - 1; next[i] = (i + 1 < n) ? (int32_t)i + 1 : -1; }
    struct Cand { int32_t rank, left, a, b; bool operator<(const Cand& o) const { return rank != o.rank ? rank > o.rank : left > o.left; } };
    std::priority_queue<Cand> pq;
    auto push = [&](int32_t l) { int32_t r = next[l]; if (r < 0) return; auto it = merges.find(((uint64_t)(uint32_t)id[l] << 32) | (uint32_t)id[r]); if (it != merges.end()) pq.push({it->second.first, l, id[l], id[r]}); };
    for (size_t i = 0; i + 1 < n; i++) push((int32_t)i);
    while (!pq.empty()) {
        Cand c = pq.top(); pq.pop(); int32_t l = c.left; if (!alive[l] || id[l] != c.a) continue; int32_t r = next[l]; if (r < 0 || !alive[r] || id[r] != c.b) continue;
        id[l] = merges.find(((uint64_t)(uint32_t)c.a << 32) | (uint32_t)c.b)->second.second; alive[r] = 0; next[l] = next[r]; if (next[r] >= 0) prev[next[r]] = l;
        if (prev[l] >= 0) push(prev[l]); push(l);
    }
    for (int32_t i = 0; i >= 0 && i < (int32_t)n; i = next[i]) { if (alive[i]) out.push_back(id[i]); if (next[i] < 0) break; }
}

std::vector<int32_t> Tokenizer::encode(const std::string& text, bool parseSpecial) const {
    std::vector<int32_t> out; std::vector<std::pair<uint32_t, uint32_t>> spans;
    auto encodePlain = [&](const std::string& chunk) {
        if (chunk.empty()) return; std::string nf = pretokNfc(chunk); pretokSplit(nf, spans);
        for (auto& sp : spans) bpe((const uint8_t*)nf.data() + sp.first, sp.second - sp.first, out);
    };
    if (!parseSpecial) { encodePlain(text); return out; }
    size_t start = 0, i = 0;
    while (i < text.size()) {
        bool hit = false;
        if (text[i] == '<') for (auto& sp : specials) if (text.compare(i, sp.first.size(), sp.first) == 0) { encodePlain(text.substr(start, i - start)); out.push_back(sp.second); i += sp.first.size(); start = i; hit = true; break; }
        if (!hit) i++;
    }
    encodePlain(text.substr(start)); return out;
}

std::string Tokenizer::piece(int32_t id) const {
    if (id < 0 || (size_t)id >= tokens.size()) return "";
    if (tokenType[id] == 3 || tokenType[id] == 4) return tokens[id];
    std::string out; const std::string& s = tokens[id];
    for (size_t i = 0; i < s.size();) {
        unsigned char c = (unsigned char)s[i]; uint32_t cp; size_t len;
        if (c < 0x80) { cp = c; len = 1; } else if ((c >> 5) == 6) { cp = c & 0x1F; len = 2; } else if ((c >> 4) == 14) { cp = c & 0x0F; len = 3; } else { cp = c & 0x07; len = 4; }
        for (size_t k = 1; k < len && i + k < s.size(); k++) cp = (cp << 6) | ((unsigned char)s[i + k] & 0x3F);
        auto it = uniByte.find(cp); if (it != uniByte.end()) out += (char)it->second; else appendUtf8(out, cp);
        i += len;
    }
    return out;
}

std::string Detok::push(int32_t id) {
    pending += t->piece(id);
    size_t n = pending.size(), cut = n;
    // hold back a trailing incomplete UTF-8 sequence
    for (size_t k = 1; k <= 3 && k <= n; k++) { unsigned char c = (unsigned char)pending[n - k]; if ((c & 0xC0) == 0x80) continue; size_t need = c < 0x80 ? 1 : (c >> 5) == 6 ? 2 : (c >> 4) == 14 ? 3 : (c >> 3) == 30 ? 4 : 1; if (need > k) cut = n - k; break; }
    std::string out = pending.substr(0, cut); pending.erase(0, cut); return out;
}
std::string Detok::flush() { std::string o = pending; pending.clear(); return o; }
