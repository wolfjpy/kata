#include "ckpt.h"
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dirent.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <sys/time.h>
#include <unistd.h>

using namespace bonsai;

namespace {
constexpr uint32_t kVersion = 2;
constexpr uint64_t kAlign = 4096;
struct Hdr {
    char magic[8]; uint32_t version, n; uint64_t modelFp; char tag[64];
    uint32_t hasMtp, nAttn, kvRow, pad0; uint64_t gdnBytes, convBytes, hash0, hash1; int32_t lastTok, pad1;
    uint64_t tokensOff, gdnOff, convOff, kvOff, mtpOff, hOff, logitsOff, total;
};
static_assert(sizeof(Hdr) < kAlign, "header must fit in one block");
constexpr size_t kHidBytes = 5120 * 4;

uint64_t alignUp(uint64_t x) { return (x + kAlign - 1) / kAlign * kAlign; }
struct Layout { uint64_t tokensOff, gdnOff, convOff, kvOff, mtpOff, hOff, logitsOff, total; };
constexpr uint32_t kMtpRow = kKvDim * 2;   // the MTP layer keeps an fp16 K/V cache in every mode
// kvRowSum = bytes per position over all KV planes of one attention layer (fp16: K + V; int8: K + V + both scale planes)
Layout layoutFor(uint32_t n, uint64_t gdnBytes, uint64_t convBytes, uint32_t nAttn, uint32_t kvRowSum, bool mtp) {
    Layout l{}; l.tokensOff = kAlign; l.gdnOff = alignUp(l.tokensOff + 4ull * n); l.convOff = alignUp(l.gdnOff + gdnBytes); l.kvOff = alignUp(l.convOff + convBytes);
    l.mtpOff = alignUp(l.kvOff + (uint64_t)nAttn * n * kvRowSum); l.hOff = alignUp(l.mtpOff + (mtp ? 2ull * n * kMtpRow : 0)); l.logitsOff = alignUp(l.hOff + (mtp ? kHidBytes : 0));
    l.total = l.logitsOff + (uint64_t)kVocab * 4; return l;
}
// two independent FNV-1a streams over the token ids; prefix hashes for every length
void prefixHashes(const int32_t* t, size_t n, std::vector<uint64_t>& h0, std::vector<uint64_t>& h1) {
    h0.assign(n + 1, 0); h1.assign(n + 1, 0); uint64_t a = 1469598103934665603ull, b = 0x9E3779B97F4A7C15ull; h0[0] = a; h1[0] = b;
    for (size_t i = 0; i < n; i++) { const uint8_t* p = (const uint8_t*)&t[i]; for (int k = 0; k < 4; k++) { a = (a ^ p[k]) * 1099511628211ull; b = (b ^ p[k] ^ 0x5bd1e995ull) * 0x100000001b3ull; } h0[i + 1] = a; h1[i + 1] = b; }
}
bool preadAll(int fd, void* buf, size_t len, uint64_t off) {
    uint8_t* p = (uint8_t*)buf; while (len) { const size_t c = std::min<size_t>(len, 1ull << 28); ssize_t r = pread(fd, p, c, (off_t)off); if (r <= 0) return false; p += r; off += (uint64_t)r; len -= (size_t)r; } return true;
}
bool pwriteAll(int fd, const void* buf, size_t len, uint64_t off) {
    const uint8_t* p = (const uint8_t*)buf; while (len) { const size_t c = std::min<size_t>(len, 1ull << 28); ssize_t r = pwrite(fd, p, c, (off_t)off); if (r <= 0) return false; p += r; off += (uint64_t)r; len -= (size_t)r; } return true;
}
std::string hex64(uint64_t v) { char b[20]; snprintf(b, sizeof b, "%016llx", (unsigned long long)v); return b; }
struct Found { std::string path; Hdr h; uint64_t bytes; double mtime; };
std::vector<Found> scan(const std::string& dir) {
    std::vector<Found> out; DIR* d = opendir(dir.c_str()); if (!d) return out;
    while (dirent* e = readdir(d)) {
        const std::string name = e->d_name; if (name.size() < 6 || name.substr(name.size() - 5) != ".ckpt") continue;
        const std::string path = dir + "/" + name; struct stat st; if (stat(path.c_str(), &st) != 0) continue;
        int fd = open(path.c_str(), O_RDONLY); if (fd < 0) continue; Hdr h; const bool ok = preadAll(fd, &h, sizeof h, 0); close(fd);
        if (!ok || memcmp(h.magic, "BNSCKPT1", 8) != 0 || h.version != kVersion) continue;
        out.push_back({path, h, (uint64_t)st.st_size, (double)st.st_mtime});
    }
    closedir(d); return out;
}
double nowSec() { return (double)time(nullptr); }
}  // namespace

std::string defaultCacheDir() {
    std::string d; if (const char* e = getenv("BONSAI_CACHE_DIR")) d = e; else { const char* h = getenv("HOME"); d = std::string(h ? h : ".") + "/.cache/bonsai"; }
    std::string cur; for (size_t i = 0; i <= d.size(); i++) { if (i == d.size() || d[i] == '/') { if (!cur.empty()) mkdir(cur.c_str(), 0755); } if (i < d.size()) cur += d[i]; }
    return d;
}

PrefixCache::PrefixCache(Engine& eng, const CacheConfig& cfg, bool wantMtp) : eng_(eng), cfg_(cfg), wantMtp_(wantMtp && eng.hasMtp) {
    if (cfg_.dir.empty()) cfg_.dir = defaultCacheDir(); else { std::string cur; for (size_t i = 0; i <= cfg_.dir.size(); i++) { if (i == cfg_.dir.size() || cfg_.dir[i] == '/') { if (!cur.empty()) mkdir(cur.c_str(), 0755); } if (i < cfg_.dir.size()) cur += cfg_.dir[i]; } }
    fp_ = eng.modelFingerprint(); tag_ = eng.numericsTag();
}

void PrefixCache::waitIdle() { if (writer_.joinable()) writer_.join(); }

std::vector<uint32_t> PrefixCache::plan(const std::vector<int32_t>& ids, int32_t imStart, int32_t userTok, uint32_t minTokens) {
    std::vector<uint32_t> pts; const uint32_t L = (uint32_t)ids.size(); int64_t boundary = -1;
    for (int64_t i = (int64_t)L - 2; i >= 0; i--) if (ids[(size_t)i] == imStart && ids[(size_t)i + 1] == userTok) { boundary = i; break; }
    if (boundary >= (int64_t)minTokens) pts.push_back((uint32_t)boundary);
    else for (uint32_t p = 8192; p + 256 <= L; p += 8192) pts.push_back(p);       // no usable message boundary: bound the recomputation to 8K tokens
    if (L >= minTokens && (pts.empty() || L - pts.back() >= 256)) pts.push_back(L);
    return pts;
}

uint32_t PrefixCache::restore(const std::vector<int32_t>& ids, int* lastTok) {
    if (!cfg_.enabled || ids.size() < cfg_.minTokens) return 0;
    waitIdle(); const auto t0 = std::chrono::steady_clock::now();
    std::vector<Found> es = scan(cfg_.dir); std::vector<uint64_t> h0, h1; prefixHashes(ids.data(), ids.size(), h0, h1);
    std::sort(es.begin(), es.end(), [](const Found& a, const Found& b) { return a.h.n > b.h.n; });
    for (const Found& f : es) {
        const Hdr& h = f.h; const uint32_t n = h.n;
        if (n == 0 || n > ids.size() || n > eng_.maxCtx || h.modelFp != fp_ || strncmp(h.tag, tag_.c_str(), sizeof h.tag) != 0 || h.nAttn != eng_.numAttn() || h.kvRow != kvRowSum() || h.gdnBytes != eng_.gdnBytes() || h.convBytes != eng_.convBytes()) continue;
        if (wantMtp_ && !h.hasMtp) continue;
        if (h.hash0 != h0[n] || h.hash1 != h1[n]) continue;
        const Layout l = layoutFor(n, h.gdnBytes, h.convBytes, h.nAttn, h.kvRow, h.hasMtp != 0); if (f.bytes != l.total) continue;
        int fd = open(f.path.c_str(), O_RDONLY); if (fd < 0) continue;
        std::vector<int32_t> toks(n); bool ok = preadAll(fd, toks.data(), 4ull * n, l.tokensOff) && memcmp(toks.data(), ids.data(), 4ull * n) == 0;
        ok = ok && preadAll(fd, eng_.gdnPtr(), h.gdnBytes, l.gdnOff) && preadAll(fd, eng_.convPtr(), h.convBytes, l.convOff);
        for (uint32_t a = 0; ok && a < h.nAttn; a++) { uint64_t pre = 0; for (uint32_t pl = 0; ok && pl < eng_.kvPlanes(); pl++) { const uint64_t rb = eng_.kvPlaneRow(pl); ok = preadAll(fd, eng_.kvPlanePtr(a, pl), (uint64_t)n * rb, l.kvOff + (uint64_t)n * ((uint64_t)a * h.kvRow + pre)); pre += rb; } }
        if (ok && wantMtp_) ok = preadAll(fd, eng_.mtpKvPtr(false), (uint64_t)n * kMtpRow, l.mtpOff) && preadAll(fd, eng_.mtpKvPtr(true), (uint64_t)n * kMtpRow, l.mtpOff + (uint64_t)n * kMtpRow) && preadAll(fd, eng_.hPrevPtr(), kHidBytes, l.hOff);
        if (ok && n == ids.size()) { ok = preadAll(fd, eng_.logitsMut(), (uint64_t)kVocab * 4, l.logitsOff); if (ok && lastTok) *lastTok = h.lastTok; }
        close(fd);
        if (!ok) { fprintf(stderr, "[cache] %s is unreadable, ignoring it\n", f.path.c_str()); eng_.resetState(); continue; }
        struct timeval tv[2]; gettimeofday(&tv[0], nullptr); tv[1] = tv[0]; utimes(f.path.c_str(), tv);
        hits++; restoreSec += std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count(); return n;
    }
    return 0;
}

void PrefixCache::save(const int32_t* ids, uint32_t n) {
    if (!cfg_.enabled || n < cfg_.minTokens || n > eng_.maxCtx) return;
    std::vector<uint64_t> h0, h1; prefixHashes(ids, n, h0, h1);
    uint64_t th = fp_; for (char c : tag_) th = (th ^ (uint8_t)c) * 1099511628211ull;                       // the file name carries the numerics tag and the model, so paths of different modes never collide
    const std::string path = cfg_.dir + "/" + hex64(h0[n]) + "-" + hex64(th).substr(0, 8) + "-" + std::to_string(n) + ".ckpt";
    const Layout want = layoutFor(n, eng_.gdnBytes(), eng_.convBytes(), eng_.numAttn(), kvRowSum(), wantMtp_);
    struct stat st; if (stat(path.c_str(), &st) == 0) {
        if ((uint64_t)st.st_size == want.total) { struct timeval tv[2]; gettimeofday(&tv[0], nullptr); tv[1] = tv[0]; utimes(path.c_str(), tv); return; }   // already there and complete: just refresh its age
        unlink(path.c_str());                                                                                       // truncated or from an older layout: rewrite it
    }
    waitIdle();
    std::vector<uint8_t> gdn(eng_.gdnPtr(), eng_.gdnPtr() + eng_.gdnBytes()), conv(eng_.convPtr(), eng_.convPtr() + eng_.convBytes()), hp;
    if (wantMtp_) hp.assign(eng_.hPrevPtr(), eng_.hPrevPtr() + kHidBytes);
    std::vector<float> logits(eng_.logitsMut(), eng_.logitsMut() + kVocab); int last = 0; for (uint32_t i = 1; i < kVocab; i++) if (logits[i] > logits[(size_t)last]) last = (int)i;
    std::vector<int32_t> toks(ids, ids + n); const std::string tmp = path + ".tmp";
    writer_ = std::thread([this, path, tmp, toks = std::move(toks), n, gdn = std::move(gdn), conv = std::move(conv), hp = std::move(hp), logits = std::move(logits), last]() mutable { writeFile(path, tmp, std::move(toks), n, std::move(gdn), std::move(conv), std::move(hp), std::move(logits), last, wantMtp_); });
}

void PrefixCache::writeFile(std::string path, std::string tmp, std::vector<int32_t> toks, uint32_t n, std::vector<uint8_t> gdn, std::vector<uint8_t> conv, std::vector<uint8_t> hprev, std::vector<float> logits, int lastTok, bool mtp) {
    const uint32_t nAttn = eng_.numAttn(), kvRow = kvRowSum(); const Layout l = layoutFor(n, gdn.size(), conv.size(), nAttn, kvRow, mtp);
    std::vector<uint64_t> h0, h1; prefixHashes(toks.data(), n, h0, h1);
    Hdr h{}; memcpy(h.magic, "BNSCKPT1", 8); h.version = kVersion; h.n = n; h.modelFp = fp_; strncpy(h.tag, tag_.c_str(), sizeof h.tag - 1); h.hasMtp = mtp ? 1 : 0; h.nAttn = nAttn; h.kvRow = kvRow;
    h.gdnBytes = gdn.size(); h.convBytes = conv.size(); h.hash0 = h0[n]; h.hash1 = h1[n]; h.lastTok = lastTok; h.tokensOff = l.tokensOff; h.gdnOff = l.gdnOff; h.convOff = l.convOff; h.kvOff = l.kvOff; h.mtpOff = l.mtpOff; h.hOff = l.hOff; h.logitsOff = l.logitsOff; h.total = l.total;
    int fd = open(tmp.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0644); if (fd < 0) { fprintf(stderr, "[cache] cannot write %s\n", tmp.c_str()); return; }
    bool ok = ftruncate(fd, (off_t)l.total) == 0 && pwriteAll(fd, &h, sizeof h, 0) && pwriteAll(fd, toks.data(), 4ull * n, l.tokensOff) && pwriteAll(fd, gdn.data(), gdn.size(), l.gdnOff) && pwriteAll(fd, conv.data(), conv.size(), l.convOff);
    for (uint32_t a = 0; ok && a < nAttn; a++) { uint64_t pre = 0; for (uint32_t pl = 0; ok && pl < eng_.kvPlanes(); pl++) { const uint64_t rb = eng_.kvPlaneRow(pl); ok = pwriteAll(fd, eng_.kvPlanePtr(a, pl), (uint64_t)n * rb, l.kvOff + (uint64_t)n * ((uint64_t)a * kvRow + pre)); pre += rb; } }
    if (ok && mtp) ok = pwriteAll(fd, eng_.mtpKvPtr(false), (uint64_t)n * kMtpRow, l.mtpOff) && pwriteAll(fd, eng_.mtpKvPtr(true), (uint64_t)n * kMtpRow, l.mtpOff + (uint64_t)n * kMtpRow) && pwriteAll(fd, hprev.data(), hprev.size(), l.hOff);
    if (ok) ok = pwriteAll(fd, logits.data(), (uint64_t)kVocab * 4, l.logitsOff);
    close(fd);
    if (!ok || rename(tmp.c_str(), path.c_str()) != 0) { fprintf(stderr, "[cache] writing %s failed\n", path.c_str()); unlink(tmp.c_str()); return; }
    savedBytes += l.total; savedCount++; enforceBudget(path);
}

void PrefixCache::enforceBudget(const std::string& keep) {
    std::vector<Found> es = scan(cfg_.dir); uint64_t total = 0; for (auto& f : es) total += f.bytes;
    std::sort(es.begin(), es.end(), [](const Found& a, const Found& b) { return a.mtime < b.mtime; });
    for (auto& f : es) { if (total <= cfg_.budgetBytes) break; if (f.path == keep) continue; if (unlink(f.path.c_str()) == 0) total -= f.bytes; }
}

std::vector<CacheEntryInfo> PrefixCache::list() const {
    std::vector<CacheEntryInfo> out; const double now = nowSec(); for (auto& f : scan(cfg_.dir)) out.push_back({f.path, f.h.n, f.bytes, now - f.mtime, f.h.hasMtp != 0});
    std::sort(out.begin(), out.end(), [](const CacheEntryInfo& a, const CacheEntryInfo& b) { return a.ageSec < b.ageSec; }); return out;
}
uint32_t PrefixCache::clear() {
    uint32_t c = 0; DIR* d = opendir(cfg_.dir.c_str()); if (!d) return 0;
    while (dirent* e = readdir(d)) { const std::string nm = e->d_name; if ((nm.size() > 5 && nm.substr(nm.size() - 5) == ".ckpt") || (nm.size() > 4 && nm.substr(nm.size() - 4) == ".tmp")) { if (unlink((cfg_.dir + "/" + nm).c_str()) == 0) c++; } }
    closedir(d); return c;
}

// bonsai cache: list or clear a cache directory without loading a model
int listCacheDir(const std::string& dir, bool clear) {
    auto es = scan(dir); uint64_t total = 0; for (auto& f : es) total += f.bytes;
    std::sort(es.begin(), es.end(), [](const Found& a, const Found& b) { return a.mtime > b.mtime; });
    printf("cache directory %s: %zu checkpoints, %.2f GB\n", dir.c_str(), es.size(), total / 1e9);
    for (auto& f : es) printf("  %7u tokens  %6.2f GB  %s  %s\n", f.h.n, f.bytes / 1e9, f.h.hasMtp ? "mtp" : "   ", f.h.tag);
    if (clear) { uint32_t c = 0; DIR* d = opendir(dir.c_str()); if (d) { while (dirent* e = readdir(d)) { const std::string nm = e->d_name; if ((nm.size() > 5 && nm.substr(nm.size() - 5) == ".ckpt") || (nm.size() > 4 && nm.substr(nm.size() - 4) == ".tmp")) if (unlink((dir + "/" + nm).c_str()) == 0) c++; } closedir(d); } printf("deleted %u files\n", c); }
    return 0;
}
