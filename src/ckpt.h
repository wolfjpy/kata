// Persistent prefix cache: a checkpoint is the complete continuation state of the engine at an exact token frontier
// (GDN and conv state, the K/V of every attention layer, the MTP K/V rows, the last hidden state and the last logits).
// A new process that is given a prompt with the same token prefix restores the longest checkpoint instead of prefilling it.
#pragma once
#include "engine.h"
#include <cstdint>
#include <string>
#include <thread>
#include <vector>

struct CacheConfig {
    bool enabled = true;
    std::string dir;                        // default: $BONSAI_CACHE_DIR or ~/.cache/bonsai
    uint64_t budgetBytes = 40ull << 30;     // least recently used checkpoints are deleted above this
    uint32_t minTokens = 2048;              // shorter prefixes are cheaper to recompute than to store
};

struct CacheEntryInfo { std::string path; uint32_t n = 0; uint64_t bytes = 0; double ageSec = 0; bool hasMtp = false; };

class PrefixCache {
public:
    PrefixCache(Engine& eng, const CacheConfig& cfg, bool wantMtp);
    ~PrefixCache() { waitIdle(); }
    // Restores the longest cached prefix of `ids` into the engine (which must be at position 0). Returns the number of restored tokens (0 = miss).
    // When the whole prompt was restored, lastTok is the greedy token after it and the engine's logits are set.
    uint32_t restore(const std::vector<int32_t>& ids, int* lastTok);
    // The engine has consumed exactly ids[0, n) and its last feed produced logits: snapshot the state and write it in the background.
    void save(const int32_t* ids, uint32_t n);
    void waitIdle();
    // Where checkpoints of a prompt should be taken: the start of the last user turn (when it is far enough in) and the end of the prompt.
    static std::vector<uint32_t> plan(const std::vector<int32_t>& ids, int32_t imStart, int32_t userTok, uint32_t minTokens);
    std::vector<CacheEntryInfo> list() const;
    uint32_t clear();
    const CacheConfig& config() const { return cfg_; }
    uint64_t savedBytes = 0; uint32_t savedCount = 0, hits = 0; double restoreSec = 0;
private:
    Engine& eng_; CacheConfig cfg_; bool wantMtp_; uint64_t fp_; std::string tag_;
    std::thread writer_;
    void writeFile(std::string path, std::string tmp, std::vector<int32_t> toks, uint32_t n, std::vector<uint8_t> gdn, std::vector<uint8_t> conv, std::vector<uint8_t> hprev, std::vector<float> logits, int lastTok, bool mtp);
    void enforceBudget(const std::string& keep);
};

// the cache directory ($BONSAI_CACHE_DIR, else ~/.cache/bonsai), created if needed
std::string defaultCacheDir();
// bonsai cache: list (and optionally delete) the checkpoints of a directory without loading a model
int listCacheDir(const std::string& dir, bool clear);
