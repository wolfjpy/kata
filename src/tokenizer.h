// Byte-level BPE tokenizer for Qwen3.5 / Qwen3.8 (vocab and merges come from the GGUF).
#pragma once
#include "gguf.h"
#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

// Implemented in pretok.mm (ICU regex via Foundation): NFC normalisation and the Qwen pre-tokenizer split.
std::string pretokNfc(const std::string& utf8);
void pretokSplit(const std::string& utf8, std::vector<std::pair<uint32_t, uint32_t>>& spans);   // byte spans [begin, end)

struct Tokenizer {
    std::vector<std::string> tokens;                 // byte-unicode strings as stored in the GGUF
    std::unordered_map<std::string, int32_t> tok2id;
    std::vector<int32_t> tokenType;
    std::unordered_map<uint64_t, std::pair<int32_t, int32_t>> merges;   // (idA<<32 | idB) -> (rank, merged id)
    int32_t byteTok[256];
    std::string byteUni[256];                        // byte -> UTF-8 of its byte-unicode char
    std::unordered_map<uint32_t, uint8_t> uniByte;   // code point -> byte
    std::vector<std::pair<std::string, int32_t>> specials;   // literal added tokens, longest first
    int32_t eosIm = -1, eosText = -1, imStart = -1, imEnd = -1;

    bool load(const Gguf& gg);
    std::vector<int32_t> encode(const std::string& text, bool parseSpecial = true) const;
    std::string piece(int32_t id) const;             // raw bytes of a token (special tokens: their literal text)
    bool isEog(int32_t id) const { return id == eosIm || id == eosText; }
private:
    void bpe(const uint8_t* s, size_t n, std::vector<int32_t>& out) const;
};

// Incremental detokenizer: holds back incomplete UTF-8 sequences so streamed output never contains broken characters.
struct Detok {
    const Tokenizer* t; std::string pending;
    explicit Detok(const Tokenizer* tk) : t(tk) {}
    std::string push(int32_t id);
    std::string flush();
};
