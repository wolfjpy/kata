// Minimal GGUF v3 reader: memory-maps the file, indexes tensors and the few metadata arrays the engine needs.
#pragma once
#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

struct GgufTensor {
    std::string name;
    uint32_t type = 0;
    uint32_t nDims = 0;
    uint64_t dims[4] = {1, 1, 1, 1};
    uint64_t offset = 0;      // relative to the data region
    uint64_t nbytes = 0;
    uint64_t fileOffset = 0;  // absolute offset in the file
};

struct GgufArray { uint32_t elemType = 0; uint64_t count = 0; uint64_t payload = 0; /* absolute file offset of first element */ };

enum GgmlType : uint32_t { GGML_F32 = 0, GGML_Q8_0 = 8, GGML_BF16 = 30, GGML_PTQ1_0 = 143 };

struct Gguf {
    int fd = -1;
    void* map = nullptr;          // page-aligned mapping of the whole file
    size_t fileSize = 0;
    size_t mapLen = 0;            // fileSize rounded up to the 16 KiB page size
    uint64_t dataStart = 0;
    std::unordered_map<std::string, GgufTensor> tensors;
    std::unordered_map<std::string, uint64_t> u32;     // scalar integer metadata
    std::unordered_map<std::string, GgufArray> arrays; // array metadata
    std::unordered_map<std::string, std::string> strs; // short string metadata (<= 256 bytes)

    bool open(const char* path);
    const GgufTensor& tensor(const std::string& name) const;   // aborts if missing
    const uint8_t* bytes(uint64_t fileOffset) const { return (const uint8_t*)map + fileOffset; }
    std::vector<int32_t> signedInts(const std::string& key) const;   // integer array -> int32 vector
    std::vector<std::string> strArray(const std::string& key) const; // string array
    void close();
};
