#include "gguf.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

namespace {
struct Reader {
    const uint8_t* base; uint64_t p;
    template <class T> T rd() { T v; memcpy(&v, base + p, sizeof(T)); p += sizeof(T); return v; }
    std::string str() { uint64_t n = rd<uint64_t>(); std::string s((const char*)base + p, n); p += n; return s; }
};
size_t elemSize(uint32_t t) {
    switch (t) { case 0: case 1: case 7: return 1; case 2: case 3: return 2; case 4: case 5: case 6: return 4; case 10: case 11: case 12: return 8; default: return 0; }
}
void skipValue(Reader& r, uint32_t t) {
    if (t == 8) { r.str(); return; }
    if (t == 9) { uint32_t et = r.rd<uint32_t>(); uint64_t n = r.rd<uint64_t>(); if (et == 8) { for (uint64_t i = 0; i < n; i++) r.str(); } else r.p += n * elemSize(et); return; }
    r.p += elemSize(t);
}
// ggml type sizes needed for tensor byte counts
uint64_t tensorBytes(uint32_t type, uint64_t nelem) {
    switch (type) {
        case GGML_F32: return nelem * 4;
        case GGML_BF16: return nelem * 2;
        case GGML_Q8_0: return nelem / 32 * 34;
        case GGML_PTQ1_0: return nelem / 128 * 28;
        case GGML_Q2_K: return nelem / 256 * 84;
        case GGML_Q3_K: return nelem / 256 * 110;
        case GGML_Q4_K: return nelem / 256 * 144;
        case GGML_Q6_K: return nelem / 256 * 210;
        default: fprintf(stderr, "gguf: unsupported tensor type %u\n", type); exit(1);
    }
}
}  // namespace

bool Gguf::open(const char* path) {
    fd = ::open(path, O_RDONLY); if (fd < 0) { perror("open"); return false; }
    struct stat st; fstat(fd, &st); fileSize = (size_t)st.st_size;
    const size_t page = 16384; mapLen = (fileSize + page - 1) / page * page;
    // Map a page-rounded length: the tail of the last page is zero-filled, so MTLBuffer(bytesNoCopy) can wrap the whole mapping.
    map = mmap(nullptr, mapLen, PROT_READ, MAP_PRIVATE, fd, 0); if (map == MAP_FAILED) { perror("mmap"); return false; }
    Reader r{(const uint8_t*)map, 0};
    if (memcmp(map, "GGUF", 4) != 0) { fprintf(stderr, "not a GGUF file\n"); return false; }
    r.p = 4; uint32_t ver = r.rd<uint32_t>(); if (ver != 3) { fprintf(stderr, "unsupported GGUF version %u\n", ver); return false; }
    uint64_t nT = r.rd<uint64_t>(), nKV = r.rd<uint64_t>();
    uint64_t alignment = 32;
    for (uint64_t i = 0; i < nKV; i++) {
        std::string key = r.str(); uint32_t t = r.rd<uint32_t>();
        if (t == 9) { GgufArray a; a.elemType = r.rd<uint32_t>(); a.count = r.rd<uint64_t>(); a.payload = r.p; arrays[key] = a;
            if (a.elemType == 8) { for (uint64_t k = 0; k < a.count; k++) r.str(); } else r.p += a.count * elemSize(a.elemType); }
        else if (t == 8) { std::string s = r.str(); if (s.size() <= 256) strs[key] = s; }
        else if (t == 4) { u32[key] = r.rd<uint32_t>(); } else if (t == 5) { u32[key] = (uint64_t)(int64_t)r.rd<int32_t>(); }
        else if (t == 10) { u32[key] = r.rd<uint64_t>(); } else if (t == 7 || t == 0 || t == 1) { u32[key] = r.rd<uint8_t>(); }
        else { skipValue(r, t); }
        if (key == "general.alignment") alignment = u32[key];
    }
    std::vector<GgufTensor> list; list.reserve(nT);
    for (uint64_t i = 0; i < nT; i++) {
        GgufTensor t; t.name = r.str(); t.nDims = r.rd<uint32_t>(); uint64_t ne = 1;
        for (uint32_t d = 0; d < t.nDims; d++) { t.dims[d] = r.rd<uint64_t>(); ne *= t.dims[d]; }
        t.type = r.rd<uint32_t>(); t.offset = r.rd<uint64_t>(); t.nbytes = tensorBytes(t.type, ne); list.push_back(t);
    }
    dataStart = (r.p + alignment - 1) / alignment * alignment;
    for (auto& t : list) { t.fileOffset = dataStart + t.offset; tensors[t.name] = t; }
    return true;
}
const GgufTensor& Gguf::tensor(const std::string& name) const {
    auto it = tensors.find(name); if (it == tensors.end()) { fprintf(stderr, "gguf: missing tensor %s\n", name.c_str()); exit(1); }
    return it->second;
}
std::vector<int32_t> Gguf::signedInts(const std::string& key) const {
    auto it = arrays.find(key); if (it == arrays.end()) { fprintf(stderr, "gguf: missing array %s\n", key.c_str()); exit(1); }
    const GgufArray& a = it->second; std::vector<int32_t> out(a.count); const uint8_t* p = bytes(a.payload);
    for (uint64_t i = 0; i < a.count; i++) {
        switch (a.elemType) {
            case 0: out[i] = ((const uint8_t*)p)[i]; break; case 1: out[i] = ((const int8_t*)p)[i]; break;
            case 4: out[i] = (int32_t)((const uint32_t*)p)[i]; break; case 5: out[i] = ((const int32_t*)p)[i]; break;
            default: fprintf(stderr, "gguf: array %s has unsupported element type %u\n", key.c_str(), a.elemType); exit(1);
        }
    }
    return out;
}
void Gguf::close() { if (map) munmap(map, mapLen); if (fd >= 0) ::close(fd); map = nullptr; fd = -1; }
std::vector<std::string> Gguf::strArray(const std::string& key) const {
    auto it = arrays.find(key); if (it == arrays.end() || it->second.elemType != 8) { fprintf(stderr, "gguf: missing string array %s\n", key.c_str()); exit(1); }
    const GgufArray& a = it->second; std::vector<std::string> out; out.reserve(a.count); uint64_t p = a.payload;
    for (uint64_t i = 0; i < a.count; i++) { uint64_t n; memcpy(&n, bytes(p), 8); p += 8; out.emplace_back((const char*)bytes(p), n); p += n; }
    return out;
}
