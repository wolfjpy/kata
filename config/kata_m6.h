// Single source of truth for every compile-time constant of this engine.
// Shared by the C++ host and the Metal shaders. Target: Ternary Bonsai-2-27B (qwen35, PTQ1_0) on Apple M6 (12 GPU cores).
// kata-pack / the loader static-assert / verify the GGUF metadata against these values.
#pragma once
#ifdef __METAL_VERSION__
  #include <metal_stdlib>
  #define KATA_CONST constant constexpr
#else
  #include <cstdint>
  #define KATA_CONST inline constexpr
#endif

namespace kata {

// ---- model geometry (qwen35 / Qwen3.8-27B) ----
KATA_CONST uint32_t kLayers        = 64;     // main layers; blk.64 is the MTP layer
KATA_CONST uint32_t kAllBlocks     = 65;
KATA_CONST uint32_t kHidden        = 5120;
KATA_CONST uint32_t kFfn           = 17408;
KATA_CONST uint32_t kVocab         = 248320;
KATA_CONST float    kRmsEps        = 1e-6f;
KATA_CONST uint32_t kFullAttnEvery = 4;      // layer il is full attention iff il % 4 == 3

// full attention
KATA_CONST uint32_t kHeads         = 24;
KATA_CONST uint32_t kKvHeads       = 4;
KATA_CONST uint32_t kHeadDim       = 256;
KATA_CONST uint32_t kRotDim        = 64;     // partial rotary: first 64 dims, pairs (i, i+32)
KATA_CONST float    kRopeTheta     = 10000000.0f;
KATA_CONST uint32_t kQProjOut      = kHeads * kHeadDim * 2;   // 12288: per head [query(256) | gate(256)]
KATA_CONST uint32_t kKvDim         = kKvHeads * kHeadDim;     // 1024
KATA_CONST uint32_t kAttnOutIn     = kHeads * kHeadDim;       // 6144

// gated delta net (linear attention)
KATA_CONST uint32_t kGdnKHeads     = 16;
KATA_CONST uint32_t kGdnVHeads     = 48;
KATA_CONST uint32_t kGdnKDim       = 128;
KATA_CONST uint32_t kGdnVDim       = 128;
KATA_CONST uint32_t kGdnConvK      = 4;
KATA_CONST uint32_t kGdnQkvDim     = 2 * kGdnKHeads * kGdnKDim + kGdnVHeads * kGdnVDim;  // 10240
KATA_CONST uint32_t kGdnZDim       = kGdnVHeads * kGdnVDim;                              // 6144

// ---- PTQ1_0 weight format ----
KATA_CONST uint32_t kGroup         = 128;    // weights per scale group
KATA_CONST uint32_t kBlockBytes    = 28;     // 24 qs + 2 qh + fp16 scale

// ---- Hadamard rotation ----
KATA_CONST uint32_t kHadBlock      = 1024;   // normalized Sylvester-Walsh-Hadamard, block-diagonal along the input dim

// ---- hardware (Mac mini M6) ----
KATA_CONST uint32_t kGpuCores      = 12;
KATA_CONST uint32_t kMaxResidentThreadsPerCore = 3072;   // measured
KATA_CONST uint32_t kThreadgroupMem = 32768;
KATA_CONST uint32_t kCacheLine     = 128;

inline bool layerIsAttention(uint32_t il) { return (il % kFullAttnEvery) == (kFullAttnEvery - 1); }

}  // namespace kata
