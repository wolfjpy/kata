// Single source of truth for every compile-time constant of this engine.
// Shared by the C++ host and the Metal shaders. Target: Ternary Bonsai 2 27B (qwen35, PTQ1_0) on Apple M6 (12 GPU cores).
// bonsai-pack / the loader static-assert / verify the GGUF metadata against these values.
#pragma once
#ifdef __METAL_VERSION__
  #include <metal_stdlib>
  #define BONSAI_CONST constant constexpr
#else
  #include <cstdint>
  #define BONSAI_CONST inline constexpr
#endif

namespace bonsai {

// ---- model geometry (qwen35 / Qwen3.8-27B) ----
BONSAI_CONST uint32_t kLayers        = 64;     // main layers; blk.64 is the MTP layer
BONSAI_CONST uint32_t kAllBlocks     = 65;
BONSAI_CONST uint32_t kHidden        = 5120;
BONSAI_CONST uint32_t kFfn           = 17408;
BONSAI_CONST uint32_t kVocab         = 248320;
BONSAI_CONST float    kRmsEps        = 1e-6f;
BONSAI_CONST uint32_t kFullAttnEvery = 4;      // layer il is full attention iff il % 4 == 3

// full attention
BONSAI_CONST uint32_t kHeads         = 24;
BONSAI_CONST uint32_t kKvHeads       = 4;
BONSAI_CONST uint32_t kHeadDim       = 256;
BONSAI_CONST uint32_t kRotDim        = 64;     // partial rotary: first 64 dims, pairs (i, i+32)
BONSAI_CONST float    kRopeTheta     = 10000000.0f;
BONSAI_CONST uint32_t kQProjOut      = kHeads * kHeadDim * 2;   // 12288: per head [query(256) | gate(256)]
BONSAI_CONST uint32_t kKvDim         = kKvHeads * kHeadDim;     // 1024
BONSAI_CONST uint32_t kAttnOutIn     = kHeads * kHeadDim;       // 6144

// gated delta net (linear attention)
BONSAI_CONST uint32_t kGdnKHeads     = 16;
BONSAI_CONST uint32_t kGdnVHeads     = 48;
BONSAI_CONST uint32_t kGdnKDim       = 128;
BONSAI_CONST uint32_t kGdnVDim       = 128;
BONSAI_CONST uint32_t kGdnConvK      = 4;
BONSAI_CONST uint32_t kGdnQkvDim     = 2 * kGdnKHeads * kGdnKDim + kGdnVHeads * kGdnVDim;  // 10240
BONSAI_CONST uint32_t kGdnZDim       = kGdnVHeads * kGdnVDim;                              // 6144

// ---- PTQ1_0 weight format ----
BONSAI_CONST uint32_t kGroup         = 128;    // weights per scale group
BONSAI_CONST uint32_t kBlockBytes    = 28;     // 24 qs + 2 qh + fp16 scale

// ---- Hadamard rotation ----
BONSAI_CONST uint32_t kHadBlock      = 1024;   // normalized Sylvester-Walsh-Hadamard, block-diagonal along the input dim

// ---- hardware (Mac mini M6) ----
BONSAI_CONST uint32_t kGpuCores      = 12;
BONSAI_CONST uint32_t kMaxResidentThreadsPerCore = 3072;   // measured
BONSAI_CONST uint32_t kThreadgroupMem = 32768;
BONSAI_CONST uint32_t kCacheLine     = 128;

inline bool layerIsAttention(uint32_t il) { return (il % kFullAttnEvery) == (kFullAttnEvery - 1); }

}  // namespace bonsai
