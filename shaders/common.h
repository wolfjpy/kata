// Shared device helpers: PTQ1_0 decoding and the 1024-point Walsh-Hadamard transform.
#pragma once
#include <metal_stdlib>
#include "kata_m6.h"
using namespace metal;

inline uint pow3(uint n) { return n == 0 ? 1u : n == 1 ? 3u : n == 2 ? 9u : n == 3 ? 27u : 81u; }

// A packed byte is a base-3 fraction of 256; digit n (0 = most significant) is ((byte*3^n mod 256) * 3) >> 8.
inline int ptq1_trit(uint b, uint n) { uint q = (b * pow3(n)) & 0xffu; return int((q * 3u) >> 8) - 1; }

// Element e (0..127) of a 28-byte PTQ1_0 block -> trit in {-1,0,+1}. Order follows docs/reference/prism_ptq1_0_quant_dequant.c.
inline int ptq1_elem(device const uchar* blk, uint e) {
    if (e < 80u)  return ptq1_trit(uint(blk[e & 15u]), e >> 4);
    if (e < 120u) { uint e2 = e - 80u; return ptq1_trit(uint(blk[16u + (e2 & 7u)]), e2 >> 3); }
    uint e3 = e - 120u; return ptq1_trit(uint(blk[24u + (e3 & 1u)]), e3 >> 1);
}
inline float ptq1_scale(device const uchar* blk) { return float(as_type<half>(ushort(uint(blk[26]) | (uint(blk[27]) << 8)))); }

// Normalized natural-order FWHT of 1024 values held as one float4 per thread (256 threads; element i = tid*4 + r).
// Stages: bits 0-1 in registers, bits 2-6 via simd_shuffle_xor, bits 7-9 via a threadgroup transpose. Result scaled by 1/32.
inline float4 fwht1024(float4 v, uint tid, uint lane, threadgroup float* xs) {
    v = float4(v.x + v.y, v.x - v.y, v.z + v.w, v.z - v.w);
    v = float4(v.x + v.z, v.y + v.w, v.x - v.z, v.y - v.w);
    for (uint k = 0; k < 5; k++) {
        uint m = 1u << k;
        float4 p = float4(simd_shuffle_xor(v.x, m), simd_shuffle_xor(v.y, m), simd_shuffle_xor(v.z, m), simd_shuffle_xor(v.w, m));
        v = (lane & m) ? (p - v) : (v + p);
    }
    xs[tid * 4 + 0] = v.x; xs[tid * 4 + 1] = v.y; xs[tid * 4 + 2] = v.z; xs[tid * 4 + 3] = v.w;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < 128) {
        float e[8];
        for (uint j = 0; j < 8; j++) e[j] = xs[tid + 128 * j];
        for (uint s = 0; s < 3; s++) { uint h = 1u << s; for (uint j = 0; j < 8; j++) if (!(j & h)) { float a = e[j], b = e[j | h]; e[j] = a + b; e[j | h] = a - b; } }
        for (uint j = 0; j < 8; j++) xs[tid + 128 * j] = e[j];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float4 r = float4(xs[tid * 4 + 0], xs[tid * 4 + 1], xs[tid * 4 + 2], xs[tid * 4 + 3]) * 0.03125f;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    return r;
}

inline float silu(float x) { return x / (1.0f + exp(-x)); }
inline float sigmoidf(float x) { return 1.0f / (1.0f + exp(-x)); }
