// Embedding, RMSNorm + Hadamard, elementwise ops, argmax.
#include "common.h"

// Token embedding: dequantize one PTQ1_0 row (rotated basis), then h = sign * (H z). One threadgroup per 1024-block.
kernel void k_embed(device const uchar* W [[buffer(0)]], device const int* tok [[buffer(1)]], device const float* sg [[buffer(2)]], device float* out [[buffer(3)]],
                    uint tgid [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
    threadgroup float xs[1024];
    device const uchar* row = W + ulong(tok[0]) * ulong(kata::kHidden / kata::kGroup * kata::kBlockBytes);
    uint i = tid * 4; uint g = i >> 7; uint e = i & 127u;
    device const uchar* b = row + (tgid * 8 + g) * kata::kBlockBytes;
    float d = ptq1_scale(b);
    float4 v = float4(float(ptq1_elem(b, e)), float(ptq1_elem(b, e + 1)), float(ptq1_elem(b, e + 2)), float(ptq1_elem(b, e + 3))) * d;
    v = fwht1024(v, tid, lane, xs);
    uint o = tgid * 1024 + i;
    out[o] = v.x * sg[o]; out[o + 1] = v.y * sg[o + 1]; out[o + 2] = v.z * sg[o + 2]; out[o + 3] = v.w * sg[o + 3];
}

// Input transform of a ternary matmul: out = H_blockdiag( sign (.) f(src) ).  One threadgroup per 1024-block.
//  mode 0: f = RMSNorm(src) with weight folded into gs (= gamma (.) sign)
//  mode 1: f = silu(src) * src2   (SwiGLU),   gs = sign
//  mode 2: f = src permuted from tiled to grouped GDN head order, gs = sign
//  mode 3: f = src, gs = sign
kernel void k_rms_fwht(device const float* src [[buffer(0)]], device const float* src2 [[buffer(1)]], device const float* gs [[buffer(2)]], device float* out [[buffer(3)]],
                       constant uint& W [[buffer(4)]], constant uint& mode [[buffer(5)]],
                       uint tgid [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]], uint sg [[simdgroup_index_in_threadgroup]]) {
    threadgroup float xs[1024]; threadgroup float part[8];
    float inv = 1.0f;
    if (mode == 0) {
        float ss = 0.0f;
        for (uint i = tid; i < W; i += 256) { float v = src[i]; ss += v * v; }
        ss = simd_sum(ss);
        if (lane == 0) part[sg] = ss;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float tot = 0.0f; for (uint s = 0; s < 8; s++) tot += part[s];
        inv = rsqrt(tot / float(W) + kata::kRmsEps);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    uint base = tgid * 1024 + tid * 4;
    float vals[4];
    for (uint r = 0; r < 4; r++) {
        uint idx = base + r; float v;
        if (mode == 0) v = src[idx] * inv * gs[idx];
        else if (mode == 1) { float g = src[idx]; v = silu(g) * src2[idx] * gs[idx]; }
        else if (mode == 2) { uint h = idx & 127u; uint t = idx >> 7; uint rr = t % 3u; uint k = t / 3u; v = src[h + 128u * (k + 16u * rr)] * gs[idx]; }
        else v = src[idx] * gs[idx];
        vals[r] = v;
    }
    float4 o = fwht1024(float4(vals[0], vals[1], vals[2], vals[3]), tid, lane, xs);
    out[base] = o.x; out[base + 1] = o.y; out[base + 2] = o.z; out[base + 3] = o.w;
}

kernel void k_add(device float* x [[buffer(0)]], device const float* y [[buffer(1)]], constant uint& n [[buffer(2)]], uint i [[thread_position_in_grid]]) { if (i < n) x[i] += y[i]; }
kernel void k_copy_f32(device const float* src [[buffer(0)]], device float* dst [[buffer(1)]], constant uint& n [[buffer(2)]], uint i [[thread_position_in_grid]]) { if (i < n) dst[i] = src[i]; }

// Greedy argmax over n floats; lowest index wins ties. One threadgroup of 1024 threads.
kernel void k_argmax(device const float* x [[buffer(0)]], device int* result [[buffer(1)]], constant uint& n [[buffer(2)]], uint tid [[thread_index_in_threadgroup]]) {
    threadgroup float bv[1024]; threadgroup int bi[1024];
    float best = -INFINITY; int bidx = 0x7fffffff;
    for (uint i = tid; i < n; i += 1024) { float v = x[i]; if (v > best) { best = v; bidx = int(i); } }
    bv[tid] = best; bi[tid] = bidx; threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = 512; s > 0; s >>= 1) {
        if (tid < s) { float ov = bv[tid + s]; int oi = bi[tid + s]; if (ov > bv[tid] || (ov == bv[tid] && oi < bi[tid])) { bv[tid] = ov; bi[tid] = oi; } }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid == 0) result[0] = bi[0];
}

// Plain RMSNorm (no rotation): out = src * rsqrt(mean(src^2) + eps) * w.  One threadgroup of 256 threads.
kernel void k_rmsnorm(device const float* src [[buffer(0)]], device const float* w [[buffer(1)]], device float* out [[buffer(2)]], constant uint& W [[buffer(3)]],
                      uint tid [[thread_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]], uint sg [[simdgroup_index_in_threadgroup]]) {
    threadgroup float part[8];
    float ss = 0.0f;
    for (uint i = tid; i < W; i += 256) { float v = src[i]; ss += v * v; }
    ss = simd_sum(ss);
    if (lane == 0) part[sg] = ss;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float tot = 0.0f; for (uint s = 0; s < 8; s++) tot += part[s];
    float inv = rsqrt(tot / float(W) + kata::kRmsEps);
    for (uint i = tid; i < W; i += 256) out[i] = src[i] * inv * w[i];
}

// Fused residual add + RMSNorm + sign + FWHT (+ optional plain normalized copy for the unrotated alpha/beta projections).
//   xnew = x + y (each threadgroup writes its own 1024-block);  out = H( gs (.) rmsnorm_sumsq(xnew) * xnew );  hout = xnew * inv * gamma (if hout != null via flag)
kernel void k_add_rms_fwht(device const float* x [[buffer(0)]], device const float* y [[buffer(1)]], device const float* gs [[buffer(2)]], device float* out [[buffer(3)]],
                           device float* xnew [[buffer(4)]], device const float* gamma [[buffer(5)]], device float* hout [[buffer(6)]], constant uint& flags [[buffer(7)]],
                           uint tgid [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]], uint sg [[simdgroup_index_in_threadgroup]]) {
    // flags: bit0 = add y (else x only), bit1 = write plain h
    threadgroup float xs[1024]; threadgroup float part[8];
    const uint W = kata::kHidden;
    float ss = 0.0f;
    for (uint i = tid; i < W; i += 256) { float v = x[i] + ((flags & 1u) ? y[i] : 0.0f); ss += v * v; }
    ss = simd_sum(ss);
    if (lane == 0) part[sg] = ss;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float tot = 0.0f; for (uint s = 0; s < 8; s++) tot += part[s];
    float inv = rsqrt(tot / float(W) + kata::kRmsEps);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint base = tgid * 1024 + tid * 4; float vals[4];
    for (uint r = 0; r < 4; r++) {
        uint idx = base + r; float v = x[idx] + ((flags & 1u) ? y[idx] : 0.0f);
        xnew[idx] = v; vals[r] = v * inv * gs[idx];
        if (flags & 2u) hout[idx] = v * inv * gamma[idx];
    }
    float4 o = fwht1024(float4(vals[0], vals[1], vals[2], vals[3]), tid, lane, xs);
    out[base] = o.x; out[base + 1] = o.y; out[base + 2] = o.z; out[base + 3] = o.w;
}
