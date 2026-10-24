// Full-attention decode step (one token): per-head q/k RMSNorm, partial RoPE, KV-cache write, softmax attention, sigmoid output gate.
#include "common.h"

// 32 threadgroups x 256 threads: [0,24) query heads, [24,28) key heads, [28,32) value heads.
kernel void k_attn_prep(device const float* qfull [[buffer(0)]], device const float* kproj [[buffer(1)]], device const float* vproj [[buffer(2)]],
                        device const float* qnorm [[buffer(3)]], device const float* knorm [[buffer(4)]],
                        device float* qout [[buffer(5)]], device float* gate_out [[buffer(6)]],
                        device half* kcache [[buffer(7)]], device half* vcache [[buffer(8)]], constant uint& pos [[buffer(9)]],
                        uint h [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]], uint sg [[simdgroup_index_in_threadgroup]]) {
    threadgroup float xs[256]; threadgroup float part[8];
    if (h >= 28) {   // value head: copy to cache
        uint vh = h - 28; vcache[ulong(pos) * bonsai::kKvDim + vh * 256 + tid] = half(vproj[vh * 256 + tid]);
        return;
    }
    float v; device const float* w;
    if (h < 24) { v = qfull[h * 512 + tid]; gate_out[h * 256 + tid] = qfull[h * 512 + 256 + tid]; w = qnorm; }
    else { v = kproj[(h - 24) * 256 + tid]; w = knorm; }
    float ss = simd_sum(v * v);
    if (lane == 0) part[sg] = ss;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float tot = 0.0f; for (uint s = 0; s < 8; s++) tot += part[s];
    xs[tid] = v * rsqrt(tot / 256.0f + bonsai::kRmsEps) * w[tid];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float outv = xs[tid];
    if (tid < 32) {          // partial RoPE: pair (i, i+32)
        float theta = float(pos) * pow(bonsai::kRopeTheta, -float(tid) / 32.0f);
        float c = cos(theta), s = sin(theta);
        float a = xs[tid], b = xs[tid + 32];
        outv = a * c - b * s;
        float outv2 = a * s + b * c;
        if (h < 24) qout[h * 256 + tid + 32] = outv2; else kcache[ulong(pos) * bonsai::kKvDim + (h - 24) * 256 + tid + 32] = half(outv2);
    }
    if (tid >= 32 && tid < 64) { /* handled by the lower half above */ }
    else if (tid < 32 || tid >= 64) {
        if (h < 24) qout[h * 256 + tid] = outv; else kcache[ulong(pos) * bonsai::kKvDim + (h - 24) * 256 + tid] = half(outv);
    }
}

// One threadgroup (256 threads, thread = head dimension) per query head; streaming softmax over positions [0, pos].
kernel void k_attn_decode(device const float* q [[buffer(0)]], device const float* gate [[buffer(1)]], device const half* kcache [[buffer(2)]], device const half* vcache [[buffer(3)]],
                          device float* out [[buffer(4)]], constant uint& pos [[buffer(5)]],
                          uint h [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]], uint sg [[simdgroup_index_in_threadgroup]]) {
    threadgroup float part[8];
    const uint kvh = h / (bonsai::kHeads / bonsai::kKvHeads);
    const float scale = 0.0625f;   // 1/sqrt(256)
    float qd = q[h * 256 + tid];
    float m = -INFINITY, l = 0.0f, acc = 0.0f;
    for (uint j = 0; j <= pos; j++) {
        float kd = float(kcache[ulong(j) * bonsai::kKvDim + kvh * 256 + tid]);
        float p = simd_sum(qd * kd);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (lane == 0) part[sg] = p;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float s = 0.0f; for (uint i = 0; i < 8; i++) s += part[i];
        s *= scale;
        float mn = max(m, s), a = exp(m - mn), e = exp(s - mn);
        l = l * a + e;
        acc = acc * a + e * float(vcache[ulong(j) * bonsai::kKvDim + kvh * 256 + tid]);
        m = mn;
    }
    out[h * 256 + tid] = (acc / l) * sigmoidf(gate[h * 256 + tid]);
}
