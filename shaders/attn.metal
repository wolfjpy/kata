// Full-attention decode step (one token): per-head q/k RMSNorm, partial RoPE, KV-cache write, softmax attention, sigmoid output gate.
#include "common.h"

// 32 threadgroups x 256 threads: [0,24) query heads, [24,28) key heads, [28,32) value heads.
kernel void k_attn_prep(device const float* qfull [[buffer(0)]], device const float* kproj [[buffer(1)]], device const float* vproj [[buffer(2)]],
                        device const float* qnorm [[buffer(3)]], device const float* knorm [[buffer(4)]],
                        device float* qout [[buffer(5)]], device float* gate_out [[buffer(6)]],
                        device half* kcache [[buffer(7)]], device half* vcache [[buffer(8)]], constant uint& pos [[buffer(9)]], constant uint& cacheOff [[buffer(10)]],
                        uint h [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]], uint sg [[simdgroup_index_in_threadgroup]]) {
    threadgroup float xs[256]; threadgroup float part[8];
    if (h >= 28) {   // value head: copy to cache
        uint vh = h - 28; vcache[ulong(pos - cacheOff) * kata::kKvDim + vh * 256 + tid] = half(vproj[vh * 256 + tid]);
        return;
    }
    float v; device const float* w;
    if (h < 24) { v = qfull[h * 512 + tid]; gate_out[h * 256 + tid] = qfull[h * 512 + 256 + tid]; w = qnorm; }
    else { v = kproj[(h - 24) * 256 + tid]; w = knorm; }
    float ss = simd_sum(v * v);
    if (lane == 0) part[sg] = ss;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float tot = 0.0f; for (uint s = 0; s < 8; s++) tot += part[s];
    xs[tid] = v * rsqrt(tot / 256.0f + kata::kRmsEps) * w[tid];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float outv = xs[tid];
    if (tid < 32) {          // partial RoPE: pair (i, i+32)
        float theta = float(pos) * pow(kata::kRopeTheta, -float(tid) / 32.0f);
        float c = cos(theta), s = sin(theta);
        float a = xs[tid], b = xs[tid + 32];
        outv = a * c - b * s;
        float outv2 = a * s + b * c;
        if (h < 24) qout[h * 256 + tid + 32] = outv2; else kcache[ulong(pos - cacheOff) * kata::kKvDim + (h - 24) * 256 + tid + 32] = half(outv2);
    }
    if (tid >= 32 && tid < 64) { /* handled by the lower half above */ }
    else if (tid < 32 || tid >= 64) {
        if (h < 24) qout[h * 256 + tid] = outv; else kcache[ulong(pos - cacheOff) * kata::kKvDim + (h - 24) * 256 + tid] = half(outv);
    }
}

// One threadgroup (256 threads, thread = head dimension) per query head; streaming softmax over positions [0, pos].
kernel void k_attn_decode(device const float* q [[buffer(0)]], device const float* gate [[buffer(1)]], device const half* kcache [[buffer(2)]], device const half* vcache [[buffer(3)]],
                          device float* out [[buffer(4)]], constant uint& pos [[buffer(5)]],
                          uint h [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]], uint sg [[simdgroup_index_in_threadgroup]]) {
    threadgroup float part[8];
    const uint kvh = h / (kata::kHeads / kata::kKvHeads);
    const float scale = 0.0625f;   // 1/sqrt(256)
    float qd = q[h * 256 + tid];
    float m = -INFINITY, l = 0.0f, acc = 0.0f;
    for (uint j = 0; j <= pos; j++) {
        float kd = float(kcache[ulong(j) * kata::kKvDim + kvh * 256 + tid]);
        float p = simd_sum(qd * kd);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (lane == 0) part[sg] = p;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float s = 0.0f; for (uint i = 0; i < 8; i++) s += part[i];
        s *= scale;
        float mn = max(m, s), a = exp(m - mn), e = exp(s - mn);
        l = l * a + e;
        acc = acc * a + e * float(vcache[ulong(j) * kata::kKvDim + kvh * 256 + tid]);
        m = mn;
    }
    out[h * 256 + tid] = (acc / l) * sigmoidf(gate[h * 256 + tid]);
}

// ---- flash-decoding: split the positions over many threadgroups; GQA group of 6 query heads shares each K/V load ----
// Grid (splits, 4 kv heads), 256 threads = 8 simdgroups. Each simdgroup walks its own contiguous slice of positions; lane owns 8 consecutive dims.
kernel void k_attn_split(device const float* q [[buffer(0)]], device const half* kcache [[buffer(1)]], device const half* vcache [[buffer(2)]],
                         device float* pm [[buffer(3)]], device float* pl [[buffer(4)]], device float* pacc [[buffer(5)]],
                         constant uint& npos [[buffer(6)]],   // number of valid positions (pos + 1)
                         uint2 tg [[threadgroup_position_in_grid]], uint2 ng [[threadgroups_per_grid]], uint tid [[thread_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]], uint sg [[simdgroup_index_in_threadgroup]]) {
    threadgroup float tm[8][6]; threadgroup float tl[8][6]; threadgroup float tacc[8][3][256];
    const uint split = tg.x, kvh = tg.y, S = ng.x;
    const uint per = (npos + S - 1) / S, lo = min(split * per, npos), hi = min(lo + per, npos);
    const uint sper = (hi - lo + 7) / 8, slo = min(lo + sg * sper, hi), shi = min(slo + sper, hi);
    float qv[6][8];
    for (uint h = 0; h < 6; h++) for (uint d = 0; d < 8; d++) qv[h][d] = q[(kvh * 6 + h) * 256 + lane * 8 + d] * 0.0625f;   // fold 1/sqrt(256)
    float m[6], l[6], acc[6][8];
    for (uint h = 0; h < 6; h++) { m[h] = -INFINITY; l[h] = 0.0f; for (uint d = 0; d < 8; d++) acc[h][d] = 0.0f; }
    for (uint j = slo; j < shi; j++) {
        device const half* kp = kcache + ulong(j) * kata::kKvDim + kvh * 256 + lane * 8;
        device const half* vp = vcache + ulong(j) * kata::kKvDim + kvh * 256 + lane * 8;
        half4 k0 = *(device const half4*)kp, k1 = *(device const half4*)(kp + 4), v0 = *(device const half4*)vp, v1 = *(device const half4*)(vp + 4);
        float kf[8] = {float(k0.x), float(k0.y), float(k0.z), float(k0.w), float(k1.x), float(k1.y), float(k1.z), float(k1.w)};
        float vf[8] = {float(v0.x), float(v0.y), float(v0.z), float(v0.w), float(v1.x), float(v1.y), float(v1.z), float(v1.w)};
        for (uint h = 0; h < 6; h++) {
            float p = 0.0f; for (uint d = 0; d < 8; d++) p += qv[h][d] * kf[d];
            float s = simd_sum(p);
            float mn = max(m[h], s), a = exp(m[h] - mn), e = exp(s - mn);
            l[h] = l[h] * a + e; for (uint d = 0; d < 8; d++) acc[h][d] = acc[h][d] * a + e * vf[d];
            m[h] = mn;
        }
    }
    if (lane == 0) { for (uint h = 0; h < 6; h++) { tm[sg][h] = m[h]; tl[sg][h] = l[h]; } }
    // merge the 8 simdgroups in two rounds of 3 heads (threadgroup memory limit); thread tid handles dimension tid
    for (uint hg = 0; hg < 2; hg++) {
        for (uint hh = 0; hh < 3; hh++) for (uint d = 0; d < 8; d++) tacc[sg][hh][lane * 8 + d] = acc[hg * 3 + hh][d];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint hh = 0; hh < 3; hh++) {
            const uint h = hg * 3 + hh;
            float M = -INFINITY; for (uint s = 0; s < 8; s++) M = max(M, tm[s][h]);
            float L = 0.0f, A = 0.0f;
            for (uint s = 0; s < 8; s++) { float w = (tm[s][h] == -INFINITY) ? 0.0f : exp(tm[s][h] - M); L += tl[s][h] * w; A += tacc[s][hh][tid] * w; }
            const uint gh = kvh * 6 + h;
            pacc[(ulong(gh) * S + split) * 256 + tid] = A;
            if (tid == 0) { pm[gh * S + split] = M; pl[gh * S + split] = L; }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}

// Merge the splits of one query head and apply the sigmoid output gate. Grid 24 threadgroups x 256 threads.
kernel void k_attn_combine(device const float* pm [[buffer(0)]], device const float* pl [[buffer(1)]], device const float* pacc [[buffer(2)]], device const float* gate [[buffer(3)]],
                           device float* out [[buffer(4)]], uint h [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]], constant uint& S [[buffer(5)]]) {
    float M = -INFINITY; for (uint s = 0; s < S; s++) M = max(M, pm[h * S + s]);
    float L = 0.0f, A = 0.0f;
    for (uint s = 0; s < S; s++) { float mm = pm[h * S + s]; float w = (mm == -INFINITY) ? 0.0f : exp(mm - M); L += pl[h * S + s] * w; A += pacc[(ulong(h) * S + s) * 256 + tid] * w; }
    out[h * 256 + tid] = (A / L) * sigmoidf(gate[h * 256 + tid]);
}
