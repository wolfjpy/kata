// Multi-token-prediction (MTP) head kernels. The MTP block (blk.64) is stored as Q8_0 (34-byte blocks: fp16 scale + 32 int8) and is not Hadamard-rotated.
#include "common.h"

// y[t][row] = sum_k W[row][k] * x[t][k] for T tokens (x fp32 [T][K]); one simdgroup per row, 8 rows per threadgroup.
template <int T>
kernel void k_gemv_q8(device const uchar* W [[buffer(0)]], device const float* x [[buffer(1)]], device float* y [[buffer(2)]], constant uint& K [[buffer(3)]], constant uint& N [[buffer(4)]],
                      uint tgid [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
    const uint row = tgid * 8 + sg; if (row >= N) return;
    const uint nb = K / 32u; device const uchar* rp = W + ulong(row) * ulong(nb) * 34ul;
    float acc[T]; for (int t = 0; t < T; t++) acc[t] = 0.0f;
    for (uint b = lane; b < nb; b += 32) {
        device const uchar* bp = rp + b * 34u; const float d = float(as_type<half>(*(device const ushort*)bp));
        for (int t = 0; t < T; t++) {
            device const float* xb = x + ulong(t) * K + b * 32u; float s = 0.0f;
            for (uint i = 0; i < 16; i++) { ushort w = *(device const ushort*)(bp + 2 + 2 * i); s += float(char(w & 0xffu)) * xb[2 * i] + float(char(w >> 8)) * xb[2 * i + 1]; }
            acc[t] += d * s;
        }
    }
    for (int t = 0; t < T; t++) { float v = simd_sum(acc[t]); if (lane == 0) y[ulong(t) * N + row] = v; }
}
template [[host_name("k_gemv_q8_t1")]] kernel void k_gemv_q8<1>(device const uchar*, device const float*, device float*, constant uint&, constant uint&, uint, uint, uint);
template [[host_name("k_gemv_q8_t2")]] kernel void k_gemv_q8<2>(device const uchar*, device const float*, device float*, constant uint&, constant uint&, uint, uint, uint);

// Q8_0 matrix [N][K] -> fp16 tile-major [N/64][K][64] (for the prefill-time catch-up GEMMs). Grid (K/128, N/64), 256 threads.
kernel void k_expand_q8(device const uchar* W [[buffer(0)]], device half* out [[buffer(1)]], constant uint& K [[buffer(2)]], uint2 tg [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {
    threadgroup half tile[128 * 64];
    const uint kb = tg.x, nt = tg.y, nb = K / 32u; const uint r = tid >> 2, q = tid & 3u;   // thread (row r, quarter q) -> one Q8 block of 32 elements
    device const uchar* bp = W + (ulong(nt * 64u + r) * ulong(nb) + kb * 4u + q) * 34ul; const float d = float(as_type<half>(*(device const ushort*)bp));
    for (uint i = 0; i < 32; i++) tile[(q * 32u + i) * 64u + r] = half(d * float(char(bp[2 + i])));
    threadgroup_barrier(mem_flags::mem_threadgroup);
    device half* o = out + (ulong(nt) * ulong(K) + ulong(kb) * 128ul) * 64ul;
    for (uint i = tid; i < 128u * 64u; i += 256u) o[i] = tile[i];
}

// rows: out[r] = RMSNorm(src[r]) * w   (plain, no rotation), grid R threadgroups of 256 threads, row length W
kernel void k_rmsnorm_rows(device const float* src [[buffer(0)]], device const float* w [[buffer(1)]], device float* out [[buffer(2)]], constant uint& W [[buffer(3)]],
                           uint r [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]], uint sg [[simdgroup_index_in_threadgroup]]) {
    threadgroup float part[8]; device const float* s = src + ulong(r) * W; device float* o = out + ulong(r) * W;
    float ss = 0.0f; for (uint i = tid; i < W; i += 256) ss += s[i] * s[i];
    ss = simd_sum(ss); if (lane == 0) part[sg] = ss;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float tot = 0.0f; for (uint k = 0; k < 8; k++) tot += part[k];
    const float inv = rsqrt(tot / float(W) + bonsai::kRmsEps);
    for (uint i = tid; i < W; i += 256) o[i] = s[i] * inv * w[i];
}
// MTP input: out[r] = [ RMSNorm(e[r]) * enorm | RMSNorm(h[r]) * hnorm ]  (10240 floats per row). Grid R, 256 threads.
kernel void k_mtp_cat(device const float* e [[buffer(0)]], device const float* h [[buffer(1)]], device const float* enorm [[buffer(2)]], device const float* hnorm [[buffer(3)]], device float* out [[buffer(4)]],
                      uint r [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]], uint sg [[simdgroup_index_in_threadgroup]]) {
    threadgroup float part[16]; const uint W = bonsai::kHidden;
    device const float* es = e + ulong(r) * W; device const float* hs = h + ulong(r) * W; device float* o = out + ulong(r) * 2 * W;
    float s1 = 0.0f, s2 = 0.0f; for (uint i = tid; i < W; i += 256) { s1 += es[i] * es[i]; s2 += hs[i] * hs[i]; }
    s1 = simd_sum(s1); s2 = simd_sum(s2); if (lane == 0) { part[sg] = s1; part[8 + sg] = s2; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float t1 = 0.0f, t2 = 0.0f; for (uint k = 0; k < 8; k++) { t1 += part[k]; t2 += part[8 + k]; }
    const float i1 = rsqrt(t1 / float(W) + bonsai::kRmsEps), i2 = rsqrt(t2 / float(W) + bonsai::kRmsEps);
    for (uint i = tid; i < W; i += 256) { o[i] = es[i] * i1 * enorm[i]; o[W + i] = hs[i] * i2 * hnorm[i]; }
}
kernel void k_swiglu(device const float* g [[buffer(0)]], device const float* u [[buffer(1)]], device float* out [[buffer(2)]], constant uint& n [[buffer(3)]], uint i [[thread_position_in_grid]]) { if (i < n) out[i] = silu(g[i]) * u[i]; }
kernel void k_add3(device const float* a [[buffer(0)]], device const float* b [[buffer(1)]], device float* out [[buffer(2)]], constant uint& n [[buffer(3)]], uint i [[thread_position_in_grid]]) { if (i < n) out[i] = a[i] + b[i]; }
kernel void k_half_to_float(device const half* src [[buffer(0)]], device float* dst [[buffer(1)]], constant uint& n [[buffer(2)]], uint i [[thread_position_in_grid]]) { if (i < n) dst[i] = float(src[i]); }
