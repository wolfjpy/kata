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
    const float inv = rsqrt(tot / float(W) + kata::kRmsEps);
    for (uint i = tid; i < W; i += 256) o[i] = s[i] * inv * w[i];
}
// MTP input: out[r] = [ RMSNorm(e[r]) * enorm | RMSNorm(h[r]) * hnorm ]  (10240 floats per row). Grid R, 256 threads.
kernel void k_mtp_cat(device const float* e [[buffer(0)]], device const float* h [[buffer(1)]], device const float* enorm [[buffer(2)]], device const float* hnorm [[buffer(3)]], device float* out [[buffer(4)]],
                      uint r [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]], uint sg [[simdgroup_index_in_threadgroup]]) {
    threadgroup float part[16]; const uint W = kata::kHidden;
    device const float* es = e + ulong(r) * W; device const float* hs = h + ulong(r) * W; device float* o = out + ulong(r) * 2 * W;
    float s1 = 0.0f, s2 = 0.0f; for (uint i = tid; i < W; i += 256) { s1 += es[i] * es[i]; s2 += hs[i] * hs[i]; }
    s1 = simd_sum(s1); s2 = simd_sum(s2); if (lane == 0) { part[sg] = s1; part[8 + sg] = s2; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float t1 = 0.0f, t2 = 0.0f; for (uint k = 0; k < 8; k++) { t1 += part[k]; t2 += part[8 + k]; }
    const float i1 = rsqrt(t1 / float(W) + kata::kRmsEps), i2 = rsqrt(t2 / float(W) + kata::kRmsEps);
    for (uint i = tid; i < W; i += 256) { o[i] = es[i] * i1 * enorm[i]; o[W + i] = hs[i] * i2 * hnorm[i]; }
}
kernel void k_swiglu(device const float* g [[buffer(0)]], device const float* u [[buffer(1)]], device float* out [[buffer(2)]], constant uint& n [[buffer(3)]], uint i [[thread_position_in_grid]]) { if (i < n) out[i] = silu(g[i]) * u[i]; }
kernel void k_add3(device const float* a [[buffer(0)]], device const float* b [[buffer(1)]], device float* out [[buffer(2)]], constant uint& n [[buffer(3)]], uint i [[thread_position_in_grid]]) { if (i < n) out[i] = a[i] + b[i]; }
kernel void k_half_to_float(device const half* src [[buffer(0)]], device float* dst [[buffer(1)]], constant uint& n [[buffer(2)]], uint i [[thread_position_in_grid]]) { if (i < n) dst[i] = float(src[i]); }

// =====================================================================================================================
// Verification pass (two tokens through the main model in one go).
// ---- PTQ1_0 GEMV for T tokens, fp16 activations x[T][K]; same lane mapping as k_gemv_v1 so each weight byte is read once for all tokens ----
template <int NR, int NSG, int T>
kernel void k_gemv_v1h(device const uchar* W [[buffer(0)]], device const half* x [[buffer(1)]], device float* y [[buffer(2)]], constant uint& K [[buffer(3)]], constant uint& N [[buffer(4)]],
                       uint tgid [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
    const uint row0 = (tgid * NSG + sg) * NR; if (row0 >= N) return;
    const uint nblk = K / 128u, ix = lane >> 3, it = lane & 7u;
    float acc[T][NR]; for (int t = 0; t < T; t++) for (int r = 0; r < NR; r++) acc[t][r] = 0.0f;
    device const uchar* rowp[NR]; for (int r = 0; r < NR; r++) rowp[r] = W + ulong(min(row0 + uint(r), N - 1u)) * ulong(nblk) * 28ul;
    const uint qn = it >> 1; const float p3q = qn == 0 ? 1.0f : qn == 1 ? 3.0f : qn == 2 ? 9.0f : 27.0f;
    for (uint b = ix; b < nblk; b += 4) {
        float c[T][15], yq[T], sumy[T];
        for (int t = 0; t < T; t++) {
            device const half* xb = x + ulong(t) * K + b * 128u; float yl[16]; float sy = 0.0f;
            for (uint k = 0; k < 2; k++) for (uint n = 0; n < 5; n++) { float v = float(xb[n * 16 + 2 * it + k]); yl[5 * k + n] = v; sy += v; }
            for (uint n = 0; n < 5; n++) { float v = float(xb[80 + n * 8 + it]); yl[10 + n] = v; sy += v; }
            yq[t] = float(xb[120 + qn * 2 + (it & 1u)]); sy += yq[t]; sumy[t] = sy;
            for (int g = 0; g < 3; g++) { const float* p = yl + 5 * g; c[t][5*g+0] = p[0] - 3.0f * p[1]; c[t][5*g+1] = p[1] - 3.0f * p[2]; c[t][5*g+2] = p[2] - 3.0f * p[3]; c[t][5*g+3] = p[3] - 3.0f * p[4]; c[t][5*g+4] = p[4]; }
        }
        for (int r = 0; r < NR; r++) {
            device const uchar* bp = rowp[r] + b * 28u;
            ushort w01 = *(device const ushort*)(bp + 2 * it);
            float u0 = float(w01 & 0xffu) * (1.0f / 256.0f), u1 = float(w01 >> 8) * (1.0f / 256.0f), u2 = float(bp[16 + it]) * (1.0f / 256.0f), uq = float(bp[24 + (it & 1u)]) * (1.0f / 256.0f);
            const float g0[5] = {floor(3.0f*u0), floor(9.0f*u0), floor(27.0f*u0), floor(81.0f*u0), floor(243.0f*u0)};
            const float g1[5] = {floor(3.0f*u1), floor(9.0f*u1), floor(27.0f*u1), floor(81.0f*u1), floor(243.0f*u1)};
            const float g2[5] = {floor(3.0f*u2), floor(9.0f*u2), floor(27.0f*u2), floor(81.0f*u2), floor(243.0f*u2)};
            const float gq = floor(3.0f * p3q * uq) - 3.0f * floor(p3q * uq);
            const float d = float(as_type<half>(*(device const ushort*)(bp + 26)));
            for (int t = 0; t < T; t++) {
                float s = 0.0f;
                for (int k = 0; k < 5; k++) s += g0[k] * c[t][k] + g1[k] * c[t][5 + k] + g2[k] * c[t][10 + k];
                s += gq * yq[t];
                acc[t][r] += d * (s - sumy[t]);
            }
        }
    }
    for (int t = 0; t < T; t++) for (int r = 0; r < NR; r++) { float v = simd_sum(acc[t][r]); if (lane == 0 && row0 + uint(r) < N) y[ulong(t) * N + row0 + uint(r)] = v; }
}
#define INSTH(NR, NSG, T) template [[host_name("k_gemv_v1h_t" #T "_nr" #NR "_sg" #NSG)]] kernel void k_gemv_v1h<NR, NSG, T>(device const uchar*, device const half*, device float*, constant uint&, constant uint&, uint, uint, uint);
INSTH(4, 4, 2) INSTH(2, 4, 2) INSTH(1, 4, 2) INSTH(2, 8, 2) INSTH(4, 2, 2) INSTH(3, 4, 2) INSTH(8, 4, 2) INSTH(1, 8, 2) INSTH(4, 4, 1) INSTH(2, 4, 1) INSTH(4, 4, 3)

// ---- GDN recurrence for T=2 tokens that leaves the INPUT state untouched and writes the state after token 0 to stateA and after token 1 to stateB ----
kernel void k_gdn_rec_v2(device const float* conv [[buffer(0)]], device const float* qk [[buffer(1)]], device const float* gb [[buffer(2)]], device const float* stateIn [[buffer(3)]],
                         device float* stateA [[buffer(4)]], device float* stateB [[buffer(5)]], device float* raw [[buffer(6)]], uint2 tg [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]]) {
    const uint hv = tg.x, cg = tg.y, hk = hv % kata::kGdnKHeads, part = lane & 3u, v = cg * 8u + (lane >> 2);
    const ulong off = ulong(hv) * 16384ul + ulong(part) * 32ul * 128ul + v;
    float st[32];
    for (uint k = 0; k < 32; k++) st[k] = stateIn[off + k * 128u];
    for (uint t = 0; t < 2; t++) {
        device const float4* qp = (device const float4*)(qk + ulong(t) * 4096ul + hk * 128 + part * 32);
        device const float4* kp = (device const float4*)(qk + ulong(t) * 4096ul + 2048ul + hk * 128 + part * 32);
        const float ge = gb[t * 96u + hv], beta = gb[t * 96u + 48u + hv];
        float skv = 0.0f;
        for (uint k4 = 0; k4 < 8; k4++) { float4 kk = kp[k4]; st[4*k4] *= ge; st[4*k4+1] *= ge; st[4*k4+2] *= ge; st[4*k4+3] *= ge; skv += st[4*k4] * kk.x + st[4*k4+1] * kk.y + st[4*k4+2] * kk.z + st[4*k4+3] * kk.w; }
        skv += simd_shuffle_xor(skv, 1); skv += simd_shuffle_xor(skv, 2);
        const float d = (conv[ulong(t) * kata::kGdnQkvDim + 4096u + hv * 128 + v] - skv) * beta;
        float y = 0.0f;
        for (uint k4 = 0; k4 < 8; k4++) { float4 kk = kp[k4], qq = qp[k4]; st[4*k4] += kk.x * d; st[4*k4+1] += kk.y * d; st[4*k4+2] += kk.z * d; st[4*k4+3] += kk.w * d; y += st[4*k4] * qq.x + st[4*k4+1] * qq.y + st[4*k4+2] * qq.z + st[4*k4+3] * qq.w; }
        y += simd_shuffle_xor(y, 1); y += simd_shuffle_xor(y, 2);
        if (part == 0) raw[ulong(t) * 6144ul + hv * 128 + v] = y * 0.08838834764831845f;
        device float* so = (t == 0) ? stateA : stateB;
        for (uint k = 0; k < 32; k++) so[off + k * 128u] = st[k];
    }
}
// causal conv for the two verified tokens (reads the old conv state; writes the state after token 0 to newA and after token 1 to newB)
kernel void k_gdn_conv_v2(device const float* qkv [[buffer(0)]], device const float* old [[buffer(1)]], device const float* w [[buffer(2)]], device float* out [[buffer(3)]],
                          device float* newA [[buffer(4)]], device float* newB [[buffer(5)]], uint c [[thread_position_in_grid]]) {
    const uint C = kata::kGdnQkvDim; if (c >= C) return;
    const float o0 = old[c], o1 = old[C + c], o2 = old[2 * C + c], x0 = qkv[c], x1 = qkv[C + c];
    const float w0 = w[4 * c], w1 = w[4 * c + 1], w2 = w[4 * c + 2], w3 = w[4 * c + 3];
    out[c] = silu(w0 * o0 + w1 * o1 + w2 * o2 + w3 * x0);
    out[C + c] = silu(w0 * o1 + w1 * o2 + w2 * x0 + w3 * x1);
    newA[c] = o1; newA[C + c] = o2; newA[2 * C + c] = x0;
    newB[c] = o2; newB[C + c] = x0; newB[2 * C + c] = x1;
}

// ---- prefill-time MTP K/V pass helpers (fp16 outputs for the matrix-unit GEMMs) ----
// out[r] = [ RMSNorm(e[r]) * enorm | RMSNorm(h[r]) * hnorm ] as half (10240 per row). Grid R, 256 threads.
kernel void k_mtp_cat_h(device const float* e [[buffer(0)]], device const float* h [[buffer(1)]], device const float* enorm [[buffer(2)]], device const float* hnorm [[buffer(3)]], device half* out [[buffer(4)]],
                        uint r [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]], uint sg [[simdgroup_index_in_threadgroup]]) {
    threadgroup float part[16]; const uint W = kata::kHidden;
    device const float* es = e + ulong(r) * W; device const float* hs = h + ulong(r) * W; device half* o = out + ulong(r) * 2 * W;
    float s1 = 0.0f, s2 = 0.0f; for (uint i = tid; i < W; i += 256) { s1 += es[i] * es[i]; s2 += hs[i] * hs[i]; }
    s1 = simd_sum(s1); s2 = simd_sum(s2); if (lane == 0) { part[sg] = s1; part[8 + sg] = s2; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float t1 = 0.0f, t2 = 0.0f; for (uint k = 0; k < 8; k++) { t1 += part[k]; t2 += part[8 + k]; }
    const float i1 = rsqrt(t1 / float(W) + kata::kRmsEps), i2 = rsqrt(t2 / float(W) + kata::kRmsEps);
    for (uint i = tid; i < W; i += 256) { o[i] = half(es[i] * i1 * enorm[i]); o[W + i] = half(hs[i] * i2 * hnorm[i]); }
}
kernel void k_rmsnorm_rows_h(device const float* src [[buffer(0)]], device const float* w [[buffer(1)]], device half* out [[buffer(2)]], constant uint& W [[buffer(3)]],
                             uint r [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]], uint sg [[simdgroup_index_in_threadgroup]]) {
    threadgroup float part[8]; device const float* s = src + ulong(r) * W; device half* o = out + ulong(r) * W;
    float ss = 0.0f; for (uint i = tid; i < W; i += 256) ss += s[i] * s[i];
    ss = simd_sum(ss); if (lane == 0) part[sg] = ss;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float tot = 0.0f; for (uint k = 0; k < 8; k++) tot += part[k];
    const float inv = rsqrt(tot / float(W) + kata::kRmsEps);
    for (uint i = tid; i < W; i += 256) o[i] = half(s[i] * inv * w[i]);
}
// half hidden rows -> float rows (n elements)

// ---- two-token GEMV (verification pass): trits come from a 256-entry threadgroup table with 16-byte entries (byte -> 5 half trits, one read per weight byte);
// the two tokens' activations are packed as half2 (interleaved [K] of half2) so one half2 FMA serves both tokens; partial sums flush to fp32 after every 5 terms. ----
template <int NR, int NSG>
kernel void k_gemv_lut2(device const uchar* W [[buffer(0)]], device const half2* x [[buffer(1)]], device float* y [[buffer(2)]], constant uint& K [[buffer(3)]], constant uint& N [[buffer(4)]],
                         uint tgid [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]], uint tid [[thread_index_in_threadgroup]]) {
    threadgroup uint4 lutv[256];
    for (uint b = tid; b < 256u; b += NSG * 32u) {
        uint p3[6] = {1u, 3u, 9u, 27u, 81u, 243u}; half t[8];
        for (uint n = 0; n < 5; n++) t[n] = half(float(int((b * p3[n + 1]) >> 8) - 3 * int((b * p3[n]) >> 8) - 1)); t[5] = 0.0h; t[6] = 0.0h; t[7] = 0.0h;
        lutv[b] = uint4(as_type<uint>(half2(t[0], t[1])), as_type<uint>(half2(t[2], t[3])), as_type<uint>(half2(t[4], t[5])), as_type<uint>(half2(t[6], t[7])));
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    threadgroup const half* lutH = (threadgroup const half*)lutv;
    const uint row0 = (tgid * NSG + sg) * NR; if (row0 >= N) return;
    const uint nblk = K / 128u, ix = lane >> 3, it = lane & 7u, qn = it >> 1;
    float2 acc[NR]; for (int r = 0; r < NR; r++) acc[r] = float2(0.0f);
    device const uchar* rowp[NR]; for (int r = 0; r < NR; r++) rowp[r] = W + ulong(min(row0 + uint(r), N - 1u)) * ulong(nblk) * 28ul;
    for (uint b = ix; b < nblk; b += 4) {
        device const half2* xb = x + ulong(b) * 128ul;
        half2 ya[2][5], yb[5]; half2 yq;
        for (uint k = 0; k < 2; k++) for (uint n = 0; n < 5; n++) ya[k][n] = xb[n * 16 + 2 * it + k];
        for (uint n = 0; n < 5; n++) yb[n] = xb[80 + n * 8 + it];
        yq = xb[120 + qn * 2 + (it & 1u)];
        for (int r = 0; r < NR; r++) {
            device const uchar* bp = rowp[r] + b * 28u;
            const ushort w01 = *(device const ushort*)(bp + 2 * it);
            const uint c0 = w01 & 0xffu, c1 = w01 >> 8, c2 = bp[16 + it], cq = bp[24 + (it & 1u)];
            const float d = float(as_type<half>(*(device const ushort*)(bp + 26)));
            const uint4 e0 = lutv[c0], e1 = lutv[c1], e2 = lutv[c2];
            const half2 p00 = as_type<half2>(e0.x), p01 = as_type<half2>(e0.y), p02 = as_type<half2>(e0.z);
            const half2 p10 = as_type<half2>(e1.x), p11 = as_type<half2>(e1.y), p12 = as_type<half2>(e1.z);
            const half2 p20 = as_type<half2>(e2.x), p21 = as_type<half2>(e2.y), p22 = as_type<half2>(e2.z);
            half2 a0 = ya[0][0] * half2(p00.x); a0 = fma(ya[0][1], half2(p00.y), a0); a0 = fma(ya[0][2], half2(p01.x), a0); a0 = fma(ya[0][3], half2(p01.y), a0); a0 = fma(ya[0][4], half2(p02.x), a0);
            half2 a1 = ya[1][0] * half2(p10.x); a1 = fma(ya[1][1], half2(p10.y), a1); a1 = fma(ya[1][2], half2(p11.x), a1); a1 = fma(ya[1][3], half2(p11.y), a1); a1 = fma(ya[1][4], half2(p12.x), a1);
            half2 a2 = yb[0] * half2(p20.x); a2 = fma(yb[1], half2(p20.y), a2); a2 = fma(yb[2], half2(p21.x), a2); a2 = fma(yb[3], half2(p21.y), a2); a2 = fma(yb[4], half2(p22.x), a2);
            const half2 aq = yq * half2(lutH[cq * 8 + qn]);
            acc[r] += d * (float2(a0) + float2(a1) + float2(a2) + float2(aq));
        }
    }
    for (int r = 0; r < NR; r++) { float v0 = simd_sum(acc[r].x), v1 = simd_sum(acc[r].y); if (lane == 0 && row0 + uint(r) < N) { y[row0 + uint(r)] = v0; y[ulong(N) + row0 + uint(r)] = v1; } }
}
template [[host_name("k_gemv_lut2_nr4_sg4")]] kernel void k_gemv_lut2<4, 4>(device const uchar*, device const half2*, device float*, constant uint&, constant uint&, uint, uint, uint, uint);

// alpha/beta projections for two tokens (bf16 [48][5120] each; h is the plain normalised hidden state as half rows [2][5120]); output rows [2][128] = alpha(48) | beta(48) | unused. Grid 96, 128 threads.
kernel void k_gdn_ab2(device const bfloat* Wa [[buffer(0)]], device const bfloat* Wb [[buffer(1)]], device const half* h [[buffer(2)]], device float* ab [[buffer(3)]],
                      uint row [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]], uint sg [[simdgroup_index_in_threadgroup]]) {
    threadgroup float part[2][4];
    const bool isb = row >= 48u; const uint r = isb ? row - 48u : row;
    device const bfloat* w = (isb ? Wb : Wa) + ulong(r) * 5120ul;
    float a0 = 0.0f, a1 = 0.0f;
    for (uint k = sg * 1280u + lane; k < (sg + 1u) * 1280u; k += 32u) { const float wv = float(w[k]); a0 += wv * float(h[k]); a1 += wv * float(h[5120u + k]); }
    a0 = simd_sum(a0); a1 = simd_sum(a1);
    if (lane == 0) { part[0][sg] = a0; part[1][sg] = a1; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) { const uint o = isb ? 48u + r : r; ab[o] = part[0][0] + part[0][1] + part[0][2] + part[0][3]; ab[128u + o] = part[1][0] + part[1][1] + part[1][2] + part[1][3]; }
}

// ---- activation layout for the ALU verification GEMVs: row-major fp16 [T][K] -> interleaved half2 [K][T/2] (keeps norm/FWHT kernels untouched) ----
kernel void k_ilv2(device const half* x [[buffer(0)]], device half2* y [[buffer(1)]], constant uint& K [[buffer(2)]], uint i [[thread_position_in_grid]]) {
    if (i >= K) return; y[i] = half2(x[i], x[K + i]);
}
kernel void k_ilv4(device const half* x [[buffer(0)]], device half2* y [[buffer(1)]], constant uint& K [[buffer(2)]], uint i [[thread_position_in_grid]]) {
    if (i >= K) return; y[2 * i] = half2(x[i], x[K + i]); y[2 * i + 1] = half2(x[2 * K + i], x[3 * K + i]);
}
