// Experimental T-token GEMV variants (ablation of load / decode cost). Not used by the engine.
#include "common.h"
// MODE 0: as k_gemv_v1h; 1: no activation loads (synthetic values); 2: no decode floor (g = raw byte-derived values); 3: activations staged in threadgroup memory
template <int NR, int NSG, int T, int MODE>
kernel void k_gemv_x(device const uchar* W [[buffer(0)]], device const half* x [[buffer(1)]], device float* y [[buffer(2)]], constant uint& K [[buffer(3)]], constant uint& N [[buffer(4)]],
                     uint tgid [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]], uint tid [[thread_index_in_threadgroup]]) {
    threadgroup half xs[MODE == 3 ? 2 * 5120 : 1];
    if (MODE == 3) { for (uint i = tid; i < T * K; i += NSG * 32) xs[i] = x[i]; threadgroup_barrier(mem_flags::mem_threadgroup); }
    const uint row0 = (tgid * NSG + sg) * NR; if (row0 >= N) return;
    const uint nblk = K / 128u, ix = lane >> 3, it = lane & 7u;
    float acc[T][NR]; for (int t = 0; t < T; t++) for (int r = 0; r < NR; r++) acc[t][r] = 0.0f;
    device const uchar* rowp[NR]; for (int r = 0; r < NR; r++) rowp[r] = W + ulong(min(row0 + uint(r), N - 1u)) * ulong(nblk) * 28ul;
    const uint qn = it >> 1; const float p3q = qn == 0 ? 1.0f : qn == 1 ? 3.0f : qn == 2 ? 9.0f : 27.0f;
    for (uint b = ix; b < nblk; b += 4) {
        float c[T][15], yq[T], sumy[T];
        for (int t = 0; t < T; t++) {
            float yl[16]; float sy = 0.0f;
            if (MODE == 1) { for (uint i = 0; i < 16; i++) { yl[i] = float(b * 16u + i + t) * 0.001f; sy += yl[i]; } yq[t] = float(b + t) * 0.002f; }
            else {
                device const half* xb = (MODE == 3 ? (threadgroup const half*)nullptr : nullptr) ? nullptr : (device const half*)(x + ulong(t) * K + b * 128u);
                if (MODE == 3) { threadgroup const half* xt = xs + t * K + b * 128u;
                    for (uint k = 0; k < 2; k++) for (uint n = 0; n < 5; n++) { float v = float(xt[n * 16 + 2 * it + k]); yl[5 * k + n] = v; sy += v; }
                    for (uint n = 0; n < 5; n++) { float v = float(xt[80 + n * 8 + it]); yl[10 + n] = v; sy += v; }
                    yq[t] = float(xt[120 + qn * 2 + (it & 1u)]); }
                else {
                    for (uint k = 0; k < 2; k++) for (uint n = 0; n < 5; n++) { float v = float(xb[n * 16 + 2 * it + k]); yl[5 * k + n] = v; sy += v; }
                    for (uint n = 0; n < 5; n++) { float v = float(xb[80 + n * 8 + it]); yl[10 + n] = v; sy += v; }
                    yq[t] = float(xb[120 + qn * 2 + (it & 1u)]); }
            }
            sy += yq[t]; sumy[t] = sy;
            for (int g = 0; g < 3; g++) { const float* p = yl + 5 * g; c[t][5*g+0] = p[0] - 3.0f * p[1]; c[t][5*g+1] = p[1] - 3.0f * p[2]; c[t][5*g+2] = p[2] - 3.0f * p[3]; c[t][5*g+3] = p[3] - 3.0f * p[4]; c[t][5*g+4] = p[4]; }
        }
        for (int r = 0; r < NR; r++) {
            device const uchar* bp = rowp[r] + b * 28u;
            ushort w01 = *(device const ushort*)(bp + 2 * it);
            float u0 = float(w01 & 0xffu) * (1.0f / 256.0f), u1 = float(w01 >> 8) * (1.0f / 256.0f), u2 = float(bp[16 + it]) * (1.0f / 256.0f), uq = float(bp[24 + (it & 1u)]) * (1.0f / 256.0f);
            float g0[5], g1[5], g2[5]; float gq;
            if (MODE == 2) { for (int k = 0; k < 5; k++) { g0[k] = u0 + float(k); g1[k] = u1 + float(k); g2[k] = u2 + float(k); } gq = uq; }
            else { g0[0]=floor(3.0f*u0); g0[1]=floor(9.0f*u0); g0[2]=floor(27.0f*u0); g0[3]=floor(81.0f*u0); g0[4]=floor(243.0f*u0);
                   g1[0]=floor(3.0f*u1); g1[1]=floor(9.0f*u1); g1[2]=floor(27.0f*u1); g1[3]=floor(81.0f*u1); g1[4]=floor(243.0f*u1);
                   g2[0]=floor(3.0f*u2); g2[1]=floor(9.0f*u2); g2[2]=floor(27.0f*u2); g2[3]=floor(81.0f*u2); g2[4]=floor(243.0f*u2);
                   gq = floor(3.0f * p3q * uq) - 3.0f * floor(p3q * uq); }
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
#define INSTX(NR, NSG, T, MODE) template [[host_name("k_gemv_x_t" #T "_m" #MODE "_nr" #NR "_sg" #NSG)]] kernel void k_gemv_x<NR, NSG, T, MODE>(device const uchar*, device const half*, device float*, constant uint&, constant uint&, uint, uint, uint, uint);
INSTX(4, 4, 1, 0) INSTX(4, 4, 1, 1) INSTX(4, 4, 1, 2) INSTX(4, 4, 1, 3)
INSTX(4, 4, 2, 0) INSTX(4, 4, 2, 1) INSTX(4, 4, 2, 2) INSTX(4, 4, 2, 3)

// ---- ALU throughput probe: each thread runs `iters` rounds over 8 independent accumulator chains ----
template <typename V>
inline V alu_fma_loop(uint iters, V a0, V m, V c) { V a[8] = {a0, a0 + c, a0 * m, a0 - c, a0 + m, a0 * c, a0 - m, a0 + c + m}; for (uint i = 0; i < iters; i++) for (int k = 0; k < 8; k++) a[k] = fma(a[k], m, c); V s = a[0]; for (int k = 1; k < 8; k++) s += a[k]; return s; }
kernel void k_alu_f32(device float* out [[buffer(0)]], constant uint& iters [[buffer(1)]], uint t [[thread_position_in_grid]]) { out[t] = alu_fma_loop<float>(iters, float(t) * 1e-6f, 0.9999f, 1e-4f); }
kernel void k_alu_f16(device half* out [[buffer(0)]], constant uint& iters [[buffer(1)]], uint t [[thread_position_in_grid]]) { out[t] = alu_fma_loop<half>(iters, half(float(t) * 1e-6f), half(0.999f), half(1e-3f)); }
kernel void k_alu_h2(device half2* out [[buffer(0)]], constant uint& iters [[buffer(1)]], uint t [[thread_position_in_grid]]) { out[t] = alu_fma_loop<half2>(iters, half2(float(t) * 1e-6f), half2(0.999h), half2(1e-3h)); }
kernel void k_alu_h4(device half4* out [[buffer(0)]], constant uint& iters [[buffer(1)]], uint t [[thread_position_in_grid]]) { out[t] = alu_fma_loop<half4>(iters, half4(float(t) * 1e-6f), half4(0.999h), half4(1e-3h)); }
kernel void k_alu_floor(device float* out [[buffer(0)]], constant uint& iters [[buffer(1)]], uint t [[thread_position_in_grid]]) { float a[8]; for (int k = 0; k < 8; k++) a[k] = float(t + k) * 0.37f; for (uint i = 0; i < iters; i++) for (int k = 0; k < 8; k++) a[k] = floor(a[k] * 1.0003f + 0.1f); float s = 0; for (int k = 0; k < 8; k++) s += a[k]; out[t] = s; }
kernel void k_alu_fract(device float* out [[buffer(0)]], constant uint& iters [[buffer(1)]], uint t [[thread_position_in_grid]]) { float a[8]; for (int k = 0; k < 8; k++) a[k] = float(t + k) * 0.37f; for (uint i = 0; i < iters; i++) for (int k = 0; k < 8; k++) a[k] = fract(a[k] * 1.0003f + 0.1f); float s = 0; for (int k = 0; k < 8; k++) s += a[k]; out[t] = s; }
kernel void k_alu_int(device uint* out [[buffer(0)]], constant uint& iters [[buffer(1)]], uint t [[thread_position_in_grid]]) { uint a[8]; for (int k = 0; k < 8; k++) a[k] = t + k; for (uint i = 0; i < iters; i++) for (int k = 0; k < 8; k++) a[k] = a[k] * 3u + 1u; uint s = 0; for (int k = 0; k < 8; k++) s += a[k]; out[t] = s; }
kernel void k_alu_i16(device ushort* out [[buffer(0)]], constant uint& iters [[buffer(1)]], uint t [[thread_position_in_grid]]) { ushort a[8]; for (int k = 0; k < 8; k++) a[k] = ushort(t + k); for (uint i = 0; i < iters; i++) for (int k = 0; k < 8; k++) a[k] = ushort(a[k] * ushort(3) + ushort(1)); ushort s = 0; for (int k = 0; k < 8; k++) s += a[k]; out[t] = s; }

// ---- LUT + half2 variant for T=2: trits come from a 256-entry threadgroup table (byte -> 5 half trits), the two tokens' activations are packed as half2 (interleaved [K] of half2) ----
// FLUSH: 0 = accumulate the whole block in half2 then convert; 1 = convert after every byte (5 terms)
template <int NR, int NSG, int FLUSH>
kernel void k_gemv_lut2(device const uchar* W [[buffer(0)]], device const half2* x [[buffer(1)]], device float* y [[buffer(2)]], constant uint& K [[buffer(3)]], constant uint& N [[buffer(4)]],
                        uint tgid [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]], uint tid [[thread_index_in_threadgroup]]) {
    threadgroup half lut[256 * 8];
    for (uint b = tid; b < 256u; b += NSG * 32u) {
        uint p3[6] = {1u, 3u, 9u, 27u, 81u, 243u};
        for (uint n = 0; n < 5; n++) lut[b * 8 + n] = half(float(int((b * p3[n + 1]) >> 8) - 3 * int((b * p3[n]) >> 8) - 1));
        lut[b * 8 + 5] = 0.0h; lut[b * 8 + 6] = 0.0h; lut[b * 8 + 7] = 0.0h;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
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
            threadgroup const half* l0 = lut + c0 * 8; threadgroup const half* l1 = lut + c1 * 8; threadgroup const half* l2 = lut + c2 * 8;
            half2 a0 = half2(0.0h), a1 = half2(0.0h), a2 = half2(0.0h);
            for (int n = 0; n < 5; n++) { a0 = fma(ya[0][n], half2(l0[n]), a0); a1 = fma(ya[1][n], half2(l1[n]), a1); a2 = fma(yb[n], half2(l2[n]), a2); }
            const half2 aq = yq * half2(lut[cq * 8 + qn]);
            float2 f;
            if (FLUSH) f = float2(a0) + float2(a1) + float2(a2) + float2(aq);
            else f = float2((a0 + a1) + (a2 + aq));
            acc[r] += d * f;
        }
    }
    for (int r = 0; r < NR; r++) { float v0 = simd_sum(acc[r].x), v1 = simd_sum(acc[r].y); if (lane == 0 && row0 + uint(r) < N) { y[row0 + uint(r)] = v0; y[ulong(N) + row0 + uint(r)] = v1; } }
}
#define INSTL(NR, NSG, FL) template [[host_name("k_gemv_lut2_f" #FL "_nr" #NR "_sg" #NSG)]] kernel void k_gemv_lut2<NR, NSG, FL>(device const uchar*, device const half2*, device float*, constant uint&, constant uint&, uint, uint, uint, uint);
INSTL(4, 4, 0) INSTL(4, 4, 1) INSTL(2, 4, 1) INSTL(8, 4, 1) INSTL(4, 8, 1) INSTL(4, 2, 1) INSTL(8, 4, 0)

// ---- ablation of the LUT kernel: ABL bit0 = no LUT reads, bit1 = no activation reads, bit2 = no weight reads (synthetic bytes) ----
template <int NR, int NSG, int ABL>
kernel void k_gemv_lut2a(device const uchar* W [[buffer(0)]], device const half2* x [[buffer(1)]], device float* y [[buffer(2)]], constant uint& K [[buffer(3)]], constant uint& N [[buffer(4)]],
                         uint tgid [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]], uint tid [[thread_index_in_threadgroup]]) {
    threadgroup half lut[256 * 8];
    for (uint b = tid; b < 256u; b += NSG * 32u) { uint p3[6] = {1u, 3u, 9u, 27u, 81u, 243u}; for (uint n = 0; n < 5; n++) lut[b * 8 + n] = half(float(int((b * p3[n + 1]) >> 8) - 3 * int((b * p3[n]) >> 8) - 1)); lut[b * 8 + 5] = 0.0h; lut[b * 8 + 6] = 0.0h; lut[b * 8 + 7] = 0.0h; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const uint row0 = (tgid * NSG + sg) * NR; if (row0 >= N) return;
    const uint nblk = K / 128u, ix = lane >> 3, it = lane & 7u, qn = it >> 1;
    float2 acc[NR]; for (int r = 0; r < NR; r++) acc[r] = float2(0.0f);
    device const uchar* rowp[NR]; for (int r = 0; r < NR; r++) rowp[r] = W + ulong(min(row0 + uint(r), N - 1u)) * ulong(nblk) * 28ul;
    for (uint b = ix; b < nblk; b += 4) {
        device const half2* xb = x + ulong(b) * 128ul;
        half2 ya[2][5], yb[5]; half2 yq;
        if (ABL & 2) { for (uint k = 0; k < 2; k++) for (uint n = 0; n < 5; n++) ya[k][n] = half2(half(float(b + n + k) * 0.01f)); for (uint n = 0; n < 5; n++) yb[n] = half2(half(float(b + n) * 0.02f)); yq = half2(half(float(b) * 0.03f)); }
        else { for (uint k = 0; k < 2; k++) for (uint n = 0; n < 5; n++) ya[k][n] = xb[n * 16 + 2 * it + k]; for (uint n = 0; n < 5; n++) yb[n] = xb[80 + n * 8 + it]; yq = xb[120 + qn * 2 + (it & 1u)]; }
        for (int r = 0; r < NR; r++) {
            device const uchar* bp = rowp[r] + b * 28u;
            uint c0, c1, c2, cq; float d;
            if (ABL & 4) { c0 = (b * 7u + it * 13u + r) & 255u; c1 = (b * 11u + it * 5u + r) & 255u; c2 = (b * 3u + it * 17u + r) & 255u; cq = (b + it) & 255u; d = 0.01f * float(r + 1); }
            else { const ushort w01 = *(device const ushort*)(bp + 2 * it); c0 = w01 & 0xffu; c1 = w01 >> 8; c2 = bp[16 + it]; cq = bp[24 + (it & 1u)]; d = float(as_type<half>(*(device const ushort*)(bp + 26))); }
            half2 a0 = half2(0.0h), a1 = half2(0.0h), a2 = half2(0.0h), aq;
            if (ABL & 1) { for (int n = 0; n < 5; n++) { a0 = fma(ya[0][n], half2(half(c0 & 1u)), a0); a1 = fma(ya[1][n], half2(half(c1 & 1u)), a1); a2 = fma(yb[n], half2(half(c2 & 1u)), a2); } aq = yq * half2(half(cq & 1u)); }
            else { threadgroup const half* l0 = lut + c0 * 8; threadgroup const half* l1 = lut + c1 * 8; threadgroup const half* l2 = lut + c2 * 8;
                   for (int n = 0; n < 5; n++) { a0 = fma(ya[0][n], half2(l0[n]), a0); a1 = fma(ya[1][n], half2(l1[n]), a1); a2 = fma(yb[n], half2(l2[n]), a2); } aq = yq * half2(lut[cq * 8 + qn]); }
            acc[r] += d * float2((a0 + a1) + (a2 + aq));
        }
    }
    for (int r = 0; r < NR; r++) { float v0 = simd_sum(acc[r].x), v1 = simd_sum(acc[r].y); if (lane == 0 && row0 + uint(r) < N) { y[row0 + uint(r)] = v0; y[ulong(N) + row0 + uint(r)] = v1; } }
}
#define INSTA(ABL) template [[host_name("k_gemv_lut2a_a" #ABL "_nr4_sg4")]] kernel void k_gemv_lut2a<4, 4, ABL>(device const uchar*, device const half2*, device float*, constant uint&, constant uint&, uint, uint, uint, uint);
INSTA(0) INSTA(1) INSTA(2) INSTA(3) INSTA(4) INSTA(5) INSTA(6) INSTA(7)

// ---- LUT with 16-byte entries: one threadgroup read per weight byte (5 half trits packed as 3 words) ----
template <int NR, int NSG, int VAR>
kernel void k_gemv_lut2v(device const uchar* W [[buffer(0)]], device const half2* x [[buffer(1)]], device float* y [[buffer(2)]], constant uint& K [[buffer(3)]], constant uint& N [[buffer(4)]],
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
            if (VAR == 0) acc[r] += d * float2((a0 + a1) + (a2 + aq));
            else acc[r] += d * (float2(a0) + float2(a1) + float2(a2) + float2(aq));
        }
    }
    for (int r = 0; r < NR; r++) { float v0 = simd_sum(acc[r].x), v1 = simd_sum(acc[r].y); if (lane == 0 && row0 + uint(r) < N) { y[row0 + uint(r)] = v0; y[ulong(N) + row0 + uint(r)] = v1; } }
}
#define INSTV(NR, NSG, VAR) template [[host_name("k_gemv_lut2v_v" #VAR "_nr" #NR "_sg" #NSG)]] kernel void k_gemv_lut2v<NR, NSG, VAR>(device const uchar*, device const half2*, device float*, constant uint&, constant uint&, uint, uint, uint, uint);
INSTV(4, 4, 0) INSTV(4, 4, 1) INSTV(2, 4, 0) INSTV(8, 4, 0) INSTV(4, 8, 0) INSTV(4, 2, 0) INSTV(3, 4, 0)

// ---- wide verification: NP half2 token pairs (T = 2*NP tokens) share one pass over the weights; activations interleaved as [K][NP] of half2 ----
template <int NR, int NSG, int NP>
kernel void k_gemv_lutp(device const uchar* W [[buffer(0)]], device const half2* x [[buffer(1)]], device float* y [[buffer(2)]], constant uint& K [[buffer(3)]], constant uint& N [[buffer(4)]],
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
    float2 acc[NR][NP]; for (int r = 0; r < NR; r++) for (int p = 0; p < NP; p++) acc[r][p] = float2(0.0f);
    device const uchar* rowp[NR]; for (int r = 0; r < NR; r++) rowp[r] = W + ulong(min(row0 + uint(r), N - 1u)) * ulong(nblk) * 28ul;
    for (uint b = ix; b < nblk; b += 4) {
        device const half2* xb = x + ulong(b) * 128ul * NP;
        half2 ya[2][5][NP], yb[5][NP], yq[NP];
        for (uint k = 0; k < 2; k++) for (uint n = 0; n < 5; n++) for (int p = 0; p < NP; p++) ya[k][n][p] = xb[(n * 16 + 2 * it + k) * NP + p];
        for (uint n = 0; n < 5; n++) for (int p = 0; p < NP; p++) yb[n][p] = xb[(80 + n * 8 + it) * NP + p];
        for (int p = 0; p < NP; p++) yq[p] = xb[(120 + qn * 2 + (it & 1u)) * NP + p];
        for (int r = 0; r < NR; r++) {
            device const uchar* bp = rowp[r] + b * 28u;
            const ushort w01 = *(device const ushort*)(bp + 2 * it);
            const uint c0 = w01 & 0xffu, c1 = w01 >> 8, c2 = bp[16 + it], cq = bp[24 + (it & 1u)];
            const float d = float(as_type<half>(*(device const ushort*)(bp + 26)));
            const uint4 e0 = lutv[c0], e1 = lutv[c1], e2 = lutv[c2];
            const half2 p00 = as_type<half2>(e0.x), p01 = as_type<half2>(e0.y), p02 = as_type<half2>(e0.z);
            const half2 p10 = as_type<half2>(e1.x), p11 = as_type<half2>(e1.y), p12 = as_type<half2>(e1.z);
            const half2 p20 = as_type<half2>(e2.x), p21 = as_type<half2>(e2.y), p22 = as_type<half2>(e2.z);
            const half tq = lutH[cq * 8 + qn];
            for (int p = 0; p < NP; p++) {
                half2 a0 = ya[0][0][p] * half2(p00.x); a0 = fma(ya[0][1][p], half2(p00.y), a0); a0 = fma(ya[0][2][p], half2(p01.x), a0); a0 = fma(ya[0][3][p], half2(p01.y), a0); a0 = fma(ya[0][4][p], half2(p02.x), a0);
                half2 a1 = ya[1][0][p] * half2(p10.x); a1 = fma(ya[1][1][p], half2(p10.y), a1); a1 = fma(ya[1][2][p], half2(p11.x), a1); a1 = fma(ya[1][3][p], half2(p11.y), a1); a1 = fma(ya[1][4][p], half2(p12.x), a1);
                half2 a2 = yb[0][p] * half2(p20.x); a2 = fma(yb[1][p], half2(p20.y), a2); a2 = fma(yb[2][p], half2(p21.x), a2); a2 = fma(yb[3][p], half2(p21.y), a2); a2 = fma(yb[4][p], half2(p22.x), a2);
                const half2 aq = yq[p] * half2(tq);
                acc[r][p] += d * (float2(a0) + float2(a1) + float2(a2) + float2(aq));
            }
        }
    }
    for (int r = 0; r < NR; r++) for (int p = 0; p < NP; p++) { float v0 = simd_sum(acc[r][p].x), v1 = simd_sum(acc[r][p].y); if (lane == 0 && row0 + uint(r) < N) { y[ulong(2 * p) * N + row0 + uint(r)] = v0; y[ulong(2 * p + 1) * N + row0 + uint(r)] = v1; } }
}
#define INSTP(NR, NSG, NP) template [[host_name("k_gemv_lutp" #NP "_nr" #NR "_sg" #NSG)]] kernel void k_gemv_lutp<NR, NSG, NP>(device const uchar*, device const half2*, device float*, constant uint&, constant uint&, uint, uint, uint, uint);
INSTP(4, 4, 1) INSTP(4, 4, 2) INSTP(2, 4, 2) INSTP(4, 4, 4) INSTP(2, 4, 4) INSTP(1, 4, 4) INSTP(2, 4, 3) INSTP(2, 4, 8) INSTP(1, 4, 8)

// ---- wide verification on the matrix unit: each threadgroup owns 64 output rows; per 128-weight K block the packed weights are decoded ONCE into a fp16 tile in threadgroup memory
// ([n][k], scale applied, exact) and multiplied with up to 16 token vectors by matmul2d. Tokens beyond the real count just compute garbage rows.
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace mpp::tensor_ops;
kernel void k_wide_gemm(device const uchar* W [[buffer(0)]], device half* x [[buffer(1)]], device float* y [[buffer(2)]], constant uint& K [[buffer(3)]], constant uint& N [[buffer(4)]],
                        uint tgid [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {
    threadgroup uint4 lutv[256]; threadgroup half Bt[64 * 128];
    for (uint b = tid; b < 256u; b += 128u) {
        uint p3[6] = {1u, 3u, 9u, 27u, 81u, 243u}; half t[8];
        for (uint n = 0; n < 5; n++) t[n] = half(float(int((b * p3[n + 1]) >> 8) - 3 * int((b * p3[n]) >> 8) - 1)); t[5] = 0.0h; t[6] = 0.0h; t[7] = 0.0h;
        lutv[b] = uint4(as_type<uint>(half2(t[0], t[1])), as_type<uint>(half2(t[2], t[3])), as_type<uint>(half2(t[4], t[5])), as_type<uint>(half2(t[6], t[7])));
    }
    threadgroup const half* lutH = (threadgroup const half*)lutv;
    const uint nblk = K / 128u, n0 = tgid * 64u, r = tid >> 1, hh = tid & 1u;
    constexpr auto desc = matmul2d_descriptor(16, 64, 128, false, true, false, matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<desc, execution_simdgroups<4>> op;
    array<int, 2> sA = {1, int(K)}, sB = {1, 128};
    tensor<device half, dextents<int32_t, 2>, tensor_inline> tA0(x, dextents<int32_t, 2>(128, 16), sA);
    tensor<threadgroup half, dextents<int32_t, 2>, tensor_inline> tB(&Bt[0], dextents<int32_t, 2>(128, 64), sB);
    auto cC = op.get_destination_cooperative_tensor<decltype(tA0), decltype(tB), float>();
    for (uint i = 0; i < cC.get_capacity(); i++) cC[i] = 0.0f;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    threadgroup half* row = Bt + r * 128u;
    for (uint kb = 0; kb < nblk; kb++) {
        device const uchar* bp = W + (ulong(min(n0 + r, N - 1u)) * ulong(nblk) + kb) * 28ul;
        const half d = as_type<half>(*(device const ushort*)(bp + 26));
        if (hh == 0) {   // elements 0..63: digits 0..3 of the 16 bytes qs[0..15]
            for (uint mq = 0; mq < 4; mq++) {
                const uint w = *(device const uint*)(bp + 4 * mq);
                const uint4 e0 = lutv[w & 0xffu], e1 = lutv[(w >> 8) & 0xffu], e2 = lutv[(w >> 16) & 0xffu], e3 = lutv[w >> 24];
                const half2 a0 = as_type<half2>(e0.x), a1 = as_type<half2>(e1.x), a2 = as_type<half2>(e2.x), a3 = as_type<half2>(e3.x);
                const half2 b0 = as_type<half2>(e0.y), b1 = as_type<half2>(e1.y), b2 = as_type<half2>(e2.y), b3 = as_type<half2>(e3.y);
                *(threadgroup half4*)(row + 0 * 16 + 4 * mq) = half4(a0.x, a1.x, a2.x, a3.x) * d;
                *(threadgroup half4*)(row + 1 * 16 + 4 * mq) = half4(a0.y, a1.y, a2.y, a3.y) * d;
                *(threadgroup half4*)(row + 2 * 16 + 4 * mq) = half4(b0.x, b1.x, b2.x, b3.x) * d;
                *(threadgroup half4*)(row + 3 * 16 + 4 * mq) = half4(b0.y, b1.y, b2.y, b3.y) * d;
            }
        } else {         // elements 64..79 (digit 4 of qs[0..15]), 80..119 (qs[16..23], 5 digits), 120..127 (qh)
            for (uint mq = 0; mq < 4; mq++) {
                const uint w = *(device const uint*)(bp + 4 * mq);
                *(threadgroup half4*)(row + 64 + 4 * mq) = half4(lutH[(w & 0xffu) * 8 + 4], lutH[((w >> 8) & 0xffu) * 8 + 4], lutH[((w >> 16) & 0xffu) * 8 + 4], lutH[(w >> 24) * 8 + 4]) * d;
            }
            for (uint mq = 0; mq < 2; mq++) {
                const uint w = *(device const uint*)(bp + 16 + 4 * mq);
                const uint4 e0 = lutv[w & 0xffu], e1 = lutv[(w >> 8) & 0xffu], e2 = lutv[(w >> 16) & 0xffu], e3 = lutv[w >> 24];
                const half2 a0 = as_type<half2>(e0.x), a1 = as_type<half2>(e1.x), a2 = as_type<half2>(e2.x), a3 = as_type<half2>(e3.x);
                const half2 b0 = as_type<half2>(e0.y), b1 = as_type<half2>(e1.y), b2 = as_type<half2>(e2.y), b3 = as_type<half2>(e3.y);
                const half2 c0 = as_type<half2>(e0.z), c1 = as_type<half2>(e1.z), c2 = as_type<half2>(e2.z), c3 = as_type<half2>(e3.z);
                *(threadgroup half4*)(row + 80 + 0 * 8 + 4 * mq) = half4(a0.x, a1.x, a2.x, a3.x) * d;
                *(threadgroup half4*)(row + 80 + 1 * 8 + 4 * mq) = half4(a0.y, a1.y, a2.y, a3.y) * d;
                *(threadgroup half4*)(row + 80 + 2 * 8 + 4 * mq) = half4(b0.x, b1.x, b2.x, b3.x) * d;
                *(threadgroup half4*)(row + 80 + 3 * 8 + 4 * mq) = half4(b0.y, b1.y, b2.y, b3.y) * d;
                *(threadgroup half4*)(row + 80 + 4 * 8 + 4 * mq) = half4(c0.x, c1.x, c2.x, c3.x) * d;
            }
            const ushort qh = *(device const ushort*)(bp + 24); const uint q0 = qh & 0xffu, q1 = qh >> 8;
            for (uint n = 0; n < 4; n++) *(threadgroup half2*)(row + 120 + 2 * n) = half2(lutH[q0 * 8 + n], lutH[q1 * 8 + n]) * half2(d);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        tensor<device half, dextents<int32_t, 2>, tensor_inline> tA(x + kb * 128u, dextents<int32_t, 2>(128, 16), sA);
        op.run(tA, tB, cC);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    for (uint i = 0; i < cC.get_capacity(); i++) if (cC.is_valid_element(i)) { auto ix = cC.get_multidimensional_index(i); const uint col = n0 + uint(ix[0]); if (col < N) y[ulong(ix[1]) * N + col] = cC[i]; }
}
