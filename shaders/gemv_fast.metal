// Optimised PTQ1_0 matrix-vector kernels working directly on the GGUF block layout.
// Decode trick (float pipe only): a packed byte b is the base-3 fraction u = b/256; the digit sequence is t_n = g_{n+1} - 3 g_n with g_k = floor(3^k u).
// Hence for activations y_0..y_4:  sum_n t_n y_n = sum_{k=1..4} g_k (y_{k-1} - 3 y_k) + g_5 y_4,  and the coefficients depend only on the activations.
#include "common.h"

// Threadgroup = NSG simdgroups; each simdgroup owns NR consecutive output rows. 8 lanes cooperate on one 128-weight block, 4 blocks per iteration.
// Lane `it` (0..7) of a block owns bytes qs[2it], qs[2it+1] (elements n*16+2it+{0,1}), qs[16+it] (elements 80+n*8+it) and one trit of qh[it&1] (element 120+(it>>1)*2+(it&1)).
template <int NR, int NSG>
kernel void k_gemv_v1(device const uchar* W [[buffer(0)]], device const float* x [[buffer(1)]], device float* y [[buffer(2)]],
                      constant uint& K [[buffer(3)]], constant uint& N [[buffer(4)]],
                      uint tgid [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
    const uint row0 = (tgid * NSG + sg) * NR;
    if (row0 >= N) return;
    const uint nblk = K / 128u;
    const uint ix = lane >> 3, it = lane & 7u;
    float acc[NR];
    for (int r = 0; r < NR; r++) acc[r] = 0.0f;
    device const uchar* rowp[NR];
    for (int r = 0; r < NR; r++) rowp[r] = W + ulong(min(row0 + uint(r), N - 1u)) * ulong(nblk) * 28ul;

    for (uint b = ix; b < nblk; b += 4) {
        device const float* xb = x + b * 128u;
        float yl[16]; float sumy = 0.0f;
        for (uint k = 0; k < 2; k++) for (uint n = 0; n < 5; n++) { float v = xb[n * 16 + 2 * it + k]; yl[5 * k + n] = v; sumy += v; }
        for (uint n = 0; n < 5; n++) { float v = xb[80 + n * 8 + it]; yl[10 + n] = v; sumy += v; }
        const uint qn = it >> 1;                       // which trit of the qh byte this lane takes
        float yq = xb[120 + qn * 2 + (it & 1u)]; sumy += yq;
        // coefficient form of the three 5-trit groups: c1..c4 = y_{k-1} - 3 y_k, c5 = y_4
        float c[15];
        for (int g = 0; g < 3; g++) { const float* p = yl + 5 * g; c[5 * g + 0] = p[0] - 3.0f * p[1]; c[5 * g + 1] = p[1] - 3.0f * p[2]; c[5 * g + 2] = p[2] - 3.0f * p[3]; c[5 * g + 3] = p[3] - 3.0f * p[4]; c[5 * g + 4] = p[4]; }
        const float p3q = qn == 0 ? 1.0f : qn == 1 ? 3.0f : qn == 2 ? 9.0f : 27.0f;
        for (int r = 0; r < NR; r++) {
            device const uchar* bp = rowp[r] + b * 28u;
            ushort w01 = *(device const ushort*)(bp + 2 * it);
            float u0 = float(w01 & 0xffu) * (1.0f / 256.0f), u1 = float(w01 >> 8) * (1.0f / 256.0f);
            float u2 = float(bp[16 + it]) * (1.0f / 256.0f);
            float uq = float(bp[24 + (it & 1u)]) * (1.0f / 256.0f);
            float s = 0.0f;
            s += floor(3.0f * u0) * c[0] + floor(9.0f * u0) * c[1] + floor(27.0f * u0) * c[2] + floor(81.0f * u0) * c[3] + floor(243.0f * u0) * c[4];
            s += floor(3.0f * u1) * c[5] + floor(9.0f * u1) * c[6] + floor(27.0f * u1) * c[7] + floor(81.0f * u1) * c[8] + floor(243.0f * u1) * c[9];
            s += floor(3.0f * u2) * c[10] + floor(9.0f * u2) * c[11] + floor(27.0f * u2) * c[12] + floor(81.0f * u2) * c[13] + floor(243.0f * u2) * c[14];
            s += (floor(3.0f * p3q * uq) - 3.0f * floor(p3q * uq)) * yq;
            float d = float(as_type<half>(*(device const ushort*)(bp + 26)));
            acc[r] += d * (s - sumy);
        }
    }
    for (int r = 0; r < NR; r++) { float v = simd_sum(acc[r]); if (lane == 0 && row0 + uint(r) < N) y[row0 + uint(r)] = v; }
}
#define INST(NR, NSG) template [[host_name("k_gemv_v1_nr" #NR "_sg" #NSG)]] kernel void k_gemv_v1<NR, NSG>(device const uchar*, device const float*, device float*, constant uint&, constant uint&, uint, uint, uint);
INST(1, 4) INST(2, 4) INST(3, 4) INST(4, 4) INST(5, 4) INST(4, 2) INST(4, 8) INST(5, 2) INST(5, 8) INST(2, 8) INST(8, 4)
