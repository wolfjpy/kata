// M12 prototype (NOT used by the engine). Staged-load GEMV: the simdgroup copies 4 rows x one quad of 4 blocks (112 bytes per row) into threadgroup memory with
// 16-byte loads and the lanes read their small fields from there. KNOWN ISSUE: the results are wrong (max|diff| ~1e6; the cause was not found); the timing is
// nevertheless representative because the instruction mix is the same: 125-135 GB/s, identical to the native-load k_gemv_v1 (127-134 GB/s). See docs/reports/M12-weight-layout.md.
#include "common.h"
// ---- staged variant: the simdgroup first copies its NR rows x one quad of 4 blocks (112 bytes per row, 16-byte aligned) into threadgroup memory with 16-byte loads, one per lane,
// then every lane reads its small fields from threadgroup memory. The next quad's global load is issued before the current quad is computed (software pipelining).
template <int NR, int NSG>
kernel void k_gemv_v1s(device const uchar* W [[buffer(0)]], device const float* x [[buffer(1)]], device float* y [[buffer(2)]],
                       constant uint& K [[buffer(3)]], constant uint& N [[buffer(4)]],
                       uint tgid [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
    threadgroup uint4 stage4[NSG][2][NR * 7];     // uint4 elements so that the 16-byte stores are 16-byte aligned
    const uint row0 = (tgid * NSG + sg) * NR;
    if (row0 >= N) return;
    const uint nblk = K / 128u, nq = nblk / 4u, rowBytes = nblk * 28u;
    const uint ix = lane >> 3, it = lane & 7u;
    const uint lr = lane / 7u, lv = lane - lr * 7u;                      // loader role: row lr, 16-byte vector lv of the quad (lanes 28..31 idle)
    device const uchar* lrow = W + ulong(min(row0 + lr, N - 1u)) * ulong(rowBytes) + lv * 16u;
    const uint qn = it >> 1;
    const float p3q = qn == 0 ? 1.0f : qn == 1 ? 3.0f : qn == 2 ? 9.0f : 27.0f;
    float acc[NR]; for (int r = 0; r < NR; r++) acc[r] = 0.0f;
    uint4 nxt = lane < 28u ? *(device const uint4*)(lrow) : uint4(0);
    for (uint q = 0; q < nq; q++) {
        threadgroup uchar* st = (threadgroup uchar*)&stage4[sg][q & 1u][0];
        if (lane < 28u) *(threadgroup uint4*)(st + lr * 112u + lv * 16u) = nxt;
        simdgroup_barrier(mem_flags::mem_threadgroup);
        if (q + 1 < nq && lane < 28u) nxt = *(device const uint4*)(lrow + (q + 1) * 112u);
        const uint b = q * 4u + ix;
        device const float* xb = x + b * 128u;
        float yl[16]; float sumy = 0.0f;
        for (uint k = 0; k < 2; k++) for (uint n = 0; n < 5; n++) { float v = xb[n * 16 + 2 * it + k]; yl[5 * k + n] = v; sumy += v; }
        for (uint n = 0; n < 5; n++) { float v = xb[80 + n * 8 + it]; yl[10 + n] = v; sumy += v; }
        float yq = xb[120 + qn * 2 + (it & 1u)]; sumy += yq;
        float c[15];
        for (int g = 0; g < 3; g++) { const float* p = yl + 5 * g; c[5 * g + 0] = p[0] - 3.0f * p[1]; c[5 * g + 1] = p[1] - 3.0f * p[2]; c[5 * g + 2] = p[2] - 3.0f * p[3]; c[5 * g + 3] = p[3] - 3.0f * p[4]; c[5 * g + 4] = p[4]; }
        for (int r = 0; r < NR; r++) {
            threadgroup const uchar* bp = st + r * 112 + ix * 28;
            const ushort w01 = *(threadgroup const ushort*)(bp + 2 * it);
            float u0 = float(w01 & 0xffu) * (1.0f / 256.0f), u1 = float(w01 >> 8) * (1.0f / 256.0f);
            float u2 = float(bp[16 + it]) * (1.0f / 256.0f);
            float uq = float(bp[24 + (it & 1u)]) * (1.0f / 256.0f);
            float s = 0.0f;
            s += floor(3.0f * u0) * c[0] + floor(9.0f * u0) * c[1] + floor(27.0f * u0) * c[2] + floor(81.0f * u0) * c[3] + floor(243.0f * u0) * c[4];
            s += floor(3.0f * u1) * c[5] + floor(9.0f * u1) * c[6] + floor(27.0f * u1) * c[7] + floor(81.0f * u1) * c[8] + floor(243.0f * u1) * c[9];
            s += floor(3.0f * u2) * c[10] + floor(9.0f * u2) * c[11] + floor(27.0f * u2) * c[12] + floor(81.0f * u2) * c[13] + floor(243.0f * u2) * c[14];
            s += (floor(3.0f * p3q * uq) - 3.0f * floor(p3q * uq)) * yq;
            const float d = float(as_type<half>(*(threadgroup const ushort*)(bp + 26)));
            acc[r] += d * (s - sumy);
        }
    }
    for (int r = 0; r < NR; r++) { float v = simd_sum(acc[r]); if (lane == 0 && row0 + uint(r) < N) y[row0 + uint(r)] = v; }
}
#define INSTS(NR, NSG) template [[host_name("k_gemv_v1s_nr" #NR "_sg" #NSG)]] kernel void k_gemv_v1s<NR, NSG>(device const uchar*, device const float*, device float*, constant uint&, constant uint&, uint, uint, uint);
INSTS(4, 4) INSTS(4, 8) INSTS(4, 2) INSTS(2, 4) INSTS(8, 4)
