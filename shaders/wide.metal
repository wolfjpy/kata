// Wide verification GEMV on the matrix unit: the same weight bytes are multiplied by MT (8 or 16) token rows in one pass.
// Each simdgroup owns N output rows and works alone (no threadgroup-wide barrier): per K chunk of KC weights it decodes its N x KC weights through the 16-byte trit LUT
// into a private fp16 tile in threadgroup memory (scale applied, exact), then one single-simdgroup matmul2d multiplies the MT token rows with that tile.
// Tile layout: column-major over K (element (k, n) at tile[n * LD + k], LD = KC + 8 to break bank conflicts).
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
#include "common.h"
using namespace mpp::tensor_ops;

// One half-job decodes half of one (row, block): h = 0: qs bytes 0..11 and qh byte 0; h = 1: qs bytes 12..23 and qh byte 1.
// qs byte j < 16 holds the digits of elements n*16 + j (n = 0..4); j >= 16 elements 80 + n*8 + (j-16); qh byte hb holds elements 120 + 2n + hb (n = 0..3).
inline void wide_halfjob(device const uchar* bp, uint h, threadgroup const uint4* lutv, threadgroup half* rowOut, half d) {
    const uint ws[3] = {*(device const uint*)(bp + 12u * h), *(device const uint*)(bp + 12u * h + 4u), *(device const uint*)(bp + 12u * h + 8u)};
    for (uint qd = 0; qd < 3; qd++) {                                   // quads of 4 consecutive qs bytes
        const uint j = 12u * h + 4u * qd, w = ws[qd];
        const uint4 e0 = lutv[w & 0xffu], e1 = lutv[(w >> 8) & 0xffu], e2 = lutv[(w >> 16) & 0xffu], e3 = lutv[w >> 24];
        const half2 a0 = as_type<half2>(e0.x), a1 = as_type<half2>(e1.x), a2 = as_type<half2>(e2.x), a3 = as_type<half2>(e3.x);
        const half2 b0 = as_type<half2>(e0.y), b1 = as_type<half2>(e1.y), b2 = as_type<half2>(e2.y), b3 = as_type<half2>(e3.y);
        const half2 c0 = as_type<half2>(e0.z), c1 = as_type<half2>(e1.z), c2 = as_type<half2>(e2.z), c3 = as_type<half2>(e3.z);
        if (j < 16u) {
            *(threadgroup half4*)(rowOut + 0 * 16 + j) = half4(a0.x, a1.x, a2.x, a3.x) * d;
            *(threadgroup half4*)(rowOut + 1 * 16 + j) = half4(a0.y, a1.y, a2.y, a3.y) * d;
            *(threadgroup half4*)(rowOut + 2 * 16 + j) = half4(b0.x, b1.x, b2.x, b3.x) * d;
            *(threadgroup half4*)(rowOut + 3 * 16 + j) = half4(b0.y, b1.y, b2.y, b3.y) * d;
            *(threadgroup half4*)(rowOut + 4 * 16 + j) = half4(c0.x, c1.x, c2.x, c3.x) * d;
        } else {
            const uint jj = j - 16u;
            *(threadgroup half4*)(rowOut + 80 + 0 * 8 + jj) = half4(a0.x, a1.x, a2.x, a3.x) * d;
            *(threadgroup half4*)(rowOut + 80 + 1 * 8 + jj) = half4(a0.y, a1.y, a2.y, a3.y) * d;
            *(threadgroup half4*)(rowOut + 80 + 2 * 8 + jj) = half4(b0.x, b1.x, b2.x, b3.x) * d;
            *(threadgroup half4*)(rowOut + 80 + 3 * 8 + jj) = half4(b0.y, b1.y, b2.y, b3.y) * d;
            *(threadgroup half4*)(rowOut + 80 + 4 * 8 + jj) = half4(c0.x, c1.x, c2.x, c3.x) * d;
        }
    }
    const uint qh = bp[24 + h]; const uint4 q = lutv[qh]; const half2 q0 = as_type<half2>(q.x), q1 = as_type<half2>(q.y);
    rowOut[120 + 0 + h] = q0.x * d; rowOut[120 + 2 + h] = q0.y * d; rowOut[120 + 4 + h] = q1.x * d; rowOut[120 + 6 + h] = q1.y * d;
}

// MT token rows (padded), N output rows per simdgroup, KC weights per matmul2d call, NSG simdgroups per threadgroup.
// DB = 1: two tiles per simdgroup, so the decode of chunk k+1 is issued while the matrix unit still works on chunk k (no dependency between them).
template <int MT, int N, int KC, int NSG, int ABL = 0, int DB = 0>
kernel void k_wide(device const uchar* W [[buffer(0)]], device half* x [[buffer(1)]], device float* y [[buffer(2)]], constant uint& K [[buffer(3)]], constant uint& Nout [[buffer(4)]],
                   uint tgid [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]], uint tid [[thread_index_in_threadgroup]]) {
    constexpr int LD = KC + 8;
    threadgroup uint4 lutv[256];
    threadgroup half Bt[NSG][(DB ? 2 : 1)][N * LD];
    for (uint b = tid; b < 256u; b += NSG * 32u) {
        uint p3[6] = {1u, 3u, 9u, 27u, 81u, 243u}; half t[8];
        for (uint n = 0; n < 5; n++) t[n] = half(float(int((b * p3[n + 1]) >> 8) - 3 * int((b * p3[n]) >> 8) - 1)); t[5] = 0.0h; t[6] = 0.0h; t[7] = 0.0h;
        lutv[b] = uint4(as_type<uint>(half2(t[0], t[1])), as_type<uint>(half2(t[2], t[3])), as_type<uint>(half2(t[4], t[5])), as_type<uint>(half2(t[6], t[7])));
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const uint row0 = (tgid * NSG + sg) * N; if (row0 >= Nout) return;
    constexpr auto desc = matmul2d_descriptor(MT, N, KC, false, true, false, matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<desc, execution_simdgroup> op;
    array<int, 2> sA = {1, int(K)}, sB = {1, LD};
    tensor<device half, dextents<int32_t, 2>, tensor_inline> tA0(x, dextents<int32_t, 2>(KC, MT), sA);
    tensor<threadgroup half, dextents<int32_t, 2>, tensor_inline> tB(&Bt[sg][0][0], dextents<int32_t, 2>(KC, N), sB);
    auto cC = op.template get_destination_cooperative_tensor<decltype(tA0), decltype(tB), float>();
    for (uint i = 0; i < cC.get_capacity(); i++) cC[i] = 0.0f;
    const uint nblk = K / 128u, nBlkChunk = KC / 128, nChunks = K / KC;
    constexpr int JOBS = N * (KC / 128) * 2;                             // half-jobs per simdgroup per chunk
    auto decodeChunk = [&](uint kc, int buf) {
        for (uint job = lane; job < uint(JOBS); job += 32u) {
            const uint h = job & 1u, rb = job >> 1, r = rb / nBlkChunk, bk = rb - r * nBlkChunk;
            device const uchar* bp = W + (ulong(min(row0 + r, Nout - 1u)) * ulong(nblk) + kc * nBlkChunk + bk) * 28ul;
            const half d = as_type<half>(*(device const ushort*)(bp + 26));
            threadgroup half* dst = &Bt[sg][DB ? buf : 0][r * LD + bk * 128];
            if (!(ABL & 1)) wide_halfjob(bp, h, lutv, dst, d); else dst[h] = d;
        }
    };
    if (DB) {   // software pipeline: decode 0, then alternate decode(k+1) / matmul(k)
        decodeChunk(0, 0);
        for (uint kc = 0; kc < nChunks; kc++) {
            simdgroup_barrier(mem_flags::mem_threadgroup);
            tensor<threadgroup half, dextents<int32_t, 2>, tensor_inline> tBk(&Bt[sg][kc & 1u][0], dextents<int32_t, 2>(KC, N), sB);
            tensor<device half, dextents<int32_t, 2>, tensor_inline> tA(x + kc * KC, dextents<int32_t, 2>(KC, MT), sA);
            if (kc + 1 < nChunks) decodeChunk(kc + 1, int((kc + 1) & 1u));    // independent of the tile the matrix unit reads now
            if (!(ABL & 2)) op.run(tA, tBk, cC);
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);
    } else {
        for (uint kc = 0; kc < nChunks; kc++) {
            decodeChunk(kc, 0);
            simdgroup_barrier(mem_flags::mem_threadgroup);
            tensor<device half, dextents<int32_t, 2>, tensor_inline> tA(x + kc * KC, dextents<int32_t, 2>(KC, MT), sA);
            if (!(ABL & 2)) op.run(tA, tB, cC);
            simdgroup_barrier(mem_flags::mem_threadgroup);
        }
    }
    for (uint i = 0; i < cC.get_capacity(); i++) if (cC.is_valid_element(i)) { auto ix = cC.get_multidimensional_index(i); const uint col = row0 + uint(ix[0]); if (col < Nout) y[ulong(ix[1]) * Nout + col] = cC[i]; }
}

#define WIDE(MT, NN, KC, SG) template [[host_name("k_wide_m" #MT "_n" #NN "_k" #KC "_g" #SG)]] kernel void k_wide<MT, NN, KC, SG, 0, 0>(device const uchar*, device half*, device float*, constant uint&, constant uint&, uint, uint, uint, uint);
#define WIDE_A(MT, NN, KC, SG, A) template [[host_name("k_wide_m" #MT "_n" #NN "_k" #KC "_g" #SG "_a" #A)]] kernel void k_wide<MT, NN, KC, SG, A, 0>(device const uchar*, device half*, device float*, constant uint&, constant uint&, uint, uint, uint, uint);
#define WIDED(MT, NN, KC, SG) template [[host_name("k_wided_m" #MT "_n" #NN "_k" #KC "_g" #SG)]] kernel void k_wide<MT, NN, KC, SG, 0, 1>(device const uchar*, device half*, device float*, constant uint&, constant uint&, uint, uint, uint, uint);
WIDE(16, 16, 128, 4) WIDE(16, 32, 128, 2) WIDE(16, 8, 128, 4) WIDE(16, 16, 256, 2)
WIDE_A(16, 16, 128, 4, 1) WIDE_A(16, 8, 128, 4, 1) WIDE_A(16, 32, 128, 2, 1)
WIDED(16, 16, 128, 3) WIDED(16, 8, 128, 4) WIDED(16, 16, 128, 2) WIDED(16, 24, 128, 2) WIDED(16, 8, 128, 8) WIDED(16, 16, 128, 4)

// =====================================================================================================================
// Hybrid: the matrix unit and the ALU (half2) work at the same time on different output rows. Each row of weights is read exactly once;
// rows [0, splitRow) go through the MMA path (M = MT token rows), rows [splitRow, Nout) through the half2 path (NP half2 pairs = 2*NP tokens).
// splitRow is chosen by the caller from the two paths' measured rates (about 2:1 in favour of the MMA path).
template <int MT, int MA_N, int MA_NSG, int AL_NR, int AL_NSG, int NP>
kernel void k_wide_hyb(device const uchar* W [[buffer(0)]], device half* xh [[buffer(1)]], device const half2* xi [[buffer(2)]], device float* y [[buffer(3)]],
                       constant uint& K [[buffer(4)]], constant uint& Nout [[buffer(5)]], constant uint& splitRow [[buffer(6)]], constant uint& tgsMma [[buffer(7)]],
                       uint tgid [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]], uint tid [[thread_index_in_threadgroup]]) {
    threadgroup uint4 lutv[256];
    for (uint b = tid; b < 256u; b += (MA_NSG > AL_NSG ? MA_NSG : AL_NSG) * 32u) {
        uint p3[6] = {1u, 3u, 9u, 27u, 81u, 243u}; half t[8];
        for (uint n = 0; n < 5; n++) t[n] = half(float(int((b * p3[n + 1]) >> 8) - 3 * int((b * p3[n]) >> 8) - 1)); t[5] = 0.0h; t[6] = 0.0h; t[7] = 0.0h;
        lutv[b] = uint4(as_type<uint>(half2(t[0], t[1])), as_type<uint>(half2(t[2], t[3])), as_type<uint>(half2(t[4], t[5])), as_type<uint>(half2(t[6], t[7])));
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const uint nblk = K / 128u, kBits = K / 128u;
    if (tgid < tgsMma) {   // ---- matrix-unit path
        constexpr int LD = 136;
        threadgroup half Bt[MA_NSG][MA_N * LD];
        const uint row0 = tgid * (MA_NSG * MA_N) + sg * MA_N; if (row0 >= splitRow) return;
        constexpr auto desc = matmul2d_descriptor(MT, MA_N, 128, false, true, false, matmul2d_descriptor::mode::multiply_accumulate);
        matmul2d<desc, execution_simdgroup> op;
        array<int, 2> sA = {1, int(K)}, sB = {1, LD};
        tensor<device half, dextents<int32_t, 2>, tensor_inline> tA0(xh, dextents<int32_t, 2>(128, MT), sA);
        tensor<threadgroup half, dextents<int32_t, 2>, tensor_inline> tB(&Bt[sg][0], dextents<int32_t, 2>(128, MA_N), sB);
        auto cC = op.template get_destination_cooperative_tensor<decltype(tA0), decltype(tB), float>();
        for (uint i = 0; i < cC.get_capacity(); i++) cC[i] = 0.0f;
        for (uint kb = 0; kb < kBits; kb++) {
            for (uint job = lane; job < uint(MA_N * 2); job += 32u) {
                const uint h = job & 1u, r = job >> 1;
                device const uchar* bp = W + (ulong(min(row0 + r, Nout - 1u)) * ulong(nblk) + kb) * 28ul;
                wide_halfjob(bp, h, lutv, &Bt[sg][r * LD], as_type<half>(*(device const ushort*)(bp + 26)));
            }
            simdgroup_barrier(mem_flags::mem_threadgroup);
            tensor<device half, dextents<int32_t, 2>, tensor_inline> tA(xh + kb * 128u, dextents<int32_t, 2>(128, MT), sA);
            op.run(tA, tB, cC);
            simdgroup_barrier(mem_flags::mem_threadgroup);
        }
        for (uint i = 0; i < cC.get_capacity(); i++) if (cC.is_valid_element(i)) { auto ix = cC.get_multidimensional_index(i); const uint col = row0 + uint(ix[0]); if (col < splitRow) y[ulong(ix[1]) * Nout + col] = cC[i]; }
    } else {               // ---- ALU (half2) path: AL_NR rows per simdgroup, NP half2 token pairs
        const uint row0 = splitRow + (tgid - tgsMma) * (AL_NSG * AL_NR) + sg * AL_NR; if (row0 >= Nout) return;
        threadgroup const half* lutH = (threadgroup const half*)lutv;
        const uint ix = lane >> 3, it = lane & 7u, qn = it >> 1;
        float2 acc[AL_NR][NP]; for (int r = 0; r < AL_NR; r++) for (int p = 0; p < NP; p++) acc[r][p] = float2(0.0f);
        device const uchar* rowp[AL_NR]; for (int r = 0; r < AL_NR; r++) rowp[r] = W + ulong(min(row0 + uint(r), Nout - 1u)) * ulong(nblk) * 28ul;
        for (uint b = ix; b < nblk; b += 4) {
            device const half2* xb = xi + ulong(b) * 128ul * NP;
            half2 ya[2][5][NP], yb[5][NP], yq[NP];
            for (uint k = 0; k < 2; k++) for (uint n = 0; n < 5; n++) for (int p = 0; p < NP; p++) ya[k][n][p] = xb[(n * 16 + 2 * it + k) * NP + p];
            for (uint n = 0; n < 5; n++) for (int p = 0; p < NP; p++) yb[n][p] = xb[(80 + n * 8 + it) * NP + p];
            for (int p = 0; p < NP; p++) yq[p] = xb[(120 + qn * 2 + (it & 1u)) * NP + p];
            for (int r = 0; r < AL_NR; r++) {
                device const uchar* bp = rowp[r] + b * 28u;
                const ushort w01 = *(device const ushort*)(bp + 2 * it);
                const uint c0 = w01 & 0xffu, c1 = w01 >> 8, c2 = bp[16 + it], cq = bp[24 + (it & 1u)];
                const float d = float(as_type<half>(*(device const ushort*)(bp + 26)));
                const half2 p00 = half2(lutH[c0 * 8 + 0]), p01 = half2(lutH[c0 * 8 + 1]), p02 = half2(lutH[c0 * 8 + 2]), p03 = half2(lutH[c0 * 8 + 3]), p04 = half2(lutH[c0 * 8 + 4]);
                const half2 q00 = half2(lutH[c1 * 8 + 0]), q01 = half2(lutH[c1 * 8 + 1]), q02 = half2(lutH[c1 * 8 + 2]), q03 = half2(lutH[c1 * 8 + 3]), q04 = half2(lutH[c1 * 8 + 4]);
                const half2 s00 = half2(lutH[c2 * 8 + 0]), s01 = half2(lutH[c2 * 8 + 1]), s02 = half2(lutH[c2 * 8 + 2]), s03 = half2(lutH[c2 * 8 + 3]), s04 = half2(lutH[c2 * 8 + 4]);
                for (int p = 0; p < NP; p++) {
                    half2 a0 = ya[0][0][p] * p00; a0 = fma(ya[0][1][p], p01, a0); a0 = fma(ya[0][2][p], p02, a0); a0 = fma(ya[0][3][p], p03, a0); a0 = fma(ya[0][4][p], p04, a0);
                    half2 a1 = ya[1][0][p] * q00; a1 = fma(ya[1][1][p], q01, a1); a1 = fma(ya[1][2][p], q02, a1); a1 = fma(ya[1][3][p], q03, a1); a1 = fma(ya[1][4][p], q04, a1);
                    half2 a2 = yb[0][p] * s00; a2 = fma(yb[1][p], s01, a2); a2 = fma(yb[2][p], s02, a2); a2 = fma(yb[3][p], s03, a2); a2 = fma(yb[4][p], s04, a2);
                    const half2 aq = yq[p] * half2(lutH[cq * 8 + qn]);
                    acc[r][p] += d * (float2(a0) + float2(a1) + float2(a2) + float2(aq));
                }
            }
        }
        for (int r = 0; r < AL_NR; r++) for (int p = 0; p < NP; p++) { float v0 = simd_sum(acc[r][p].x), v1 = simd_sum(acc[r][p].y); if (lane == 0 && row0 + uint(r) < Nout) { y[ulong(2 * p) * Nout + row0 + uint(r)] = v0; y[ulong(2 * p + 1) * Nout + row0 + uint(r)] = v1; } }
    }
}
#define HYB(MT, MAN, MANG, ALNR, ALNG, NP) template [[host_name("k_wide_hyb_" #MT "_" #MAN "_" #MANG "_" #ALNR "_" #ALNG "_" #NP)]] kernel void k_wide_hyb<MT, MAN, MANG, ALNR, ALNG, NP>(device const uchar*, device half*, device const half2*, device float*, constant uint&, constant uint&, constant uint&, constant uint&, uint, uint, uint, uint);
HYB(16, 16, 4, 4, 4, 4) HYB(16, 16, 4, 4, 4, 2) HYB(16, 16, 2, 4, 4, 4)
