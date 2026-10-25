// Wide verification GEMV on the matrix unit (M14 gate experiment). Each simdgroup owns R output rows and works alone (no threadgroup-wide barriers): per K chunk it decodes its
// R x KC weights through the 16-byte trit LUT into a private fp16 tile in threadgroup memory (scale applied, exact), then one single-simdgroup matmul2d multiplies the
// MT token rows (fp16 activations straight from device memory) with that tile and accumulates in a cooperative tensor. The weights are decoded once for all tokens.
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
#include "common.h"
using namespace mpp::tensor_ops;

// One half-job decodes half of one (row, block): h = 0: qs bytes 0..11 and qh byte 0; h = 1: qs bytes 12..23 and qh byte 1.
// qs byte j < 16 holds the digits of elements n*16 + j (n = 0..4); j >= 16 elements 80 + n*8 + (j - 16); qh byte hb holds elements 120 + 2n + hb (n = 0..3).
inline void wide_halfjob(device const uchar* bp, uint h, threadgroup const uint4* lutv, threadgroup half* rowOut, half d) {
    const uint w0 = *(device const uint*)(bp + 12u * h), w1 = *(device const uint*)(bp + 12u * h + 4u), w2 = *(device const uint*)(bp + 12u * h + 8u);
    const uint ws[3] = {w0, w1, w2};
    for (uint qd = 0; qd < 3; qd++) {                                   // quads of 4 consecutive bytes
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
    const uint qh = bp[24 + h];
    const uint4 q = *(threadgroup const uint4*)(&lutv[qh]); const half2 q0 = as_type<half2>(q.x), q1 = as_type<half2>(q.y);
    rowOut[120 + 0 + h] = q0.x * d; rowOut[120 + 2 + h] = q0.y * d; rowOut[120 + 4 + h] = q1.x * d; rowOut[120 + 6 + h] = q1.y * d;
}

template <int MT, int R, int KC, int NSG, int ABL = 0>   // ABL bit 0: skip the decode, bit 1: skip the matmul (ablation)
kernel void k_wide_sg(device const uchar* W [[buffer(0)]], device half* x [[buffer(1)]], device float* y [[buffer(2)]], constant uint& K [[buffer(3)]], constant uint& N [[buffer(4)]],
                      uint tgid [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]], uint tid [[thread_index_in_threadgroup]]) {
    constexpr int LD = KC + 8;                                            // padded row stride of the private tile (16-byte multiple, breaks bank conflicts)
    threadgroup uint4 lutv[256]; threadgroup half Bt[NSG][R * LD];
    for (uint b = tid; b < 256u; b += NSG * 32u) {
        uint p3[6] = {1u, 3u, 9u, 27u, 81u, 243u}; half t[8];
        for (uint n = 0; n < 5; n++) t[n] = half(float(int((b * p3[n + 1]) >> 8) - 3 * int((b * p3[n]) >> 8) - 1)); t[5] = 0.0h; t[6] = 0.0h; t[7] = 0.0h;
        lutv[b] = uint4(as_type<uint>(half2(t[0], t[1])), as_type<uint>(half2(t[2], t[3])), as_type<uint>(half2(t[4], t[5])), as_type<uint>(half2(t[6], t[7])));
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const uint row0 = (tgid * NSG + sg) * R; if (row0 >= N) return;
    constexpr auto desc = matmul2d_descriptor(MT, R, KC, false, true, false, matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<desc, execution_simdgroup> op;
    array<int, 2> sA = {1, int(K)}, sB = {1, LD};
    tensor<device half, dextents<int32_t, 2>, tensor_inline> tA0(x, dextents<int32_t, 2>(KC, MT), sA);
    tensor<threadgroup half, dextents<int32_t, 2>, tensor_inline> tB(&Bt[sg][0], dextents<int32_t, 2>(KC, R), sB);
    auto cC = op.template get_destination_cooperative_tensor<decltype(tA0), decltype(tB), float>();
    for (uint i = 0; i < cC.get_capacity(); i++) cC[i] = 0.0f;
    const uint nblk = K / 128u, nBlkChunk = KC / 128;
    constexpr int JOBS = R * (KC / 128) * 2;                              // half-jobs per chunk
    for (uint kc = 0; kc < K / KC; kc++) {
        for (uint job = lane; job < uint(JOBS); job += 32u) {
            const uint h = job & 1u, rb = job >> 1, r = rb / nBlkChunk, bk = rb - r * nBlkChunk;
            device const uchar* bp = W + (ulong(min(row0 + r, N - 1u)) * ulong(nblk) + kc * nBlkChunk + bk) * 28ul;
            const half d = as_type<half>(*(device const ushort*)(bp + 26));
            if (!(ABL & 1)) wide_halfjob(bp, h, lutv, &Bt[sg][r * LD + bk * 128], d); else Bt[sg][r * LD + bk * 128 + h] = d;
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);
        tensor<device half, dextents<int32_t, 2>, tensor_inline> tA(x + kc * KC, dextents<int32_t, 2>(KC, MT), sA);
        if (!(ABL & 2)) op.run(tA, tB, cC);
        simdgroup_barrier(mem_flags::mem_threadgroup);
    }
    for (uint i = 0; i < cC.get_capacity(); i++) if (cC.is_valid_element(i)) { auto ix = cC.get_multidimensional_index(i); const uint col = row0 + uint(ix[0]); if (col < N) y[ulong(ix[1]) * N + col] = cC[i]; }
}
#define INSTW(MT, R, KC, NSG) template [[host_name("k_wide_sg_m" #MT "_r" #R "_k" #KC "_g" #NSG)]] kernel void k_wide_sg<MT, R, KC, NSG, 0>(device const uchar*, device half*, device float*, constant uint&, constant uint&, uint, uint, uint, uint);
#define INSTA(MT, R, KC, NSG, A) template [[host_name("k_wide_sg_m" #MT "_r" #R "_k" #KC "_g" #NSG "_a" #A)]] kernel void k_wide_sg<MT, R, KC, NSG, A>(device const uchar*, device half*, device float*, constant uint&, constant uint&, uint, uint, uint, uint);
INSTA(8, 16, 128, 4, 1) INSTA(8, 16, 128, 4, 2) INSTA(8, 16, 128, 4, 3)
INSTW(16, 16, 128, 4) INSTW(16, 32, 128, 2) INSTW(8, 16, 128, 4) INSTW(8, 32, 128, 2) INSTW(16, 16, 128, 2) INSTW(16, 32, 256, 1) INSTW(8, 16, 256, 2) INSTW(8, 16, 128, 1) INSTW(8, 16, 512, 1) INSTW(8, 32, 256, 1) INSTW(8, 64, 128, 1) INSTW(8, 32, 128, 1) INSTW(8, 16, 512, 2) INSTW(8, 32, 256, 2)

// ---- int8 variant: the private tile holds the trits themselves as int8 (exact, scale not applied), the matmul is half x int8 -> float per 128-block, and the block's row scales
// are applied to the MT x R partial result before it is accumulated (four elements per lane). KC is 128 so one scale per row per step.
inline void wide_halfjob_q8(device const uchar* bp, uint h, threadgroup const uint2* lut8, threadgroup uchar* rowOut) {
    const uint ws[3] = {*(device const uint*)(bp + 12u * h), *(device const uint*)(bp + 12u * h + 4u), *(device const uint*)(bp + 12u * h + 8u)};
    for (uint qd = 0; qd < 3; qd++) {
        const uint j = 12u * h + 4u * qd, w = ws[qd];
        const uint2 e0 = lut8[w & 0xffu], e1 = lut8[(w >> 8) & 0xffu], e2 = lut8[(w >> 16) & 0xffu], e3 = lut8[w >> 24];
        for (uint n = 0; n < 5; n++) {
            const uint sh = 8u * (n & 3u); const bool hi = n == 4;
            const uint v0 = ((hi ? e0.y : e0.x) >> sh) & 0xffu, v1 = ((hi ? e1.y : e1.x) >> sh) & 0xffu, v2 = ((hi ? e2.y : e2.x) >> sh) & 0xffu, v3 = ((hi ? e3.y : e3.x) >> sh) & 0xffu;
            const uint pk = v0 | (v1 << 8) | (v2 << 16) | (v3 << 24);
            *(threadgroup uint*)(rowOut + (j < 16u ? n * 16u + j : 80u + n * 8u + (j - 16u))) = pk;
        }
    }
    const uint qh = bp[24 + h]; const uint2 q = lut8[qh];
    rowOut[120 + 0 + h] = uchar(q.x & 0xffu); rowOut[120 + 2 + h] = uchar((q.x >> 8) & 0xffu); rowOut[120 + 4 + h] = uchar((q.x >> 16) & 0xffu); rowOut[120 + 6 + h] = uchar((q.x >> 24) & 0xffu);
}

template <int MT, int R, int NSG>
kernel void k_wide_q8(device const uchar* W [[buffer(0)]], device half* x [[buffer(1)]], device float* y [[buffer(2)]], constant uint& K [[buffer(3)]], constant uint& N [[buffer(4)]],
                      uint tgid [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]], uint tid [[thread_index_in_threadgroup]]) {
    constexpr int KC = 128, LD = KC + 16;
    threadgroup uint2 lut8[256]; threadgroup uint4 BtV[NSG][R * LD / 16]; threadgroup half dS[NSG][R];
    for (uint b = tid; b < 256u; b += NSG * 32u) {
        uint p3[6] = {1u, 3u, 9u, 27u, 81u, 243u}; uint t[8];
        for (uint n = 0; n < 5; n++) t[n] = uint(int((b * p3[n + 1]) >> 8) - 3 * int((b * p3[n]) >> 8) - 1) & 0xffu; t[5] = 0; t[6] = 0; t[7] = 0;
        lut8[b] = uint2(t[0] | (t[1] << 8) | (t[2] << 16) | (t[3] << 24), t[4]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const uint row0 = (tgid * NSG + sg) * R; if (row0 >= N) return;
    threadgroup uchar* Bt = (threadgroup uchar*)&BtV[sg][0];
    constexpr auto desc = matmul2d_descriptor(MT, R, KC, false, true, false, matmul2d_descriptor::mode::multiply);
    matmul2d<desc, execution_simdgroup> op;
    array<int, 2> sA = {1, int(K)}, sB = {1, LD};
    tensor<device half, dextents<int32_t, 2>, tensor_inline> tA0(x, dextents<int32_t, 2>(KC, MT), sA);
    tensor<threadgroup int8_t, dextents<int32_t, 2>, tensor_inline> tB((threadgroup int8_t*)Bt, dextents<int32_t, 2>(KC, R), sB);
    auto cT = op.template get_destination_cooperative_tensor<decltype(tA0), decltype(tB), float>();
    auto cA = op.template get_destination_cooperative_tensor<decltype(tA0), decltype(tB), float>();
    const uint cap = cA.get_capacity(); uint16_t colOf[8];
    for (uint i = 0; i < cap; i++) { cA[i] = 0.0f; auto ix = cA.get_multidimensional_index(i); colOf[i] = uint16_t(ix[0]); }
    const uint nblk = K / 128u;
    for (uint kc = 0; kc < nblk; kc++) {
        for (uint job = lane; job < uint(R * 2); job += 32u) {
            const uint h = job & 1u, r = job >> 1;
            device const uchar* bp = W + (ulong(min(row0 + r, N - 1u)) * ulong(nblk) + kc) * 28ul;
            wide_halfjob_q8(bp, h, lut8, Bt + r * LD);
            if (h == 0) dS[sg][r] = as_type<half>(*(device const ushort*)(bp + 26));
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);
        tensor<device half, dextents<int32_t, 2>, tensor_inline> tA(x + kc * KC, dextents<int32_t, 2>(KC, MT), sA);
        op.run(tA, tB, cT);
        for (uint i = 0; i < cap; i++) if (cT.is_valid_element(i)) cA[i] += cT[i] * float(dS[sg][colOf[i]]);
        simdgroup_barrier(mem_flags::mem_threadgroup);
    }
    for (uint i = 0; i < cap; i++) if (cA.is_valid_element(i)) { auto ix = cA.get_multidimensional_index(i); const uint col = row0 + uint(ix[0]); if (col < N) y[ulong(ix[1]) * N + col] = cA[i]; }
}
#define INSTQ(MT, R, NSG) template [[host_name("k_wide_q8_m" #MT "_r" #R "_g" #NSG)]] kernel void k_wide_q8<MT, R, NSG>(device const uchar*, device half*, device float*, constant uint&, constant uint&, uint, uint, uint, uint);
INSTQ(8, 16, 4) INSTQ(8, 32, 4) INSTQ(8, 16, 8) INSTQ(16, 16, 4) INSTQ(8, 32, 2) INSTQ(8, 64, 2)
