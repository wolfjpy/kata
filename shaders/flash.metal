// Prefill attention on the matrix unit (flash style, register-resident output): one simdgroup per (16 query rows, 1 head); the 6 heads that share a KV head form a threadgroup.
//   S = Q K^T (matmul2d, K read transposed straight from the fp16 cache)  ->  masked online softmax  ->  O += P V (matmul2d, accumulate into a cooperative tensor).
// q is fp16 [Mtok][6144], already scaled by 1/sqrt(256). out[t][h*256+d] = softmax(...)V * sigmoid(gate).
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
#include "common.h"
using namespace mpp::tensor_ops;

kernel void k_flash_prefill(device half* q [[buffer(0)]], device half* kc [[buffer(1)]], device half* vc [[buffer(2)]], device const float* gate [[buffer(3)]],
                            device float* out [[buffer(4)]], constant uint& posBase [[buffer(5)]], constant uint& Mtok [[buffer(6)]],
                            uint2 tg [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
    constexpr int QR = 16, KT = 64, D = 256, KVD = 1024, QD = 6144;
    threadgroup half Ptg[6][QR][KT];
    threadgroup float mTG[6][QR], lTG[6][QR], aTG[6][QR], sTG[6][QR], tmpTG[6][QR];
    const uint qt = tg.x, kvh = tg.y, h = kvh * 6u + sg; const uint q0 = qt * QR;
    const uint rowsValid = min(uint(QR), Mtok - q0);
    const uint posEnd = posBase + q0 + rowsValid;                       // keys [0, posEnd) are needed by this tile
    const uint nTiles = (posEnd + KT - 1) / KT;
    constexpr auto d1 = matmul2d_descriptor(QR, KT, D, false, true, false, matmul2d_descriptor::mode::multiply);
    constexpr auto d2 = matmul2d_descriptor(QR, D, KT, false, false, false, matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<d1, execution_simdgroup> op1; matmul2d<d2, execution_simdgroup> op2;
    array<int, 2> sQ = {1, QD}, sKV = {1, KVD}, sP = {1, KT};
    tensor<device half, dextents<int32_t, 2>, tensor_inline> tQ(q + ulong(q0) * QD + h * D, dextents<int32_t, 2>(D, QR), sQ);
    tensor<device half, dextents<int32_t, 2>, tensor_inline> tK0(kc + kvh * D, dextents<int32_t, 2>(D, KT), sKV);
    tensor<device half, dextents<int32_t, 2>, tensor_inline> tV0(vc + kvh * D, dextents<int32_t, 2>(D, KT), sKV);
    tensor<threadgroup half, dextents<int32_t, 2>, tensor_inline> tP(&Ptg[sg][0][0], dextents<int32_t, 2>(KT, QR), sP);
    auto cS = op1.get_destination_cooperative_tensor<decltype(tQ), decltype(tK0), float>();
    auto cO = op2.get_destination_cooperative_tensor<decltype(tP), decltype(tV0), float>();
    auto cR = op1.get_row_reduction_destination_cooperative_tensor<decltype(tQ), decltype(tK0), float>();
    array<int, 1> s1 = {1};
    tensor<threadgroup float, dextents<int32_t, 1>, tensor_inline> tTmp(&tmpTG[sg][0], dextents<int32_t, 1>(QR), s1);
    const uint capS = cS.get_capacity(), capO = cO.get_capacity();
    for (uint i = 0; i < capO; i++) cO[i] = 0.0f;
    if (lane < QR) { mTG[sg][lane] = -INFINITY; lTG[sg][lane] = 0.0f; }
    simdgroup_barrier(mem_flags::mem_threadgroup);
    for (uint kt = 0; kt < nTiles; kt++) {
        const uint k0 = kt * KT;
        const int nk = int(min(uint(KT), posEnd - k0));                       // valid keys in this tile (the tail tile is shorter); the tensor extent keeps the unit from reading past the cache
        tensor<device half, dextents<int32_t, 2>, tensor_inline> tK(kc + ulong(k0) * KVD + kvh * D, dextents<int32_t, 2>(D, nk), sKV);
        op1.run(tQ, tK, cS);
        for (uint i = 0; i < capS; i++) if (cS.is_valid_element(i)) { auto ix = cS.get_multidimensional_index(i); const uint key = k0 + uint(ix[0]), pos = posBase + q0 + uint(ix[1]); if (key > pos) cS[i] = -INFINITY; }
        reduce_rows(cS, cR, reduction_operation::max, -INFINITY);
        cR.store(tTmp);
        simdgroup_barrier(mem_flags::mem_threadgroup);
        if (lane < QR) { const float mo = mTG[sg][lane], mn = max(mo, tmpTG[sg][lane]); aTG[sg][lane] = (mn == -INFINITY) ? 1.0f : exp(mo - mn); mTG[sg][lane] = mn; sTG[sg][lane] = (mn == -INFINITY) ? 0.0f : mn; }
        simdgroup_barrier(mem_flags::mem_threadgroup);
        for (uint i = 0; i < capS; i++) if (cS.is_valid_element(i)) { auto ix = cS.get_multidimensional_index(i); const float p = exp(cS[i] - sTG[sg][ix[1]]); cS[i] = p; Ptg[sg][ix[1]][ix[0]] = half(p); }
        reduce_rows(cS, cR, reduction_operation::sum, 0.0f);
        cR.store(tTmp);
        simdgroup_barrier(mem_flags::mem_threadgroup);
        if (lane < QR) lTG[sg][lane] = lTG[sg][lane] * aTG[sg][lane] + tmpTG[sg][lane];
        for (uint j = 0; j < capO; j++) if (cO.is_valid_element(j)) { auto ix = cO.get_multidimensional_index(j); cO[j] *= aTG[sg][ix[1]]; }
        tensor<device half, dextents<int32_t, 2>, tensor_inline> tV(vc + ulong(k0) * KVD + kvh * D, dextents<int32_t, 2>(D, nk), sKV);
        op2.run(tP, tV, cO);
        simdgroup_barrier(mem_flags::mem_threadgroup);
    }
    for (uint j = 0; j < capO; j++) if (cO.is_valid_element(j)) {
        auto ix = cO.get_multidimensional_index(j); const uint r = uint(ix[1]), d = uint(ix[0]);
        if (r < rowsValid) { const ulong o = ulong(q0 + r) * QD + h * D + d; out[o] = cO[j] / lTG[sg][r] * sigmoidf(gate[o]); }
    }
}

// q (fp32, RoPE and q-norm applied) -> fp16 pre-scaled by 1/sqrt(256) (exact in fp16: power of two)
kernel void k_cvt_q_half(device const float* src [[buffer(0)]], device half* dst [[buffer(1)]], constant uint& n [[buffer(2)]], uint i [[thread_position_in_grid]]) { if (i < n) dst[i] = half(src[i] * 0.0625f); }
