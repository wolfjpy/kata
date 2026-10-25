// Prefill attention on the matrix unit (flash style, register-resident output): one simdgroup per (16 query rows, 1 head); the 6 heads that share a KV head form a threadgroup.
//   S = Q K^T (matmul2d, K read transposed straight from the fp16 cache)  ->  masked online softmax  ->  O += P V (matmul2d, accumulate into a cooperative tensor).
// q is fp16 [Mtok][6144], already scaled by 1/sqrt(256). out[t][h*256+d] = softmax(...)V * sigmoid(gate).
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
#include "common.h"
using namespace mpp::tensor_ops;

// KVT = half: fp16 cache.  KVT = int8_t (Q8): int8 cache with one fp16 scale per (position, KV head) for K (ksc) and V (vsc); the scores are scaled per key column and P per key before P V.
template <typename KVT, bool Q8>
kernel void k_flash_prefill(device half* q [[buffer(0)]], device KVT* kc [[buffer(1)]], device KVT* vc [[buffer(2)]], device const float* gate [[buffer(3)]],
                            device float* out [[buffer(4)]], constant uint& posBase [[buffer(5)]], constant uint& Mtok [[buffer(6)]], device const half* ksc [[buffer(7)]], device const half* vsc [[buffer(8)]],
                            uint2 tg [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
    constexpr int QR = 16, KT = 64, D = 256, KVD = 1024, QD = 6144;
    threadgroup half Ptg[6][QR][KT]; threadgroup float kS[6][KT], vS[6][KT];
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
    tensor<device KVT, dextents<int32_t, 2>, tensor_inline> tK0(kc + kvh * D, dextents<int32_t, 2>(D, KT), sKV);
    tensor<device KVT, dextents<int32_t, 2>, tensor_inline> tV0(vc + kvh * D, dextents<int32_t, 2>(D, KT), sKV);
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
        tensor<device KVT, dextents<int32_t, 2>, tensor_inline> tK(kc + ulong(k0) * KVD + kvh * D, dextents<int32_t, 2>(D, nk), sKV);
        if (Q8) { for (uint c = lane; c < uint(KT); c += 32u) { const bool ok = int(c) < nk; kS[sg][c] = ok ? float(ksc[ulong(k0 + c) * 4ul + kvh]) : 0.0f; vS[sg][c] = ok ? float(vsc[ulong(k0 + c) * 4ul + kvh]) : 0.0f; } simdgroup_barrier(mem_flags::mem_threadgroup); }
        op1.run(tQ, tK, cS);
        for (uint i = 0; i < capS; i++) if (cS.is_valid_element(i)) { auto ix = cS.get_multidimensional_index(i); const uint key = k0 + uint(ix[0]), pos = posBase + q0 + uint(ix[1]); if (Q8) cS[i] *= kS[sg][uint(ix[0])]; if (key > pos) cS[i] = -INFINITY; }
        reduce_rows(cS, cR, reduction_operation::max, -INFINITY);
        cR.store(tTmp);
        simdgroup_barrier(mem_flags::mem_threadgroup);
        if (lane < QR) { const float mo = mTG[sg][lane], mn = max(mo, tmpTG[sg][lane]); aTG[sg][lane] = (mn == -INFINITY) ? 1.0f : exp(mo - mn); mTG[sg][lane] = mn; sTG[sg][lane] = (mn == -INFINITY) ? 0.0f : mn; }
        simdgroup_barrier(mem_flags::mem_threadgroup);
        for (uint i = 0; i < capS; i++) if (cS.is_valid_element(i)) { auto ix = cS.get_multidimensional_index(i); const float p = exp(cS[i] - sTG[sg][ix[1]]); cS[i] = p; Ptg[sg][ix[1]][ix[0]] = half(Q8 ? p * vS[sg][uint(ix[0])] : p); }
        reduce_rows(cS, cR, reduction_operation::sum, 0.0f);
        cR.store(tTmp);
        simdgroup_barrier(mem_flags::mem_threadgroup);
        if (lane < QR) lTG[sg][lane] = lTG[sg][lane] * aTG[sg][lane] + tmpTG[sg][lane];
        for (uint j = 0; j < capO; j++) if (cO.is_valid_element(j)) { auto ix = cO.get_multidimensional_index(j); cO[j] *= aTG[sg][ix[1]]; }
        tensor<device KVT, dextents<int32_t, 2>, tensor_inline> tV(vc + ulong(k0) * KVD + kvh * D, dextents<int32_t, 2>(D, nk), sKV);
        op2.run(tP, tV, cO);
        simdgroup_barrier(mem_flags::mem_threadgroup);
    }
    for (uint j = 0; j < capO; j++) if (cO.is_valid_element(j)) {
        auto ix = cO.get_multidimensional_index(j); const uint r = uint(ix[1]), d = uint(ix[0]);
        if (r < rowsValid) { const ulong o = ulong(q0 + r) * QD + h * D + d; out[o] = cO[j] / lTG[sg][r] * sigmoidf(gate[o]); }
    }
}

template [[host_name("k_flash_prefill")]] kernel decltype(k_flash_prefill<half, false>) k_flash_prefill<half, false>;
template [[host_name("k_flash_prefill_q8")]] kernel decltype(k_flash_prefill<int8_t, true>) k_flash_prefill<int8_t, true>;

// q (fp32, RoPE and q-norm applied) -> fp16 pre-scaled by 1/sqrt(256) (exact in fp16: power of two)
kernel void k_cvt_q_half(device const float* src [[buffer(0)]], device half* dst [[buffer(1)]], constant uint& n [[buffer(2)]], uint i [[thread_position_in_grid]]) { if (i < n) dst[i] = half(src[i] * 0.0625f); }

// =====================================================================================================================
// Decode / verification attention on the matrix unit (flash-decoding): rows of the 16-row matrix block = the 6 heads of one KV head x up to 2 tokens.
// Q is packed as fp16 [kvh][16][256], pre-scaled by 1/16 and zero padded. Each (split, kvh) simdgroup walks a range of `chunk` keys and writes its partial (unnormalised O, m, l);
// k_attn_dec_combine merges the splits, applies the sigmoid gate and writes the attention output.
kernel void k_dec_q_pack(device const float* q [[buffer(0)]], device half* qp [[buffer(1)]], constant uint& T [[buffer(2)]], uint i [[thread_position_in_grid]]) {
    if (i >= 4u * 16u * 256u) return;
    const uint d = i & 255u, r = (i >> 8) & 15u, kvh = i >> 12;
    const uint t = r / 6u, j = r % 6u;
    qp[i] = (r < 6u * T) ? half(q[t * 6144u + (kvh * 6u + j) * 256u + d] * 0.0625f) : half(0.0h);
}

template <typename KVT, bool Q8>
kernel void k_attn_dec_split(device half* qp [[buffer(0)]], device KVT* kc [[buffer(1)]], device KVT* vc [[buffer(2)]], device float* pacc [[buffer(3)]], device float* pm [[buffer(4)]], device float* pl [[buffer(5)]],
                             constant uint4& prm [[buffer(6)]], device const half* ksc [[buffer(7)]], device const half* vsc [[buffer(8)]],       // x = position of token 0, y = tokens (1..2), z = first valid position, w = keys per split
                             uint2 tg [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]]) {
    constexpr int QR = 16, KT = 64, D = 256, KVD = 1024;
    threadgroup half Ptg[QR][KT]; threadgroup float kS[KT], vS[KT]; threadgroup float mTG[QR], lTG[QR], aTG[QR], sTG[QR], tmpTG[QR];
    const uint split = tg.x, kvh = tg.y;
    const uint posBase = prm.x, T = prm.y, firstPos = prm.z, chunk = prm.w;
    const uint posEnd = posBase + T;
    const uint lo = firstPos + split * chunk, hi = min(lo + chunk, posEnd);
    constexpr auto d1 = matmul2d_descriptor(QR, KT, D, false, true, false, matmul2d_descriptor::mode::multiply);
    constexpr auto d2 = matmul2d_descriptor(QR, D, KT, false, false, false, matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<d1, execution_simdgroup> op1; matmul2d<d2, execution_simdgroup> op2;
    array<int, 2> sQ = {1, D}, sKV = {1, KVD}, sP = {1, KT};
    tensor<device half, dextents<int32_t, 2>, tensor_inline> tQ(qp + kvh * QR * D, dextents<int32_t, 2>(D, QR), sQ);
    tensor<device KVT, dextents<int32_t, 2>, tensor_inline> tK0(kc + kvh * D, dextents<int32_t, 2>(D, KT), sKV);
    tensor<device KVT, dextents<int32_t, 2>, tensor_inline> tV0(vc + kvh * D, dextents<int32_t, 2>(D, KT), sKV);
    tensor<threadgroup half, dextents<int32_t, 2>, tensor_inline> tP(&Ptg[0][0], dextents<int32_t, 2>(KT, QR), sP);
    auto cS = op1.get_destination_cooperative_tensor<decltype(tQ), decltype(tK0), float>();
    auto cO = op2.get_destination_cooperative_tensor<decltype(tP), decltype(tV0), float>();
    auto cR = op1.get_row_reduction_destination_cooperative_tensor<decltype(tQ), decltype(tK0), float>();
    array<int, 1> s1 = {1};
    tensor<threadgroup float, dextents<int32_t, 1>, tensor_inline> tTmp(&tmpTG[0], dextents<int32_t, 1>(QR), s1);
    const uint capS = cS.get_capacity(), capO = cO.get_capacity();
    for (uint i = 0; i < capO; i++) cO[i] = 0.0f;
    if (lane < QR) { mTG[lane] = -INFINITY; lTG[lane] = 0.0f; }
    simdgroup_barrier(mem_flags::mem_threadgroup);
    for (uint k0 = lo; k0 < hi; k0 += KT) {
        const int nk = int(min(uint(KT), hi - k0));
        tensor<device KVT, dextents<int32_t, 2>, tensor_inline> tK(kc + ulong(k0) * KVD + kvh * D, dextents<int32_t, 2>(D, nk), sKV);
        if (Q8) { for (uint c = lane; c < uint(KT); c += 32u) { const bool ok = int(c) < nk; kS[c] = ok ? float(ksc[ulong(k0 + c) * 4ul + kvh]) : 0.0f; vS[c] = ok ? float(vsc[ulong(k0 + c) * 4ul + kvh]) : 0.0f; } simdgroup_barrier(mem_flags::mem_threadgroup); }
        op1.run(tQ, tK, cS);
        for (uint i = 0; i < capS; i++) if (cS.is_valid_element(i)) { auto ix = cS.get_multidimensional_index(i); const uint key = k0 + uint(ix[0]), tok = uint(ix[1]) / 6u; if (Q8) cS[i] *= kS[uint(ix[0])]; if (uint(ix[1]) >= 6u * T || key > posBase + tok || key >= hi) cS[i] = -INFINITY; }
        reduce_rows(cS, cR, reduction_operation::max, -INFINITY);
        cR.store(tTmp);
        simdgroup_barrier(mem_flags::mem_threadgroup);
        if (lane < QR) { const float mo = mTG[lane], mn = max(mo, tmpTG[lane]); aTG[lane] = (mn == -INFINITY) ? 1.0f : exp(mo - mn); mTG[lane] = mn; sTG[lane] = (mn == -INFINITY) ? 0.0f : mn; }
        simdgroup_barrier(mem_flags::mem_threadgroup);
        for (uint i = 0; i < capS; i++) if (cS.is_valid_element(i)) { auto ix = cS.get_multidimensional_index(i); const float p = exp(cS[i] - sTG[ix[1]]); cS[i] = p; Ptg[ix[1]][ix[0]] = half(Q8 ? p * vS[uint(ix[0])] : p); }
        reduce_rows(cS, cR, reduction_operation::sum, 0.0f);
        cR.store(tTmp);
        simdgroup_barrier(mem_flags::mem_threadgroup);
        if (lane < QR) lTG[lane] = lTG[lane] * aTG[lane] + tmpTG[lane];
        for (uint j = 0; j < capO; j++) if (cO.is_valid_element(j)) { auto ix = cO.get_multidimensional_index(j); cO[j] *= aTG[ix[1]]; }
        tensor<device KVT, dextents<int32_t, 2>, tensor_inline> tV(vc + ulong(k0) * KVD + kvh * D, dextents<int32_t, 2>(D, nk), sKV);
        op2.run(tP, tV, cO);
        simdgroup_barrier(mem_flags::mem_threadgroup);
    }
    const uint nSplits = (posEnd - firstPos + chunk - 1) / chunk;
    const ulong rowBase = (ulong(kvh) * nSplits + split) * QR;
    for (uint j = 0; j < capO; j++) if (cO.is_valid_element(j)) { auto ix = cO.get_multidimensional_index(j); pacc[(rowBase + uint(ix[1])) * D + uint(ix[0])] = cO[j]; }
    if (lane < QR) { pm[rowBase + lane] = mTG[lane]; pl[rowBase + lane] = lTG[lane]; }
}

template [[host_name("k_attn_dec_split")]] kernel decltype(k_attn_dec_split<half, false>) k_attn_dec_split<half, false>;
template [[host_name("k_attn_dec_split_q8")]] kernel decltype(k_attn_dec_split<int8_t, true>) k_attn_dec_split<int8_t, true>;

// grid (24 heads, T tokens), 256 threads (one per head dimension)
kernel void k_attn_dec_combine(device const float* pacc [[buffer(0)]], device const float* pm [[buffer(1)]], device const float* pl [[buffer(2)]], device const float* gate [[buffer(3)]], device float* out [[buffer(4)]],
                               constant uint4& prm [[buffer(5)]], uint2 tg [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {
    const uint h = tg.x, t = tg.y, kvh = h / 6u, row = t * 6u + h % 6u;
    const uint posEnd = prm.x + prm.y, firstPos = prm.z, chunk = prm.w; const uint nSplits = (posEnd - firstPos + chunk - 1) / chunk;
    float M = -INFINITY; for (uint s = 0; s < nSplits; s++) M = max(M, pm[(ulong(kvh) * nSplits + s) * 16u + row]);
    float L = 0.0f, A = 0.0f;
    for (uint s = 0; s < nSplits; s++) { const ulong r = (ulong(kvh) * nSplits + s) * 16u + row; const float mm = pm[r]; const float w = (mm == -INFINITY) ? 0.0f : exp(mm - M); L += pl[r] * w; A += pacc[r * 256u + tid] * w; }
    const ulong o = ulong(t) * 6144ul + h * 256u + tid;
    out[o] = (A / L) * sigmoidf(gate[o]);
}

// =====================================================================================================================
// int8 KV cache. The prep kernels write the new rows as fp16 into a small scratch ([T][1024] for K and for V); this kernel quantises them into the cache at
// positions pos0 + t: one scale per (position, KV head) = fp16(amax / 127), values round(x / scale) clamped to +-127. Grid (8, T): x < 4 key head x, else value head x - 4. 256 threads.
kernel void k_kv_quant(device const half* ks [[buffer(0)]], device const half* vs [[buffer(1)]], device int8_t* kq [[buffer(2)]], device int8_t* vq [[buffer(3)]],
                       device half* ksc [[buffer(4)]], device half* vsc [[buffer(5)]], constant uint& pos0 [[buffer(6)]],
                       uint2 tg [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]], uint sg [[simdgroup_index_in_threadgroup]]) {
    threadgroup float part[8];
    const bool isV = tg.x >= 4u; const uint head = tg.x & 3u, t = tg.y;
    device const half* src = (isV ? vs : ks) + ulong(t) * 1024ul + head * 256u; const float v = float(src[tid]);
    float m = simd_max(fabs(v)); if (lane == 0) part[sg] = m;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float amax = 0.0f; for (uint s = 0; s < 8; s++) amax = max(amax, part[s]);
    const half sc = amax > 0.0f ? half(amax / 127.0f) : half(1.0h); const float inv = 1.0f / float(sc);
    const ulong pos = ulong(pos0 + t);
    (isV ? vq : kq)[pos * 1024ul + head * 256u + tid] = int8_t(clamp(rint(v * inv), -127.0f, 127.0f));
    if (tid == 0) (isV ? vsc : ksc)[pos * 4ul + head] = sc;
}
