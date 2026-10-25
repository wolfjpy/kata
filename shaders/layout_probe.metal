#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
#include "common.h"
using namespace mpp::tensor_ops;
// Dumps the (col,row) of every cooperative-tensor element for each of the 128 threads of a 64x64 int8 matmul2d (4 simdgroups).
kernel void k_layout_probe(device int8_t* A [[buffer(0)]], device int8_t* B [[buffer(1)]], device ushort* dump [[buffer(2)]], uint tid [[thread_index_in_threadgroup]]) {
    tensor<device int8_t, dextents<int32_t, 2>, tensor_inline> a0(A, dextents<int32_t, 2>(128, 64));
    tensor<device int8_t, dextents<int32_t, 2>, tensor_inline> b0(B, dextents<int32_t, 2>(64, 128));
    constexpr auto desc = matmul2d_descriptor(64, 64, 128, false, false, false, matmul2d_descriptor::mode::multiply);
    matmul2d<desc, execution_simdgroups<4>> op;
    auto c = op.get_destination_cooperative_tensor<decltype(a0), decltype(b0), int>();
    const uint cap = c.get_capacity();
    for (uint i = 0; i < cap; i++) { auto ix = c.get_multidimensional_index(i); dump[(tid * 64 + i) * 2] = c.is_valid_element(i) ? ushort(ix[0]) : ushort(9999); dump[(tid * 64 + i) * 2 + 1] = ushort(ix[1]); }
    if (tid == 0) dump[64 * 128 * 2] = ushort(cap);
}

// KV read-bandwidth probe: each (split, kvh) simdgroup streams `keys` rows of 256 halves of K and V. strided: cache [pos][4][256] (row stride 2 KiB); contiguous: [kvh][pos][256].
kernel void k_kv_read(device const half* kc [[buffer(0)]], device const half* vc [[buffer(1)]], device float* out [[buffer(2)]], constant uint4& prm [[buffer(3)]], uint2 tg [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]]) {
    const uint keysPer = prm.x, maxCtx = prm.y, contiguous = prm.z; const uint split = tg.x, kvh = tg.y;
    float acc = 0.0f;
    for (uint j = split * keysPer; j < (split + 1) * keysPer; j++) {
        const ulong off = contiguous ? (ulong(kvh) * maxCtx + j) * 256ul : ulong(j) * 1024ul + kvh * 256ul;
        const half4 k = *(device const half4*)(kc + off + lane * 8), k2 = *(device const half4*)(kc + off + lane * 8 + 4), v = *(device const half4*)(vc + off + lane * 8), v2 = *(device const half4*)(vc + off + lane * 8 + 4);
        acc += float(k.x) + float(k.w) + float(k2.y) + float(v.z) + float(v2.x) + float(v2.w);
    }
    out[(tg.y * 1024 + tg.x) * 32 + lane] = acc;
}

// Weight-read bandwidth probe: the native GGUF layout is read exactly like k_gemv_v1 (8 lanes per 28-byte block: one ushort, two bytes, one ushort per lane), versus the same number of bytes read as 16-byte vectors.
kernel void k_read_native(device const uchar* W [[buffer(0)]], device float* out [[buffer(1)]], constant uint& K [[buffer(2)]], constant uint& N [[buffer(3)]],
                          uint tgid [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
    const uint row0 = (tgid * 4 + sg) * 4; if (row0 >= N) return; const uint nblk = K / 128u, ix = lane >> 3, it = lane & 7u; float acc = 0.0f;
    for (int r = 0; r < 4; r++) { device const uchar* rp = W + ulong(min(row0 + uint(r), N - 1u)) * ulong(nblk) * 28ul;
        for (uint b = ix; b < nblk; b += 4) { device const uchar* bp = rp + b * 28u; const ushort w01 = *(device const ushort*)(bp + 2 * it); const uint u2 = bp[16 + it], uq = bp[24 + (it & 1u)]; const ushort ds = *(device const ushort*)(bp + 26); acc += float(w01) + float(u2) + float(uq) + float(ds); } }
    out[(tgid * 4 + sg) * 32 + lane] = acc;
}
kernel void k_read_wide(device const uchar* W [[buffer(0)]], device float* out [[buffer(1)]], constant uint& K [[buffer(2)]], constant uint& N [[buffer(3)]],
                        uint tgid [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
    const uint row0 = (tgid * 4 + sg) * 4; if (row0 >= N) return; const uint nblk = K / 128u; float acc = 0.0f;
    // the same rows, but each lane pulls 16 bytes at a time from the row's contiguous bytes (28 * nblk bytes, a multiple of 4 only, so read as uint4 from a 4-byte aligned address)
    for (int r = 0; r < 4; r++) { device const uchar* rp = W + ulong(min(row0 + uint(r), N - 1u)) * ulong(nblk) * 28ul; const uint nvec = (nblk * 28u) / 16u;
        for (uint v = lane; v < nvec; v += 32u) { const uint4 q = *(device const uint4*)(rp + v * 16u); acc += float(q.x ^ q.y) + float(q.z ^ q.w); } }
    out[(tgid * 4 + sg) * 32 + lane] = acc;
}
