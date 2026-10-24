// Reference-quality matrix-vector kernels (correctness first; optimised kernels come in later milestones).
#include "common.h"

// y[row] = sum_k W[row][k] * x[k]  for a PTQ1_0 matrix. One simdgroup per row, 8 rows per threadgroup.
kernel void k_gemv_ptq1(device const uchar* W [[buffer(0)]], device const float* x [[buffer(1)]], device float* y [[buffer(2)]],
                        constant uint& K [[buffer(3)]], constant uint& N [[buffer(4)]],
                        uint tgid [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
    uint row = tgid * 8 + sg; if (row >= N) return;
    uint nblk = K / bonsai::kGroup;
    device const uchar* rp = W + ulong(row) * ulong(nblk) * bonsai::kBlockBytes;
    float acc = 0.0f;
    for (uint b = lane; b < nblk; b += 32) {
        device const uchar* bp = rp + b * bonsai::kBlockBytes;
        float s = 0.0f;
        for (uint e = 0; e < 128; e++) s += float(ptq1_elem(bp, e)) * x[b * 128 + e];
        acc += ptq1_scale(bp) * s;
    }
    acc = simd_sum(acc);
    if (lane == 0) y[row] = acc;
}

// bf16 matrix [N][K] times fp32 vector.
kernel void k_gemv_bf16(device const bfloat* W [[buffer(0)]], device const float* x [[buffer(1)]], device float* y [[buffer(2)]],
                        constant uint& K [[buffer(3)]], constant uint& N [[buffer(4)]],
                        uint tgid [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
    uint row = tgid * 8 + sg; if (row >= N) return;
    device const bfloat* rp = W + ulong(row) * ulong(K);
    float acc = 0.0f;
    for (uint k = lane; k < K; k += 32) acc += float(rp[k]) * x[k];
    acc = simd_sum(acc);
    if (lane == 0) y[row] = acc;
}
