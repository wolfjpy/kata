#include <metal_stdlib>
using namespace metal;
// Throughput of classic simdgroup_matrix 8x8x8 multiply-accumulate: 4 independent accumulators per simdgroup, operands loaded once.
template <typename T, typename TA>
inline void mma_loop(device float* out, uint iters, uint gid, uint lane) {
    simdgroup_matrix<T, 8, 8> A, B; simdgroup_matrix<TA, 8, 8> C0 = make_filled_simdgroup_matrix<TA, 8, 8>(TA(0)), C1 = C0, C2 = C0, C3 = C0;
    A = make_filled_simdgroup_matrix<T, 8, 8>(T(0.5f)); B = make_filled_simdgroup_matrix<T, 8, 8>(T(0.25f));
    for (uint i = 0; i < iters; i++) { simdgroup_multiply_accumulate(C0, A, B, C0); simdgroup_multiply_accumulate(C1, A, B, C1); simdgroup_multiply_accumulate(C2, A, B, C2); simdgroup_multiply_accumulate(C3, A, B, C3); }
    out[gid] = float(C0.thread_elements()[0]) + float(C1.thread_elements()[0]) + float(C2.thread_elements()[0]) + float(C3.thread_elements()[0]);
}
kernel void k_mma_h_f(device float* out [[buffer(0)]], constant uint& iters [[buffer(1)]], uint gid [[thread_position_in_grid]], uint lane [[thread_index_in_simdgroup]]) { mma_loop<half, float>(out, iters, gid, lane); }
kernel void k_mma_h_h(device float* out [[buffer(0)]], constant uint& iters [[buffer(1)]], uint gid [[thread_position_in_grid]], uint lane [[thread_index_in_simdgroup]]) { mma_loop<half, half>(out, iters, gid, lane); }
kernel void k_mma_f_f(device float* out [[buffer(0)]], constant uint& iters [[buffer(1)]], uint gid [[thread_position_in_grid]], uint lane [[thread_index_in_simdgroup]]) { mma_loop<float, float>(out, iters, gid, lane); }
