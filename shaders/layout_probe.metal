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
