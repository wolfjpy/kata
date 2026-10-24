#include <metal_stdlib>
#include "bonsai_m6.h"
using namespace metal;
// Smoke test: proves the shared config header compiles in MSL and the offline toolchain works.
kernel void smoke(device uint* out [[buffer(0)]], uint i [[thread_position_in_grid]]) { if (i == 0) out[0] = bonsai::kHidden + bonsai::kHadBlock; }
