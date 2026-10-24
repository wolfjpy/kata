// Gated DeltaNet decode step (one token): causal conv + SiLU, q/k L2-norm, delta-rule state update, gated RMSNorm.
#include "common.h"

// conv_out[c] = silu( sum_{i<4} w[4c+i] * x[t-3+i] ); the state holds the previous 3 inputs per channel ([3][10240]).
kernel void k_gdn_conv(device const float* qkv [[buffer(0)]], device float* state [[buffer(1)]], device const float* w [[buffer(2)]], device float* out [[buffer(3)]],
                       uint c [[thread_position_in_grid]]) {
    const uint C = bonsai::kGdnQkvDim; if (c >= C) return;
    float x0 = state[c], x1 = state[C + c], x2 = state[2 * C + c], x3 = qkv[c];
    float y = w[4 * c] * x0 + w[4 * c + 1] * x1 + w[4 * c + 2] * x2 + w[4 * c + 3] * x3;
    out[c] = silu(y);
    state[c] = x1; state[C + c] = x2; state[2 * C + c] = x3;
}

// One threadgroup per V head (48), 128 threads; thread v owns row v of the transposed state S[v][key] (128 fp32 in registers).
kernel void k_gdn_step(device const float* conv_out [[buffer(0)]], device const float* alpha [[buffer(1)]], device const float* braw [[buffer(2)]],
                       device const float* ssm_a [[buffer(3)]], device const float* dt [[buffer(4)]], device float* state [[buffer(5)]],
                       device const float* z [[buffer(6)]], device const float* norm_w [[buffer(7)]], device float* out [[buffer(8)]],
                       uint hv [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]], uint sg [[simdgroup_index_in_threadgroup]]) {
    threadgroup float qs[128]; threadgroup float ks[128]; threadgroup float part[8];
    const uint hk = hv % bonsai::kGdnKHeads;
    float qv = conv_out[hk * 128 + tid], kv = conv_out[bonsai::kGdnKHeads * 128 + hk * 128 + tid];
    float sq = simd_sum(qv * qv), sk = simd_sum(kv * kv);
    if (lane == 0) { part[sg] = sq; part[4 + sg] = sk; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float tq = part[0] + part[1] + part[2] + part[3], tk = part[4] + part[5] + part[6] + part[7];
    qs[tid] = qv / max(sqrt(tq), bonsai::kRmsEps); ks[tid] = kv / max(sqrt(tk), bonsai::kRmsEps);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float xg = alpha[hv] + dt[hv];
    float g = ssm_a[hv] * (xg > 20.0f ? xg : log(1.0f + exp(xg)));
    float ge = exp(g), beta = sigmoidf(braw[hv]);

    device float* srow = state + (ulong(hv) * 128 + tid) * 128;
    float st[128]; float skv = 0.0f;
    for (uint k = 0; k < 128; k++) { float s = srow[k] * ge; st[k] = s; skv += s * ks[k]; }
    float vin = conv_out[2 * bonsai::kGdnKHeads * 128 + hv * 128 + tid];
    float d = (vin - skv) * beta;
    float y = 0.0f;
    for (uint k = 0; k < 128; k++) { float s = st[k] + ks[k] * d; srow[k] = s; y += s * qs[k]; }
    float o = y * 0.08838834764831845f;     // 1/sqrt(128)

    // gated RMSNorm over this head's 128 values
    float so = simd_sum(o * o);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lane == 0) part[sg] = so;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float tot = part[0] + part[1] + part[2] + part[3];
    float n = o * rsqrt(tot / 128.0f + bonsai::kRmsEps) * norm_w[tid];
    out[hv * 128 + tid] = silu(z[hv * 128 + tid]) * n;
}
