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

// alpha/beta projections (bf16 [48][5120] each, applied to the plain normalized hidden state): one threadgroup per output row, 4 simdgroups split K.
kernel void k_gdn_ab(device const bfloat* Wa [[buffer(0)]], device const bfloat* Wb [[buffer(1)]], device const float* h [[buffer(2)]], device float* alpha [[buffer(3)]], device float* beta [[buffer(4)]],
                     uint row [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]], uint sg [[simdgroup_index_in_threadgroup]]) {
    threadgroup float part[4];
    const bool isb = row >= 48u; const uint r = isb ? row - 48u : row;
    device const bfloat* w = (isb ? Wb : Wa) + ulong(r) * 5120ul;
    float acc = 0.0f;
    for (uint k = sg * 1280u + lane; k < (sg + 1u) * 1280u; k += 32u) acc += float(w[k]) * h[k];
    acc = simd_sum(acc);
    if (lane == 0) part[sg] = acc;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) { float t = part[0] + part[1] + part[2] + part[3]; if (isb) beta[r] = t; else alpha[r] = t; }
}

// One threadgroup per V head (48), 128 threads; thread v owns column v of the key-major state S[key][v], so every state access is a coalesced 128-byte line.
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

    device float* scol = state + ulong(hv) * 128ul * 128ul + tid;
    float st[128]; float skv = 0.0f;
    for (uint k = 0; k < 128; k++) { float s = scol[k * 128u] * ge; st[k] = s; skv += s * ks[k]; }
    float vin = conv_out[2 * bonsai::kGdnKHeads * 128 + hv * 128 + tid];
    float d = (vin - skv) * beta;
    float y = 0.0f;
    for (uint k = 0; k < 128; k++) { float s = st[k] + ks[k] * d; scol[k * 128u] = s; y += s * qs[k]; }
    float o = y * 0.08838834764831845f;     // 1/sqrt(128)

    float so = simd_sum(o * o);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lane == 0) part[sg] = so;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float tot = part[0] + part[1] + part[2] + part[3];
    float n = o * rsqrt(tot / 128.0f + bonsai::kRmsEps) * norm_w[tid];
    out[hv * 128 + tid] = silu(z[hv * 128 + tid]) * n;
}
