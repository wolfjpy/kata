// Batched (prefill) kernels. Path A: PTQ1_0 weights are expanded on the GPU to fp16 (w = scale * trit, exact) in a tile-major layout and
// multiplied with fp16 activations by the matrix unit (MPP matmul2d), accumulating in fp32.
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
#include "common.h"
using namespace mpp::tensor_ops;


// Grouped tile order: linear threadgroup id -> (m tile, n tile). Groups of `gm` consecutive M tiles walk the N tiles together, so one B strip is reused by gm tiles while their A rows stay in cache.
inline uint2 tile_order(uint id, uint Mt, uint Nt, uint gm) {
    if (gm == 0u) return uint2(id % Nt, id / Nt);                                    // (n, m): N-fast
    const uint perGroup = gm * Nt, grp = id / perGroup, firstM = grp * gm, sz = min(Mt - firstM, gm), inGrp = id % perGroup;
    return uint2(inGrp / sz, firstM + inGrp % sz);                                  // (n, m)
}

// ---- weight expansion: PTQ1_0 [N rows][K/128 blocks] -> fp16 tile-major [N/64][K][64]  (element [nt][k][n % 64]) ----
// Grid (K/128, N/64) threadgroups of 256 threads; one threadgroup expands 64 rows x 128 k and writes one contiguous 16 KiB tile.
kernel void k_expand_ptq1(device const uchar* W [[buffer(0)]], device half* out [[buffer(1)]], constant uint& K [[buffer(2)]],
                          uint2 tg [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {
    threadgroup half tile[128 * 64];
    const uint kb = tg.x, nt = tg.y, nblk = K / 128u;
    const uint r = tid >> 2, q = tid & 3u;
    device const uchar* bp = W + (ulong(nt * 64u + r) * ulong(nblk) + kb) * 28ul;
    const float d = ptq1_scale(bp);
    for (uint e = q * 32u; e < q * 32u + 32u; e++) tile[e * 64u + r] = half(float(ptq1_elem(bp, e)) * d);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    device half* o = out + (ulong(nt) * ulong(K) + ulong(kb) * 128ul) * 64ul;
    for (uint i = tid; i < 128u * 64u; i += 256u) o[i] = tile[i];
}

// ---- GEMM: C[M][N] (fp32) = A[M][K] (fp16) x W^T, with W in the tile-major layout above. tile 32x64, one simdgroup, N-fast grid. ----
kernel void k_gemm_f16(device half* A [[buffer(0)]], device half* B [[buffer(1)]], device float* C [[buffer(2)]], constant uint4& dims [[buffer(3)]], uint tgid1 [[threadgroup_position_in_grid]]) {
    const int M = int(dims.x), N = int(dims.y), K = int(dims.z); const uint2 tl = tile_order(tgid1, uint(M) / 32u, uint(N) / 64u, dims.w); const uint nt = tl.x, mt = tl.y;
    tensor<device half, dextents<int32_t, 2>, tensor_inline> tA(A, dextents<int32_t, 2>(K, M));
    tensor<device float, dextents<int32_t, 2>, tensor_inline> tC(C, dextents<int32_t, 2>(N, M));
    constexpr auto desc = matmul2d_descriptor(32, 64, static_cast<int>(dynamic_extent));
    matmul2d<desc, execution_simdgroups<1>> op;
    auto mA = tA.slice(0, int(mt) * 32); auto mC = tC.slice(int(nt) * 64, int(mt) * 32);
    tensor<device half, dextents<int32_t, 2>, tensor_inline> tB(B + ulong(nt) * ulong(K) * 64ul, dextents<int32_t, 2>(64, K));
    op.run(mA, tB, mC);
}

// ---- batched embedding: grid (5, M) ----
kernel void k_embed_b(device const uchar* W [[buffer(0)]], device const int* toks [[buffer(1)]], device const float* sg [[buffer(2)]], device float* out [[buffer(3)]],
                      uint2 tg [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
    threadgroup float xs[1024];
    const uint blk = tg.x, t = tg.y;
    device const uchar* row = W + ulong(toks[t]) * ulong(bonsai::kHidden / bonsai::kGroup * bonsai::kBlockBytes);
    uint i = tid * 4; uint g = i >> 7; uint e = i & 127u;
    device const uchar* b = row + (blk * 8 + g) * bonsai::kBlockBytes;
    float d = ptq1_scale(b);
    float4 v = float4(float(ptq1_elem(b, e)), float(ptq1_elem(b, e + 1)), float(ptq1_elem(b, e + 2)), float(ptq1_elem(b, e + 3))) * d;
    v = fwht1024(v, tid, lane, xs);
    uint o = t * bonsai::kHidden + blk * 1024 + i;
    out[o] = v.x * sg[blk * 1024 + i]; out[o + 1] = v.y * sg[blk * 1024 + i + 1]; out[o + 2] = v.z * sg[blk * 1024 + i + 2]; out[o + 3] = v.w * sg[blk * 1024 + i + 3];
}

// ---- batched fused residual-add + RMSNorm + sign + FWHT, fp16 output; grid (5, M) ----
kernel void k_add_rms_fwht_b(device const float* x [[buffer(0)]], device const float* y [[buffer(1)]], device const float* gs [[buffer(2)]], device half* out [[buffer(3)]],
                             device float* xnew [[buffer(4)]], device const float* gamma [[buffer(5)]], device float* hout [[buffer(6)]], constant uint& flags [[buffer(7)]],
                             uint2 tg [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]], uint sg [[simdgroup_index_in_threadgroup]]) {
    threadgroup float xs[1024]; threadgroup float part[8];
    const uint W = bonsai::kHidden, blk = tg.x, t = tg.y; const ulong ro = ulong(t) * W;
    float ss = 0.0f;
    for (uint i = tid; i < W; i += 256) { float v = x[ro + i] + ((flags & 1u) ? y[ro + i] : 0.0f); ss += v * v; }
    ss = simd_sum(ss);
    if (lane == 0) part[sg] = ss;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float tot = 0.0f; for (uint s = 0; s < 8; s++) tot += part[s];
    float inv = rsqrt(tot / float(W) + bonsai::kRmsEps);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint base = blk * 1024 + tid * 4; float vals[4];
    for (uint r = 0; r < 4; r++) {
        uint idx = base + r; float v = x[ro + idx] + ((flags & 1u) ? y[ro + idx] : 0.0f);
        xnew[ro + idx] = v; vals[r] = v * inv * gs[idx];
        if (flags & 2u) hout[ro + idx] = v * inv * gamma[idx];
    }
    float4 o = fwht1024(float4(vals[0], vals[1], vals[2], vals[3]), tid, lane, xs);
    out[ro + base] = half(o.x); out[ro + base + 1] = half(o.y); out[ro + base + 2] = half(o.z); out[ro + base + 3] = half(o.w);
}

// ---- batched input transform (modes 1..3 of k_rms_fwht), fp16 output; grid (W/1024, M); token rows have stride W ----
kernel void k_rms_fwht_b(device const float* src [[buffer(0)]], device const float* src2 [[buffer(1)]], device const float* gs [[buffer(2)]], device half* out [[buffer(3)]],
                         constant uint& W [[buffer(4)]], constant uint& mode [[buffer(5)]],
                         uint2 tg [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
    threadgroup float xs[1024];
    const uint blk = tg.x, t = tg.y; const ulong ro = ulong(t) * W;
    uint base = blk * 1024 + tid * 4; float vals[4];
    for (uint r = 0; r < 4; r++) {
        uint idx = base + r; float v;
        if ((mode & 7u) == 1) { float g = src[ro + idx]; v = silu(g) * src2[ro + idx] * gs[idx]; }
        else if ((mode & 7u) == 2) { uint h = idx & 127u; uint tt = idx >> 7; uint rr = tt % 3u; uint k = tt / 3u; v = src[ro + h + 128u * (k + 16u * rr)] * gs[idx]; }
        else v = src[ro + idx] * gs[idx];
        vals[r] = v;
    }
    float4 o = fwht1024(float4(vals[0], vals[1], vals[2], vals[3]), tid, lane, xs);
    if (mode & 8u) { out[base * 2 + t] = half(o.x); out[(base + 1) * 2 + t] = half(o.y); out[(base + 2) * 2 + t] = half(o.z); out[(base + 3) * 2 + t] = half(o.w); }   // [K] of half2 (two verified tokens)
    else { out[ro + base] = half(o.x); out[ro + base + 1] = half(o.y); out[ro + base + 2] = half(o.z); out[ro + base + 3] = half(o.w); }
}

kernel void k_add2(device const float* x [[buffer(0)]], device const float* y [[buffer(1)]], device float* out [[buffer(2)]], constant uint& n [[buffer(3)]], uint i [[thread_position_in_grid]]) { if (i < n) out[i] = x[i] + y[i]; }

// ---- batched GDN ----
// alpha/beta for every token: grid (96, M), 128 threads
kernel void k_gdn_ab_b(device const bfloat* Wa [[buffer(0)]], device const bfloat* Wb [[buffer(1)]], device const float* h [[buffer(2)]], device float* alpha [[buffer(3)]], device float* beta [[buffer(4)]],
                       uint2 tg [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]], uint sg [[simdgroup_index_in_threadgroup]]) {
    threadgroup float part[4];
    const uint row = tg.x, t = tg.y; const bool isb = row >= 48u; const uint r = isb ? row - 48u : row;
    device const bfloat* w = (isb ? Wb : Wa) + ulong(r) * 5120ul; device const float* hh = h + ulong(t) * 5120ul;
    float acc = 0.0f;
    for (uint k = sg * 1280u + lane; k < (sg + 1u) * 1280u; k += 32u) acc += float(w[k]) * hh[k];
    acc = simd_sum(acc);
    if (lane == 0) part[sg] = acc;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) { float s = part[0] + part[1] + part[2] + part[3]; if (isb) beta[t * 48 + r] = s; else alpha[t * 48 + r] = s; }
}

// causal conv over M tokens (thread per channel x token); the 3 inputs before the chunk come from the state
kernel void k_gdn_conv_b(device const float* qkv [[buffer(0)]], device const float* state [[buffer(1)]], device const float* w [[buffer(2)]], device float* out [[buffer(3)]],
                         uint2 gid [[thread_position_in_grid]]) {
    const uint C = bonsai::kGdnQkvDim; const uint c = gid.x, t = gid.y; if (c >= C) return;
    float x[4];
    for (uint i = 0; i < 4; i++) { int j = int(t) - 3 + int(i); x[i] = j >= 0 ? qkv[ulong(j) * C + c] : state[ulong(j + 3) * C + c]; }
    float y = w[4 * c] * x[0] + w[4 * c + 1] * x[1] + w[4 * c + 2] * x[2] + w[4 * c + 3] * x[3];
    out[ulong(t) * C + c] = silu(y);
}
// new conv state = the last 3 inputs of the chunk (shifting in the old state when the chunk is shorter than 3)
kernel void k_gdn_conv_state(device const float* qkv [[buffer(0)]], device float* state [[buffer(1)]], constant uint& M [[buffer(2)]], uint c [[thread_position_in_grid]]) {
    const uint C = bonsai::kGdnQkvDim; if (c >= C) return;
    float old0 = state[c], old1 = state[C + c], old2 = state[2 * C + c], nw[3];
    for (uint i = 0; i < 3; i++) { int j = int(M) - 3 + int(i); nw[i] = j >= 0 ? qkv[ulong(j) * C + c] : (j == -3 ? old0 : j == -2 ? old1 : old2); }
    // careful: j+3 indexes the old state: j=-1 -> old2, j=-2 -> old1, j=-3 -> old0
    state[c] = nw[0]; state[C + c] = nw[1]; state[2 * C + c] = nw[2];
}

// delta-rule recurrence over M tokens; the 128x128 state of one V head stays in registers for the whole chunk
kernel void k_gdn_step_b(device const float* conv_out [[buffer(0)]], device const float* alpha [[buffer(1)]], device const float* braw [[buffer(2)]],
                         device const float* ssm_a [[buffer(3)]], device const float* dt [[buffer(4)]], device float* state [[buffer(5)]],
                         device const float* z [[buffer(6)]], device const float* norm_w [[buffer(7)]], device float* out [[buffer(8)]], constant uint& M [[buffer(9)]],
                         uint hv [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]], uint sg [[simdgroup_index_in_threadgroup]]) {
    threadgroup float qs[128]; threadgroup float ks[128]; threadgroup float part[8];
    const uint hk = hv % bonsai::kGdnKHeads; const uint C = bonsai::kGdnQkvDim;
    device float* scol = state + ulong(hv) * 128ul * 128ul + tid;
    float st[128];
    for (uint k = 0; k < 128; k++) st[k] = scol[k * 128u];
    const float a = ssm_a[hv], dtb = dt[hv], nw = norm_w[tid];
    for (uint t = 0; t < M; t++) {
        device const float* cp = conv_out + ulong(t) * C;
        float qv = cp[hk * 128 + tid], kv = cp[bonsai::kGdnKHeads * 128 + hk * 128 + tid];
        float sq = simd_sum(qv * qv), sk = simd_sum(kv * kv);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (lane == 0) { part[sg] = sq; part[4 + sg] = sk; }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float tq = part[0] + part[1] + part[2] + part[3], tk = part[4] + part[5] + part[6] + part[7];
        qs[tid] = qv / max(sqrt(tq), bonsai::kRmsEps); ks[tid] = kv / max(sqrt(tk), bonsai::kRmsEps);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float xg = alpha[t * 48 + hv] + dtb;
        float g = a * (xg > 20.0f ? xg : log(1.0f + exp(xg)));
        float ge = exp(g), beta = sigmoidf(braw[t * 48 + hv]);
        float skv = 0.0f;
        for (uint k = 0; k < 128; k++) { st[k] *= ge; skv += st[k] * ks[k]; }
        float d = (cp[2 * bonsai::kGdnKHeads * 128 + hv * 128 + tid] - skv) * beta;
        float y = 0.0f;
        for (uint k = 0; k < 128; k++) { st[k] += ks[k] * d; y += st[k] * qs[k]; }
        float o = y * 0.08838834764831845f;
        float so = simd_sum(o * o);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (lane == 0) part[sg] = so;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float tot = part[0] + part[1] + part[2] + part[3];
        float n = o * rsqrt(tot / 128.0f + bonsai::kRmsEps) * nw;
        out[ulong(t) * 6144ul + hv * 128 + tid] = silu(z[ulong(t) * 6144ul + hv * 128 + tid]) * n;
    }
    for (uint k = 0; k < 128; k++) scol[k * 128u] = st[k];
}

// ---- batched attention ----
// grid (32, M): same work as k_attn_prep for token t at position pos0 + t
kernel void k_attn_prep_b(device const float* qfull [[buffer(0)]], device const float* kproj [[buffer(1)]], device const float* vproj [[buffer(2)]],
                          device const float* qnorm [[buffer(3)]], device const float* knorm [[buffer(4)]],
                          device float* qout [[buffer(5)]], device float* gate_out [[buffer(6)]],
                          device half* kcache [[buffer(7)]], device half* vcache [[buffer(8)]], constant uint& pos0 [[buffer(9)]], constant uint& h0 [[buffer(10)]], constant uint& cacheOff [[buffer(11)]],
                          uint2 tgp [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]], uint sg [[simdgroup_index_in_threadgroup]]) {
    threadgroup float xs[256]; threadgroup float part[8];
    const uint h = tgp.x + h0, t = tgp.y; const uint pos = pos0 + t;
    device const float* qf = qfull + ulong(t) * 12288ul; device const float* kp = kproj + ulong(t) * 1024ul; device const float* vp = vproj + ulong(t) * 1024ul;
    device float* qo = qout + ulong(t) * 6144ul; device float* go = gate_out + ulong(t) * 6144ul;
    if (h >= 28) { uint vh = h - 28; vcache[ulong(pos - cacheOff) * bonsai::kKvDim + vh * 256 + tid] = half(vp[vh * 256 + tid]); return; }
    float v; device const float* w;
    if (h < 24) { v = qf[h * 512 + tid]; go[h * 256 + tid] = qf[h * 512 + 256 + tid]; w = qnorm; }
    else { v = kp[(h - 24) * 256 + tid]; w = knorm; }
    float ss = simd_sum(v * v);
    if (lane == 0) part[sg] = ss;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float tot = 0.0f; for (uint s = 0; s < 8; s++) tot += part[s];
    xs[tid] = v * rsqrt(tot / 256.0f + bonsai::kRmsEps) * w[tid];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float outv = xs[tid];
    if (tid < 32) {
        float theta = float(pos) * pow(bonsai::kRopeTheta, -float(tid) / 32.0f);
        float c = cos(theta), s = sin(theta); float a = xs[tid], b = xs[tid + 32];
        outv = a * c - b * s; float outv2 = a * s + b * c;
        if (h < 24) qo[h * 256 + tid + 32] = outv2; else kcache[ulong(pos - cacheOff) * bonsai::kKvDim + (h - 24) * 256 + tid + 32] = half(outv2);
    }
    if (tid >= 32 && tid < 64) { } else { if (h < 24) qo[h * 256 + tid] = outv; else kcache[ulong(pos - cacheOff) * bonsai::kKvDim + (h - 24) * 256 + tid] = half(outv); }
}

// split attention for a batch of tokens: grid (S, 4, Mtok); token t has npos = pos0 + tokBase + t + 1 valid positions
kernel void k_attn_split_b(device const float* q [[buffer(0)]], device const half* kcache [[buffer(1)]], device const half* vcache [[buffer(2)]],
                           device float* pm [[buffer(3)]], device float* pl [[buffer(4)]], device float* pacc [[buffer(5)]],
                           constant uint& posBase [[buffer(6)]],      // position of the first token of this sub-batch
                           constant uint& firstPos [[buffer(7)]],     // first valid cache position (0 for the main model, 1 for the MTP cache)
                           uint3 tg [[threadgroup_position_in_grid]], uint3 ng [[threadgroups_per_grid]], uint tid [[thread_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]], uint sg [[simdgroup_index_in_threadgroup]]) {
    threadgroup float tm[8][6]; threadgroup float tl[8][6]; threadgroup float tacc[8][3][256];
    const uint split = tg.x, kvh = tg.y, t = tg.z, S = ng.x; const uint npos = posBase + t + 1;
    const uint span = npos - firstPos, per = (span + S - 1) / S, lo = firstPos + min(split * per, span), hi = min(lo + per, npos);
    const uint sper = (hi - lo + 7) / 8, slo = min(lo + sg * sper, hi), shi = min(slo + sper, hi);
    device const float* qt = q + ulong(t) * 6144ul;
    float qv[6][8];
    for (uint h = 0; h < 6; h++) for (uint d = 0; d < 8; d++) qv[h][d] = qt[(kvh * 6 + h) * 256 + lane * 8 + d] * 0.0625f;
    float m[6], l[6], acc[6][8];
    for (uint h = 0; h < 6; h++) { m[h] = -INFINITY; l[h] = 0.0f; for (uint d = 0; d < 8; d++) acc[h][d] = 0.0f; }
    for (uint j = slo; j < shi; j++) {
        device const half* kp = kcache + ulong(j) * bonsai::kKvDim + kvh * 256 + lane * 8;
        device const half* vp = vcache + ulong(j) * bonsai::kKvDim + kvh * 256 + lane * 8;
        half4 k0 = *(device const half4*)kp, k1 = *(device const half4*)(kp + 4), v0 = *(device const half4*)vp, v1 = *(device const half4*)(vp + 4);
        float kf[8] = {float(k0.x), float(k0.y), float(k0.z), float(k0.w), float(k1.x), float(k1.y), float(k1.z), float(k1.w)};
        float vf[8] = {float(v0.x), float(v0.y), float(v0.z), float(v0.w), float(v1.x), float(v1.y), float(v1.z), float(v1.w)};
        for (uint h = 0; h < 6; h++) {
            float p = 0.0f; for (uint d = 0; d < 8; d++) p += qv[h][d] * kf[d];
            float s = simd_sum(p);
            float mn = max(m[h], s), a = exp(m[h] - mn), e = exp(s - mn);
            l[h] = l[h] * a + e; for (uint d = 0; d < 8; d++) acc[h][d] = acc[h][d] * a + e * vf[d];
            m[h] = mn;
        }
    }
    if (lane == 0) { for (uint h = 0; h < 6; h++) { tm[sg][h] = m[h]; tl[sg][h] = l[h]; } }
    for (uint hg = 0; hg < 2; hg++) {
        for (uint hh = 0; hh < 3; hh++) for (uint d = 0; d < 8; d++) tacc[sg][hh][lane * 8 + d] = acc[hg * 3 + hh][d];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint hh = 0; hh < 3; hh++) {
            const uint h = hg * 3 + hh;
            float M = -INFINITY; for (uint s = 0; s < 8; s++) M = max(M, tm[s][h]);
            float L = 0.0f, A = 0.0f;
            for (uint s = 0; s < 8; s++) { float w = (tm[s][h] == -INFINITY) ? 0.0f : exp(tm[s][h] - M); L += tl[s][h] * w; A += tacc[s][hh][tid] * w; }
            const uint gh = kvh * 6 + h; const ulong ti = ulong(t) * 24ul + gh;
            pacc[(ti * S + split) * 256 + tid] = A;
            if (tid == 0) { pm[ti * S + split] = M; pl[ti * S + split] = L; }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}
// grid (24, Mtok)
kernel void k_attn_combine_b(device const float* pm [[buffer(0)]], device const float* pl [[buffer(1)]], device const float* pacc [[buffer(2)]], device const float* gate [[buffer(3)]],
                             device float* out [[buffer(4)]], constant uint& S [[buffer(5)]], uint2 tg [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {
    const uint h = tg.x, t = tg.y; const ulong ti = ulong(t) * 24ul + h;
    float M = -INFINITY; for (uint s = 0; s < S; s++) M = max(M, pm[ti * S + s]);
    float L = 0.0f, A = 0.0f;
    for (uint s = 0; s < S; s++) { float mm = pm[ti * S + s]; float w = (mm == -INFINITY) ? 0.0f : exp(mm - M); L += pl[ti * S + s] * w; A += pacc[(ti * S + s) * 256 + tid] * w; }
    out[ulong(t) * 6144ul + h * 256 + tid] = (A / L) * sigmoidf(gate[ulong(t) * 6144ul + h * 256 + tid]);
}

// =====================================================================================================================
// Prefill v2: restructured non-GEMM kernels.
// ---- one threadgroup per token: residual add + RMSNorm + sign + FWHT over all five 1024-blocks (fp16 out) + optional plain normalised copy (fp16) ----
kernel void k_add_rms_fwht_t(device const float* x [[buffer(0)]], device const float* y [[buffer(1)]], device const float* gs [[buffer(2)]], device half* out [[buffer(3)]],
                             device float* xnew [[buffer(4)]], device const float* gamma [[buffer(5)]], device half* hout [[buffer(6)]], constant uint& flags [[buffer(7)]],
                             uint t [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]], uint sg [[simdgroup_index_in_threadgroup]]) {
    threadgroup float v[5120]; threadgroup float xs[1024]; threadgroup float part[8];
    const uint W = bonsai::kHidden; const ulong ro = ulong(t) * W;
    float ss = 0.0f;
    for (uint i = tid; i < W; i += 256) { float a = x[ro + i] + ((flags & 1u) ? y[ro + i] : 0.0f); v[i] = a; xnew[ro + i] = a; ss += a * a; }
    ss = simd_sum(ss);
    if (lane == 0) part[sg] = ss;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float tot = 0.0f; for (uint s = 0; s < 8; s++) tot += part[s];
    const float inv = rsqrt(tot / float(W) + bonsai::kRmsEps);
    for (uint blk = 0; blk < 5; blk++) {
        uint base = blk * 1024 + tid * 4; float vals[4];
        for (uint r = 0; r < 4; r++) { uint idx = base + r; vals[r] = v[idx] * inv * gs[idx]; if (flags & 2u) hout[ro + idx] = half(v[idx] * inv * gamma[idx]); }
        float4 o = fwht1024(float4(vals[0], vals[1], vals[2], vals[3]), tid, lane, xs);
        if (flags & 4u) { out[(base) * 2 + t] = half(o.x); out[(base + 1) * 2 + t] = half(o.y); out[(base + 2) * 2 + t] = half(o.z); out[(base + 3) * 2 + t] = half(o.w); }   // two-token verify: [K] of half2
        else { out[ro + base] = half(o.x); out[ro + base + 1] = half(o.y); out[ro + base + 2] = half(o.z); out[ro + base + 3] = half(o.w); }
    }
}

// ---- conv: thread = (4 channels, a run of 16 tokens), sliding window over the 3 previous inputs ----
kernel void k_gdn_conv_t(device const float* qkv [[buffer(0)]], device const float* state [[buffer(1)]], device const float* w [[buffer(2)]], device float* out [[buffer(3)]], constant uint& M [[buffer(4)]],
                         uint2 gid [[thread_position_in_grid]]) {
    const uint C = bonsai::kGdnQkvDim; const uint c4 = gid.x, t0 = gid.y * 16u; if (c4 * 4u >= C || t0 >= M) return;
    const uint c = c4 * 4u;
    float4 w0 = *(device const float4*)(w + 4 * (c + 0)), w1 = *(device const float4*)(w + 4 * (c + 1)), w2 = *(device const float4*)(w + 4 * (c + 2)), w3 = *(device const float4*)(w + 4 * (c + 3));
    float4 x0, x1, x2;   // inputs at t-3, t-2, t-1
    if (t0 >= 3) { x0 = *(device const float4*)(qkv + ulong(t0 - 3) * C + c); x1 = *(device const float4*)(qkv + ulong(t0 - 2) * C + c); x2 = *(device const float4*)(qkv + ulong(t0 - 1) * C + c); }
    else { x0 = *(device const float4*)(state + c); x1 = *(device const float4*)(state + C + c); x2 = *(device const float4*)(state + 2 * C + c);
           if (t0 == 1) { x0 = x1; x1 = x2; x2 = *(device const float4*)(qkv + c); } else if (t0 == 2) { x0 = x2; x1 = *(device const float4*)(qkv + c); x2 = *(device const float4*)(qkv + C + c); } }
    const uint tend = min(t0 + 16u, M);
    for (uint t = t0; t < tend; t++) {
        float4 x3 = *(device const float4*)(qkv + ulong(t) * C + c);
        float4 y = float4(dot(w0, float4(x0.x, x1.x, x2.x, x3.x)), dot(w1, float4(x0.y, x1.y, x2.y, x3.y)), dot(w2, float4(x0.z, x1.z, x2.z, x3.z)), dot(w3, float4(x0.w, x1.w, x2.w, x3.w)));
        *(device float4*)(out + ulong(t) * C + c) = float4(silu(y.x), silu(y.y), silu(y.z), silu(y.w));
        x0 = x1; x1 = x2; x2 = x3;
    }
}

// ---- L2-normalise q and k per head for all tokens: grid (M, 32), 128 threads ----
kernel void k_gdn_qk_b(device const float* conv [[buffer(0)]], device float* qk [[buffer(1)]], uint2 tg [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]], uint sg [[simdgroup_index_in_threadgroup]]) {
    threadgroup float part[4];
    const uint t = tg.x, j = tg.y; const ulong o = ulong(t) * 4096ul + j * 128 + tid;
    float v = conv[ulong(t) * bonsai::kGdnQkvDim + j * 128 + tid];
    float s = simd_sum(v * v); if (lane == 0) part[sg] = s;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float tot = part[0] + part[1] + part[2] + part[3];
    qk[o] = v / max(sqrt(tot), bonsai::kRmsEps);
}
// ---- decay and beta per (token, v-head); alpha/beta come from one GEMM with 128 output columns: [alpha(48) | beta(48) | pad] ----
kernel void k_gdn_gates_b(device const float* ab [[buffer(0)]], device const float* ssm_a [[buffer(1)]], device const float* dt [[buffer(2)]], device float* gb [[buffer(3)]], constant uint& M [[buffer(4)]], uint i [[thread_position_in_grid]]) {
    if (i >= M * 48u) return; const uint t = i / 48u, hv = i % 48u;
    float xg = ab[t * 128u + hv] + dt[hv]; float g = ssm_a[hv] * (xg > 20.0f ? xg : log(1.0f + exp(xg)));
    gb[t * 96u + hv] = exp(g); gb[t * 96u + 48u + hv] = sigmoidf(ab[t * 128u + 48u + hv]);
}
// ---- delta-rule recurrence: one simdgroup per (V head, 32 value columns); the state column lives in registers over the whole chunk ----
kernel void k_gdn_rec_b(device const float* conv [[buffer(0)]], device const float* qk [[buffer(1)]], device const float* gb [[buffer(2)]], device float* state [[buffer(3)]],
                        device float* raw [[buffer(4)]], constant uint& M [[buffer(5)]], uint2 tg [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]]) {
    const uint hv = tg.x, cg = tg.y, hk = hv % bonsai::kGdnKHeads, v = cg * 32u + lane;
    device float* scol = state + ulong(hv) * 16384ul + v;
    float st[128];
    for (uint k = 0; k < 128; k++) st[k] = scol[k * 128u];
    for (uint t = 0; t < M; t++) {
        device const float4* qp = (device const float4*)(qk + ulong(t) * 4096ul + hk * 128);
        device const float4* kp = (device const float4*)(qk + ulong(t) * 4096ul + 2048ul + hk * 128);
        const float ge = gb[t * 96u + hv], beta = gb[t * 96u + 48u + hv];
        float skv = 0.0f;
        for (uint k4 = 0; k4 < 32; k4++) { float4 kk = kp[k4]; st[4*k4] *= ge; st[4*k4+1] *= ge; st[4*k4+2] *= ge; st[4*k4+3] *= ge; skv += st[4*k4] * kk.x + st[4*k4+1] * kk.y + st[4*k4+2] * kk.z + st[4*k4+3] * kk.w; }
        const float d = (conv[ulong(t) * bonsai::kGdnQkvDim + 4096u + hv * 128 + v] - skv) * beta;
        float y = 0.0f;
        for (uint k4 = 0; k4 < 32; k4++) { float4 kk = kp[k4], qq = qp[k4]; st[4*k4] += kk.x * d; st[4*k4+1] += kk.y * d; st[4*k4+2] += kk.z * d; st[4*k4+3] += kk.w * d; y += st[4*k4] * qq.x + st[4*k4+1] * qq.y + st[4*k4+2] * qq.z + st[4*k4+3] * qq.w; }
        raw[ulong(t) * 6144ul + hv * 128 + v] = y * 0.08838834764831845f;
    }
    for (uint k = 0; k < 128; k++) scol[k * 128u] = st[k];
}
// ---- gated RMSNorm per (token, head): grid (M, 48), 128 threads ----
kernel void k_gdn_gnorm_b(device const float* raw [[buffer(0)]], device const float* z [[buffer(1)]], device const float* norm_w [[buffer(2)]], device float* out [[buffer(3)]],
                          uint2 tg [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]], uint sg [[simdgroup_index_in_threadgroup]]) {
    threadgroup float part[4];
    const uint t = tg.x, hv = tg.y; const ulong i = ulong(t) * 6144ul + hv * 128 + tid;
    float o = raw[i]; float s = simd_sum(o * o); if (lane == 0) part[sg] = s;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float tot = part[0] + part[1] + part[2] + part[3];
    out[i] = silu(z[i]) * (o * rsqrt(tot / 128.0f + bonsai::kRmsEps) * norm_w[tid]);
}

// =====================================================================================================================
// Path B (int8): ternary weights as int8 trits + per-(row, 128-group) fp32 scales; activations quantised to int8 with one scale per token.
// out = SX[m] * sum_g SW[n][g] * (sum_{k in g} A8[m][k] * T[n][k])      (exact int32 accumulation per group, fp32 across groups)
// ---- weight expansion: PTQ1_0 -> int8 trits tile-major [N/64][K][64], scales fp32 [N/64][K/128][64] ----
kernel void k_expand_i8(device const uchar* W [[buffer(0)]], device char* out [[buffer(1)]], device float* sw [[buffer(2)]], constant uint& K [[buffer(3)]],
                        uint2 tg [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {
    threadgroup char tile[128 * 64];
    const uint kb = tg.x, nt = tg.y, nblk = K / 128u;
    const uint r = tid >> 2, q = tid & 3u;
    device const uchar* bp = W + (ulong(nt * 64u + r) * ulong(nblk) + kb) * 28ul;
    for (uint e = q * 32u; e < q * 32u + 32u; e++) tile[e * 64u + r] = char(ptq1_elem(bp, e));
    if (q == 0) sw[(ulong(nt) * nblk + kb) * 64ul + r] = ptq1_scale(bp);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    device char* o = out + (ulong(nt) * ulong(K) + ulong(kb) * 128ul) * 64ul;
    for (uint i = tid; i < 128u * 64u; i += 256u) o[i] = tile[i];
}
// ---- activation quantisation, one scale per (token, 128-element group): fp16 [M][K] -> int8 [M][K] + scales sx[g][M] (group-major so the GEMM epilogue reads consecutive rows) ----
kernel void k_quant_rows(device const half* x [[buffer(0)]], device char* q [[buffer(1)]], device float* sx [[buffer(2)]], constant uint& K [[buffer(3)]], constant uint& ldS [[buffer(4)]],
                         uint t [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]], uint sg [[simdgroup_index_in_threadgroup]]) {
    device const half* xr = x + ulong(t) * K; device char* qr = q + ulong(t) * K; const uint G = K / 128u;
    for (uint g = sg; g < G; g += 8u) {
        const half4 v = *(device const half4*)(xr + g * 128u + lane * 4u);
        float mx = max(max(fabs(float(v.x)), fabs(float(v.y))), max(fabs(float(v.z)), fabs(float(v.w)))); mx = simd_max(mx);
        const float scale = mx > 0.0f ? mx / 127.0f : 1.0f, inv = 1.0f / scale;
        char4 o = char4(char(clamp(rint(float(v.x) * inv), -127.0f, 127.0f)), char(clamp(rint(float(v.y) * inv), -127.0f, 127.0f)), char(clamp(rint(float(v.z) * inv), -127.0f, 127.0f)), char(clamp(rint(float(v.w) * inv), -127.0f, 127.0f)));
        *(device char4*)(qr + g * 128u + lane * 4u) = o;
        if (lane == 0) sx[ulong(g) * ldS + t] = scale;
    }
}
// ---- int8 GEMM with the per-group scale epilogue; tile 64x64, 4 simdgroups, N-fast grid. M must be a multiple of 64. ----
kernel void k_gemm_i8(device int8_t* A [[buffer(0)]], device int8_t* B [[buffer(1)]], device float* SW [[buffer(2)]], device float* SX [[buffer(3)]], device float* C [[buffer(4)]],
                      constant uint4& dims [[buffer(5)]], uint tgid1 [[threadgroup_position_in_grid]]) {
    const int M = int(dims.x), N = int(dims.y), K = int(dims.z); const uint mode = dims.w & 0xffu; const uint2 tl = tile_order(tgid1, uint(M) / 64u, uint(N) / 64u, dims.w >> 8); const int nt = int(tl.x), mt = int(tl.y); const int G = K / 128;
    tensor<device float, dextents<int32_t, 2>, tensor_inline> tC(C, dextents<int32_t, 2>(N, M));
    constexpr auto desc = matmul2d_descriptor(64, 64, 128, false, false, false, matmul2d_descriptor::mode::multiply);
    matmul2d<desc, execution_simdgroups<4>> op;
    array<int, 2> strA = {1, K};
    tensor<device int8_t, dextents<int32_t, 2>, tensor_inline> a0(A + ulong(mt) * 64ul * ulong(K), dextents<int32_t, 2>(128, 64), strA);
    tensor<device int8_t, dextents<int32_t, 2>, tensor_inline> b0(B + ulong(nt) * ulong(K) * 64ul, dextents<int32_t, 2>(64, 128));
    auto cInt = op.get_destination_cooperative_tensor<decltype(a0), decltype(b0), int>();
    auto cAcc = op.get_destination_cooperative_tensor<decltype(a0), decltype(b0), float>();
    const uint16_t cap = cAcc.get_capacity(); uint16_t col[64], row[64];
    for (uint16_t i = 0; i < cap; i++) { cAcc[i] = 0.0f; auto ix = cAcc.get_multidimensional_index(i); col[i] = uint16_t(ix[0]); row[i] = uint16_t(ix[1]); }
    // Element layout of the destination (verified by tools/layout_probe): element i = ((rowBlock*2 + colBlock)*2 + rowHalf)*4 + c, i.e. columns c0+c and c0+32+c, rows r0, r0+8, r0+32, r0+40.
    // So a lane needs only eight distinct weight scales (two float4 reads) and four activation scales per K group.
    const uint c0 = col[0], r0 = row[0];
    bool vec = (cap == 32) && (mode == 0) && ((c0 & 3u) == 0u);
    for (uint16_t i = 0; i < 32 && vec; i++) if (col[i] != c0 + (i & 3u) + ((i >> 3) & 1u) * 32u || row[i] != r0 + ((i >> 2) & 1u) * 8u + ((i >> 4) & 1u) * 32u) vec = false;
    for (int g = 0; g < G; g++) {
        tensor<device int8_t, dextents<int32_t, 2>, tensor_inline> ag(A + ulong(mt) * 64ul * ulong(K) + g * 128, dextents<int32_t, 2>(128, 64), strA);
        tensor<device int8_t, dextents<int32_t, 2>, tensor_inline> bg(B + ulong(nt) * ulong(K) * 64ul + ulong(g) * 128ul * 64ul, dextents<int32_t, 2>(64, 128));
        op.run(ag, bg, cInt);
        device const float* swg = SW + (ulong(nt) * G + g) * 64ul; device const float* sxg = SX + ulong(g) * ulong(M) + ulong(mt) * 64ul;
        if (vec) {
            const float4 sa = *(device const float4*)(swg + c0), sb = *(device const float4*)(swg + c0 + 32u);
            const float x0 = sxg[r0], x1 = sxg[r0 + 8u], x2 = sxg[r0 + 32u], x3 = sxg[r0 + 40u];
            for (uint16_t i = 0; i < 32; i++) { const float4 sw = ((i >> 3) & 1u) ? sb : sa; const float sxv = ((i >> 4) & 1u) ? (((i >> 2) & 1u) ? x3 : x2) : (((i >> 2) & 1u) ? x1 : x0); cAcc[i] += float(cInt[i]) * (sw[i & 3u] * sxv); }
        }
        else { for (uint16_t i = 0; i < cap; i++) if (cInt.is_valid_element(i)) cAcc[i] += float(cInt[i]) * (swg[col[i]] * sxg[row[i]]); }
    }
    auto mC = tC.slice(nt * 64, mt * 64);
    cAcc.store(mC);
}

// ---- delta-rule recurrence v2: 4 lanes share one state column (32 keys each, so the state fits in registers without spilling) ----
// One simdgroup per (V head, 8 value columns); grid (48, 16).
kernel void k_gdn_rec_b2(device const float* conv [[buffer(0)]], device const float* qk [[buffer(1)]], device const float* gb [[buffer(2)]], device float* state [[buffer(3)]],
                         device float* raw [[buffer(4)]], constant uint& M [[buffer(5)]], uint2 tg [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]]) {
    const uint hv = tg.x, cg = tg.y, hk = hv % bonsai::kGdnKHeads, part = lane & 3u, v = cg * 8u + (lane >> 2);
    device float* scol = state + ulong(hv) * 16384ul + ulong(part) * 32ul * 128ul + v;
    float st[32];
    for (uint k = 0; k < 32; k++) st[k] = scol[k * 128u];
    for (uint t = 0; t < M; t++) {
        device const float4* qp = (device const float4*)(qk + ulong(t) * 4096ul + hk * 128 + part * 32);
        device const float4* kp = (device const float4*)(qk + ulong(t) * 4096ul + 2048ul + hk * 128 + part * 32);
        const float ge = gb[t * 96u + hv], beta = gb[t * 96u + 48u + hv];
        float skv = 0.0f;
        for (uint k4 = 0; k4 < 8; k4++) { float4 kk = kp[k4]; st[4*k4] *= ge; st[4*k4+1] *= ge; st[4*k4+2] *= ge; st[4*k4+3] *= ge; skv += st[4*k4] * kk.x + st[4*k4+1] * kk.y + st[4*k4+2] * kk.z + st[4*k4+3] * kk.w; }
        skv += simd_shuffle_xor(skv, 1); skv += simd_shuffle_xor(skv, 2);
        const float d = (conv[ulong(t) * bonsai::kGdnQkvDim + 4096u + hv * 128 + v] - skv) * beta;
        float y = 0.0f;
        for (uint k4 = 0; k4 < 8; k4++) { float4 kk = kp[k4], qq = qp[k4]; st[4*k4] += kk.x * d; st[4*k4+1] += kk.y * d; st[4*k4+2] += kk.z * d; st[4*k4+3] += kk.w * d; y += st[4*k4] * qq.x + st[4*k4+1] * qq.y + st[4*k4+2] * qq.z + st[4*k4+3] * qq.w; }
        y += simd_shuffle_xor(y, 1); y += simd_shuffle_xor(y, 2);
        if (part == 0) raw[ulong(t) * 6144ul + hv * 128 + v] = y * 0.08838834764831845f;
    }
    for (uint k = 0; k < 32; k++) scol[k * 128u] = st[k];
}

// =====================================================================================================================
// Weight expansion v2: no threadgroup staging. Thread = (row group of 4 rows, k-slice of 8) -> 16 x 16 = 256 threads per 64-row x 128-k tile.
// A slice [8s, 8s+8) of a block always shares one trit digit n over 8 consecutive bytes (s<10: n=s/2, bytes (s&1)*8..+7; s=10..14: n=s-10, bytes 16..23; s=15: qh, 2 bytes x 4 digits),
// so the decode is a uniform float "floor(3^(n+1) u) - 3 floor(3^n u)" and each thread stores four rows' trits for one k as a single 32-bit word.
inline float p3n(uint n) { return n == 0 ? 1.0f : n == 1 ? 3.0f : n == 2 ? 9.0f : n == 3 ? 27.0f : n == 4 ? 81.0f : 243.0f; }
inline void ptq1_slice(device const uchar* bp, uint s, thread float* t) {
    if (s < 15) {
        const uint n = s < 10 ? s >> 1 : s - 10, off = s < 10 ? (s & 1u) * 8u : 16u;
        const uint2 w = uint2(*(device const uint*)(bp + off), *(device const uint*)(bp + off + 4));
        const float a = p3n(n) * (1.0f / 256.0f), b = p3n(n + 1) * (1.0f / 256.0f);
        for (uint j = 0; j < 8; j++) { const float u = float((j < 4 ? w.x >> (8 * j) : w.y >> (8 * (j - 4))) & 0xffu); t[j] = floor(u * b) - 3.0f * floor(u * a) - 1.0f; }
    } else {
        const ushort h = *(device const ushort*)(bp + 24);
        for (uint j = 0; j < 8; j++) { const uint n = j >> 1, hb = j & 1u; const float u = float((h >> (8 * hb)) & 0xffu); t[j] = floor(u * p3n(n + 1) * (1.0f / 256.0f)) - 3.0f * floor(u * p3n(n) * (1.0f / 256.0f)) - 1.0f; }
    }
}
kernel void k_expand_i8_v2(device const uchar* W [[buffer(0)]], device char* out [[buffer(1)]], device float* sw [[buffer(2)]], constant uint& K [[buffer(3)]],
                           uint2 tg [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {
    const uint kb = tg.x, nt = tg.y, nblk = K / 128u; const uint rg = tid & 15u, s = tid >> 4;
    float t[4][8];
    for (uint r = 0; r < 4; r++) {
        device const uchar* bp = W + (ulong(nt * 64u + rg * 4u + r) * ulong(nblk) + kb) * 28ul;
        ptq1_slice(bp, s, t[r]);
        if (s == 0) sw[(ulong(nt) * nblk + kb) * 64ul + rg * 4u + r] = ptq1_scale(bp);
    }
    device char* o = out + (ulong(nt) * ulong(K) + ulong(kb) * 128ul + s * 8ul) * 64ul + rg * 4u;
    for (uint j = 0; j < 8; j++) {
        const uint w = (uint(uchar(char(t[0][j]))) ) | (uint(uchar(char(t[1][j]))) << 8) | (uint(uchar(char(t[2][j]))) << 16) | (uint(uchar(char(t[3][j]))) << 24);
        *(device uint*)(o + ulong(j) * 64ul) = w;
    }
}
kernel void k_expand_ptq1_v2(device const uchar* W [[buffer(0)]], device half* out [[buffer(1)]], constant uint& K [[buffer(2)]],
                             uint2 tg [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {
    const uint kb = tg.x, nt = tg.y, nblk = K / 128u; const uint rg = tid & 15u, s = tid >> 4;
    float t[4][8]; float d[4];
    for (uint r = 0; r < 4; r++) { device const uchar* bp = W + (ulong(nt * 64u + rg * 4u + r) * ulong(nblk) + kb) * 28ul; ptq1_slice(bp, s, t[r]); d[r] = ptq1_scale(bp); }
    device half* o = out + (ulong(nt) * ulong(K) + ulong(kb) * 128ul + s * 8ul) * 64ul + rg * 4u;
    for (uint j = 0; j < 8; j++) { *(device half4*)(o + ulong(j) * 64ul) = half4(half(t[0][j] * d[0]), half(t[1][j] * d[1]), half(t[2][j] * d[2]), half(t[3][j] * d[3])); }
}

// ---- GDN recurrence for a verify window: like k_gdn_rec_b2, but the state is never committed in place; instead the state after token t is written to snapshot t
// (snapshot[t] = state after tokens 0..t of the window, snapshot[0] = the input state). The caller picks the snapshot matching the accepted prefix.
kernel void k_gdn_rec_snap(device const float* conv [[buffer(0)]], device const float* qk [[buffer(1)]], device const float* gb [[buffer(2)]], device float* stateIn [[buffer(3)]],
                           device float* snaps [[buffer(4)]], device float* raw [[buffer(5)]], constant uint& M [[buffer(6)]], constant uint& snapStride [[buffer(7)]],
                           uint2 tg [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]]) {
    const uint hv = tg.x, cg = tg.y, hk = hv % bonsai::kGdnKHeads, part = lane & 3u, v = cg * 8u + (lane >> 2);
    const ulong off = ulong(hv) * 16384ul + ulong(part) * 32ul * 128ul + v;
    float st[32];
    for (uint k = 0; k < 32; k++) st[k] = stateIn[off + k * 128u];
    for (uint t = 0; t < M; t++) {
        device const float4* qp = (device const float4*)(qk + ulong(t) * 4096ul + hk * 128 + part * 32);
        device const float4* kp = (device const float4*)(qk + ulong(t) * 4096ul + 2048ul + hk * 128 + part * 32);
        const float ge = gb[t * 96u + hv], beta = gb[t * 96u + 48u + hv];
        float skv = 0.0f;
        for (uint k4 = 0; k4 < 8; k4++) { float4 kk = kp[k4]; st[4*k4] *= ge; st[4*k4+1] *= ge; st[4*k4+2] *= ge; st[4*k4+3] *= ge; skv += st[4*k4] * kk.x + st[4*k4+1] * kk.y + st[4*k4+2] * kk.z + st[4*k4+3] * kk.w; }
        skv += simd_shuffle_xor(skv, 1); skv += simd_shuffle_xor(skv, 2);
        const float d = (conv[ulong(t) * bonsai::kGdnQkvDim + 4096u + hv * 128 + v] - skv) * beta;
        float y = 0.0f;
        for (uint k4 = 0; k4 < 8; k4++) { float4 kk = kp[k4], qq = qp[k4]; st[4*k4] += kk.x * d; st[4*k4+1] += kk.y * d; st[4*k4+2] += kk.z * d; st[4*k4+3] += kk.w * d; y += st[4*k4] * qq.x + st[4*k4+1] * qq.y + st[4*k4+2] * qq.z + st[4*k4+3] * qq.w; }
        y += simd_shuffle_xor(y, 1); y += simd_shuffle_xor(y, 2);
        if (part == 0) raw[ulong(t) * 6144ul + hv * 128 + v] = y * 0.08838834764831845f;
        device float* so = snaps + (ulong(t) + 1ul) * snapStride;
        for (uint k = 0; k < 32; k++) so[off + k * 128u] = st[k];
    }
}

// ---- conv state after accepting F tokens of a verify window: the window's first F qkv rows extend the old 3-column history (oldest first).  c in [0, 10240). ----
kernel void k_gdn_conv_snap(device const float* qkv [[buffer(0)]], device const float* old [[buffer(1)]], device float* out [[buffer(2)]], constant uint& F [[buffer(3)]], uint c [[thread_position_in_grid]]) {
    const uint C = bonsai::kGdnQkvDim; if (c >= C) return;
    // logical history is [x(t-3), x(t-2), x(t-1)] with the window's tokens appended after it
    for (uint k = 0; k < 3; k++) {
        const int idx = int(F) + int(k) - 3;                       // index into the extended history: <0 -> old state
        out[k * C + c] = idx < 0 ? old[(uint)(idx + 3) * C + c] : qkv[(ulong)idx * C + c];
    }
}
