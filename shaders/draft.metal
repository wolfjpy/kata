// DFlash2 draft model: Q4_K / Q6_K dequantisation (validated element by element against ggml's dequantize_row_q4_K/q6_K).
#include <metal_stdlib>
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace metal;
using namespace mpp::tensor_ops;

// Q4_K: 256 weights per 144-byte super-block: half d, half dmin, 12 packed 6-bit (scale, min) pairs, 128 nibble bytes.
inline void q4k_scale_min(uint j, device const uchar* q, thread uint& sc, thread uint& m) {
    if (j < 4) { sc = q[j] & 63u; m = q[j + 4] & 63u; }
    else { sc = (q[j + 4] & 0x0Fu) | ((q[j - 4] >> 6) << 4); m = (q[j + 4] >> 4) | ((q[j] >> 6) << 4); }
}
kernel void k_dq_q4k(device const uchar* src [[buffer(0)]], device float* dst [[buffer(1)]], constant uint& K [[buffer(2)]], constant uint& blkIdx [[buffer(3)]], uint e [[thread_position_in_grid]]) {
    if (e >= 256u) return;
    device const uchar* bp = src + ulong(blkIdx) * 144ul;
    const float d = float(as_type<half>(*(device const ushort*)(bp + 0)));
    const float dmin = float(as_type<half>(*(device const ushort*)(bp + 2)));
    // element e: j = e / 64 (0..3) selects a 64-weight group; inside it the low nibbles are the first 32 weights, high nibbles the next 32
    const uint j = e >> 6, l = e & 63u, sub = l >> 5, pos = j * 2u + sub;          // pos = 2*(j) + (0 for low nibble, 1 for high)
    uint sc, m; q4k_scale_min(pos, bp + 4, sc, m);
    // q pointer: groups of 64 weights consume 32 bytes; within a group the low/high nibble cover 32 weights each
    const uint qbyte = (e >> 6) * 32u + (l & 31u);   // one 64-weight group consumes 32 nibble bytes
    const uint nib = (sub == 0) ? (bp[16 + qbyte] & 0x0Fu) : (bp[16 + qbyte] >> 4);
    dst[e] = d * float(sc) * float(nib) - dmin * float(m);
}

// Q6_K: 256 weights per 210-byte super-block: 128 ql, 64 qh, 16 int8 scales, half d.
kernel void k_dq_q6k(device const uchar* src [[buffer(0)]], device float* dst [[buffer(1)]], constant uint& K [[buffer(2)]], constant uint& blkIdx [[buffer(3)]], uint e [[thread_position_in_grid]]) {
    if (e >= 256u) return;
    device const uchar* bp = src + ulong(blkIdx) * 210ul;
    device const uchar* ql = bp; device const uchar* qh = bp + 128; device const char* sc = (device const char*)(bp + 192);
    const float d = float(as_type<half>(*(device const ushort*)(bp + 208)));
    // reference loop: n = 0,128; l = 0..31 producing y[l], y[l+32], y[l+64], y[l+96] with scales sc[is+0,2,4,6], is = l/16
    const uint n = e >> 7, r = e & 127u;                    // which 128-half, and which of the 128 outputs
    const uint l = r & 31u, q4 = r >> 5;                   // q4 = 0..3 -> y[l], y[l+32], y[l+64], y[l+96]
    device const uchar* qln = ql + n * 64, * qhn = qh + n * 32; device const char* scn = sc + n * 8;
    const uint is = l >> 4;
    uint v;
    if (q4 == 0) v = (qln[l] & 0x0Fu) | (((qhn[l] >> 0) & 3u) << 4);
    else if (q4 == 1) v = (qln[l + 32] & 0x0Fu) | (((qhn[l] >> 2) & 3u) << 4);
    else if (q4 == 2) v = (qln[l] >> 4) | (((qhn[l] >> 4) & 3u) << 4);
    else v = (qln[l + 32] >> 4) | (((qhn[l] >> 6) & 3u) << 4);
    const int qq = int(v) - 32;
    dst[e] = d * float(scn[is + 2u * q4]) * float(qq);
}

// =====================================================================================================================
// Draft matrix multiply: MT token rows (the block's 8 columns) x N output rows, Q4_K / Q6_K weights decoded once into a private fp16 tile and multiplied on the matrix unit.
// Same tile geometry as k_wide: N rows per simdgroup, KC=128 weights per matmul2d call, NSG simdgroups per threadgroup, column-major tile with stride KC+8.
// Q4_K job: one 16-byte piece of a 64-weight group = 16 low-nibble values (elements g*64 + 0..15) and 16 high-nibble values (elements g*64 + 32..47).
// QT: 0 = Q4_K, 1 = Q6_K, 2 = Q2_K, 3 = Q3_K.  A 128-weight chunk is one half of a 256-weight super-block for all four.
inline void q2k_chunk(device const uchar* bp, uint half_, uint jobAll, uint rows, threadgroup half* dst, uint LD) {
    // 8 groups of 16 weights; job = (row, group). Job layout matches q4k: 4 jobs per row per chunk.
    const uint r = jobAll / 8u, g = jobAll % 8u;                        // 8 groups per chunk
    if (r >= rows) return;
    device const uchar* q = bp + 64u - 64u;                             // qs at offset 16
    device const uchar* qs = bp + 16;
    const uint sc = bp[half_ * 8u + g];                                 // scales: 16 bytes, 8 per half
    const float d = float(as_type<half>(*(device const ushort*)(bp + 80)));
    const float dmin = float(as_type<half>(*(device const ushort*)(bp + 82)));
    const float dl = d * float(sc & 0x0Fu), ml = dmin * float(sc >> 4);
    const uint j = g >> 1, sh = 2u * j, h = g & 1u;                      // 4 shifts x 2 halves of 16
    threadgroup half* out = dst + r * LD + g * 16u;
    for (uint l = 0; l < 16; l++) {
        const uint code = (qs[half_ * 32u + l + h * 16u] >> sh) & 3u;
        out[l] = half(dl * float(code) - ml);
    }
    (void)q;
}
inline void q3k_chunk(device const uchar* bp, uint half_, uint jobAll, uint rows, threadgroup half* dst, uint LD) {
    const uint r = jobAll / 8u, g = jobAll % 8u;
    if (r >= rows) return;
    const float d = float(as_type<half>(*(device const ushort*)(bp + 108)));
    // unpack the 16 int8 scales from the 12 packed bytes
    const uint k1 = 0x03030303u, k2 = 0x0f0f0f0fu;
    uint aux[4]; for (uint i = 0; i < 3; i++) aux[i] = *(device const uint*)(bp + 96 + 4u * i); aux[3] = 0;
    const uint t0 = aux[2];
    uint a0 = (aux[0] & k2) | (((t0 >> 0) & k1) << 4), a1 = (aux[1] & k2) | (((t0 >> 2) & k1) << 4);
    uint a2 = ((aux[0] >> 4) & k2) | (((t0 >> 4) & k1) << 4), a3 = ((aux[1] >> 4) & k2) | (((t0 >> 6) & k1) << 4);
    const uint words[4] = {a0, a1, a2, a3};
    const uint is = half_ * 8u + g;   // 8 scales per 128-weight half
    const uint sc = (words[is >> 2] >> (8u * (is & 3u))) & 0xffu;
    const float dl = d * (float(sc) - 32.0f);
    const uint j = g >> 1, sh = 2u * j, h = g & 1u;
    const uchar m = (uchar)(1u << (half_ * 4u + j));
    threadgroup half* out = dst + r * LD + g * 16u;
    for (uint l = 0; l < 16; l++) {
        const uint idx = half_ * 32u + l + h * 16u;
        const int code = int((bp[32u + idx] >> sh) & 3u) - ((bp[l + h * 16u] & m) ? 0 : 4);   // qs starts after the 32-byte hmask
        out[l] = half(dl * float(code));
    }
}

template <int MT, int N, int NSG, int QT>
kernel void k_draft_mm(device const uchar* W [[buffer(0)]], device half* x [[buffer(1)]], device float* y [[buffer(2)]], constant uint& K [[buffer(3)]], constant uint& Nout [[buffer(4)]],
                       uint tgid [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]], uint tid [[thread_index_in_threadgroup]]) {
    constexpr int KC = 128, LD = KC + 8;
    threadgroup half Bt[NSG][N * LD];
    const uint row0 = (tgid * NSG + sg) * N; if (row0 >= Nout) return;
    const uint nblk = K / 256u;                                   // 256-weight super-blocks
    constexpr auto desc = matmul2d_descriptor(MT, N, KC, false, true, false, matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<desc, execution_simdgroup> op;
    array<int, 2> sA = {1, int(K)}, sB = {1, LD};
    tensor<device half, dextents<int32_t, 2>, tensor_inline> tA0(x, dextents<int32_t, 2>(KC, MT), sA);
    tensor<threadgroup half, dextents<int32_t, 2>, tensor_inline> tB(&Bt[sg][0], dextents<int32_t, 2>(KC, N), sB);
    auto cC = op.template get_destination_cooperative_tensor<decltype(tA0), decltype(tB), float>();
    for (uint i = 0; i < cC.get_capacity(); i++) cC[i] = 0.0f;
    const uint half1 = nblk * 2u;                                 // KC=128 chunks (two per super-block)
    for (uint kc = 0; kc < half1; kc++) {
        const uint sb = kc >> 1, halfSel = kc & 1u;               // super-block and which 128-weight half
        if (QT == 0) {
            // Q4_K: 4 jobs per row per chunk (each 32 of the chunk's 128 weights); lanes: 32 = N rows x (32/N) jobs
            constexpr uint JOBS_PER_ROW = 4, PER_LANE = (N * JOBS_PER_ROW) / 32u;
            for (uint j = 0; j < PER_LANE; j++) {
                const uint jobAll = lane + j * 32u, r = jobAll / JOBS_PER_ROW, jr = jobAll % JOBS_PER_ROW;   // r = row, jr = 32-weight piece
                device const uchar* bp = W + (ulong(min(row0 + r, Nout - 1u)) * ulong(nblk) + sb) * 144ul;
                const float d = float(as_type<half>(*(device const ushort*)(bp + 0))), dmin = float(as_type<half>(*(device const ushort*)(bp + 2)));
                const uint g = halfSel * 2u + (jr >> 1), piece = (jr & 1u) * 16u;              // 64-weight group, and which 16 of its 32 bytes
                uint sc, m, sc2, mh; q4k_scale_min(2u * g, bp + 4, sc, m); q4k_scale_min(2u * g + 1u, bp + 4, sc2, mh);   // low / high nibbles have separate scales
                const uint4 v = *(device const uint4*)(bp + 16 + g * 32u + piece);
                const uint w4[4] = {v.x, v.y, v.z, v.w};
                threadgroup half* out = &Bt[sg][r * LD + (jr >> 1) * 64u];
                const float d1 = d * float(sc), m1 = dmin * float(m), d2 = d * float(sc2), m2 = dmin * float(mh);
                for (uint k = 0; k < 4; k++) {                                                    // 4 bytes -> 4 low + 4 high values
                    half4 lo, hi; lo.x = half(d1 * float(w4[k] & 0x0Fu) - m1); lo.y = half(d1 * float((w4[k] >> 8) & 0x0Fu) - m1);
                    lo.z = half(d1 * float((w4[k] >> 16) & 0x0Fu) - m1); lo.w = half(d1 * float((w4[k] >> 24) & 0x0Fu) - m1);
                    hi.x = half(d2 * float((w4[k] >> 4) & 0x0Fu) - m2); hi.y = half(d2 * float((w4[k] >> 12) & 0x0Fu) - m2);
                    hi.z = half(d2 * float((w4[k] >> 20) & 0x0Fu) - m2); hi.w = half(d2 * float((w4[k] >> 28) & 0x0Fu) - m2);
                    *(threadgroup half4*)(out + piece + 4 * k) = lo;                              // elements g*64 + piece + 4k .. +3
                    *(threadgroup half4*)(out + piece + 32 + 4 * k) = hi;                         // elements g*64 + 32 + piece + 4k .. +3
                }
            }
        } else if (QT == 4) {   // PQ2_0: 34 bytes per 128 weights; lane decodes the 4 codes of its own byte
            const uint rowBlocks = nblk * 2u;
            for (uint r = 0; r < N; r++) {
                device const uchar* bp = W + (ulong(min(row0 + r, Nout - 1u)) * ulong(rowBlocks) + kc) * 34ul;
                const float d = float(as_type<half>(*(device const ushort*)(bp)));
                const uint byte = bp[2 + lane];
                threadgroup half* o = &Bt[sg][r * LD + lane * 4u];
                o[0] = half(d * float(int(byte & 3u) - 1));
                o[1] = half(d * float(int((byte >> 2) & 3u) - 1));
                o[2] = half(d * float(int((byte >> 4) & 3u) - 1));
                o[3] = half(d * float(int((byte >> 6) & 3u) - 1));
            }
        } else if (QT == 2) {
            device const uchar* bp_of = W + (ulong(min(row0, Nout - 1u)) * ulong(nblk) + sb) * 84ul; (void)bp_of;
            for (uint rr = 0; rr < N; rr++) for (uint jobAll = lane; jobAll < 8u; jobAll += 32u) {
                device const uchar* bp = W + (ulong(min(row0 + rr, Nout - 1u)) * ulong(nblk) + sb) * 84ul; q2k_chunk(bp, halfSel, rr * 8u + jobAll, N + 1u, &Bt[sg][0], LD); }
        } else if (QT == 3) {
            for (uint rr = 0; rr < N; rr++) for (uint jobAll = lane; jobAll < 8u; jobAll += 32u) {
                device const uchar* bp = W + (ulong(min(row0 + rr, Nout - 1u)) * ulong(nblk) + sb) * 110ul; q3k_chunk(bp, halfSel, rr * 8u + jobAll, N + 1u, &Bt[sg][0], LD); }
        } else {
            // Q6_K: a 128-weight chunk is one half of a super-block; the reference order is y[l], y[l+32], y[l+64], y[l+96] for l = 0..31,
            // so one job = the 32 elements with a fixed quarter q4 (32 ql bytes + 32 qh bytes + 2 scales).
            constexpr uint JOBS_PER_ROW = 4;                          // quarters; 32 elements each
            for (uint jobAll = lane; jobAll < N * JOBS_PER_ROW; jobAll += 32u) {
                const uint r = jobAll / JOBS_PER_ROW, q4 = jobAll % JOBS_PER_ROW;
                device const uchar* bp = W + (ulong(min(row0 + r, Nout - 1u)) * ulong(nblk) + sb) * 210ul;
                const float d = float(as_type<half>(*(device const ushort*)(bp + 208)));
                device const uchar* ql = bp + halfSel * 64u + (((q4 == 1u) || (q4 == 3u)) ? 32u : 0u);   // q4 1/3 read ql[l+32], q4 0/2 read ql[l]
                device const uint4* qhp = (device const uint4*)(bp + 128 + halfSel * 32u);
                device const char* sc = (device const char*)(bp + 192 + halfSel * 8u);
                const uint sh = (q4 == 0u) ? 0u : (q4 == 1u) ? 2u : (q4 == 2u) ? 4u : 6u;  // qh bit pair
                const bool hi = (q4 >= 2u);
                const float dsc0 = d * float(sc[0 + 2u * q4]), dsc1 = d * float(sc[1 + 2u * q4]);
                threadgroup half* out = &Bt[sg][r * LD + q4 * 32u];
                for (uint e = 0; e < 32; e++) {
                    const uint qlb = ql[e];
                    const uint nib = hi ? (qlb >> 4) : (qlb & 0x0Fu);
                    const uint hb = (((device const uchar*)qhp)[e] >> sh) & 3u;
                    const int q = int(nib | (hb << 4)) - 32;
                    out[e] = half((e < 16 ? dsc0 : dsc1) * float(q));
                }
            }
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);
        tensor<device half, dextents<int32_t, 2>, tensor_inline> tA(x + kc * KC, dextents<int32_t, 2>(KC, MT), sA);
        op.run(tA, tB, cC);
        simdgroup_barrier(mem_flags::mem_threadgroup);
    }
    for (uint i = 0; i < cC.get_capacity(); i++) if (cC.is_valid_element(i)) { auto ix = cC.get_multidimensional_index(i); const uint col = row0 + uint(ix[0]); if (col < Nout) y[ulong(ix[1]) * Nout + col] = cC[i]; }
}
#define DMM(MT, N, SG, QT) template [[host_name("k_draft_mm_m" #MT "_n" #N "_g" #SG "_q" #QT)]] kernel void k_draft_mm<MT, N, SG, QT>(device const uchar*, device half*, device float*, constant uint&, constant uint&, uint, uint, uint, uint);
DMM(8, 16, 4, 0) DMM(8, 16, 4, 1) DMM(8, 16, 4, 2) DMM(8, 16, 4, 3) DMM(8, 16, 4, 4) DMM(16, 16, 4, 0) DMM(16, 16, 4, 1)

// =====================================================================================================================
// DFlash2 draft layer ops.
// Dynamic grouped conv (2 taps, group 16): out[t][c] = sum_tap (base[side][tap][c] + delta[t][g(c)][tap][side]) * X[t-tap][c], X[-1] = 0.
// delta is the conv_proj output, 1280 per token ordered [group(320)][tap(2)][side(2)].
kernel void k_draft_conv(device const float* x [[buffer(0)]], device const float* dyn [[buffer(1)]], device const float* base [[buffer(2)]], device float* out [[buffer(3)]],
                         constant uint& T [[buffer(4)]], constant uint& W [[buffer(5)]], constant uint& side [[buffer(6)]], uint2 gid [[thread_position_in_grid]]) {
    const uint t = gid.y, c = gid.x; if (t >= T || c >= W) return;
    const uint nGroups = W / 16u, g = c / 16u;
    float acc = 0.0f;
    for (uint tap = 0; tap < 2u; tap++) {
        if (tap > t) break;
        // dynamic is [n_groups][kernel][2 sides][token] in ggml (group fastest), so the index is g + nGroups*(tap + 2*side)
        const float b = base[(side * 2u + tap) * W + c], d = dyn[ulong(t) * (nGroups * 4u) + g + nGroups * (tap + 2u * side)];
        acc += (b + d) * x[ulong(t - tap) * W + c];
    }
    out[ulong(t) * W + c] = acc;
}

// Per-head RMSNorm + partial RoPE (first 64 of 128 dims, pairs (i, i+32)) for the draft's Q and K, and the K/V ring write.
// grid (heads, T), 128 threads. qOut is [T][heads*128]; the ring holds [pos][heads_kv*128] for K and V.
kernel void k_draft_qk(device const float* qin [[buffer(0)]], device const float* kin [[buffer(1)]], device const float* vin [[buffer(2)]], device const float* qnorm [[buffer(3)]], device const float* knorm [[buffer(4)]],
                       device float* qout [[buffer(5)]], device half* kring [[buffer(6)]], device half* vring [[buffer(7)]], constant uint& pos0 [[buffer(8)]], constant uint& H [[buffer(9)]], constant uint& HKV [[buffer(10)]],
                       uint2 tg [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]], uint sg [[simdgroup_index_in_threadgroup]]) {
    threadgroup float part[4]; threadgroup float xs[128];
    const uint h = tg.x, t = tg.y; const uint pos = pos0 + t;
    if (h >= H + HKV) {   // V head: copy straight into the ring (no norm, no rope)
        const uint vh = h - (H + HKV); vring[ulong(pos) * (HKV * 128u) + vh * 128u + tid] = half(vin[ulong(t) * (HKV * 128u) + vh * 128u + tid]); return;
    }
    const bool isK = h >= H; const uint hh = isK ? h - H : h;   // [0,H) = Q, [H, H+HKV) = K, [H+HKV, H+2*HKV) = V
    device const float* src = isK ? kin : qin; device const float* w = isK ? knorm : qnorm;
    const float v = src[ulong(t) * ((isK ? HKV : H) * 128u) + hh * 128u + tid];
    float ss = simd_sum(v * v); if (lane == 0) part[sg] = ss;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float tot = part[0] + part[1] + part[2] + part[3];
    xs[tid] = v * rsqrt(tot / 128.0f + 1e-6f) * w[tid];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float o = xs[tid];
    // MROPE sections [64,0,0,0]: the sections sum to n_rot/2, so n_rot = 128 (the whole head), NEOX pairs (i, i+64),
    // theta_i = pos * 1e7^(-2i/128). Threads 0..63 each produce one pair.
    if (tid < 64) {
        const float theta = float(pos) * pow(10000000.0f, -float(tid) / 64.0f);
        const float c = cos(theta), s = sin(theta), a = xs[tid], b = xs[tid + 64];
        o = a * c - b * s;
        const float o2 = a * s + b * c;
        if (isK) kring[ulong(pos) * (HKV * 128u) + hh * 128u + tid + 64] = half(o2); else qout[ulong(t) * (H * 128u) + hh * 128u + tid + 64] = o2;
    }
    if (isK) kring[ulong(pos) * (HKV * 128u) + hh * 128u + tid] = half(o); else qout[ulong(t) * (H * 128u) + hh * 128u + tid] = o;
}

// Non-causal sliding-window attention over the ring: query row t (position pos0+t) attends to keys in [pos-2047, pos+W-1] that exist.
// One threadgroup per (head, row), 128 threads; keys are scanned from the ring in position order with an online softmax.
kernel void k_draft_attn(device const float* q [[buffer(0)]], device const half* kring [[buffer(1)]], device const half* vring [[buffer(2)]], device float* out [[buffer(3)]],
                         constant uint& pos0 [[buffer(4)]], constant uint& T [[buffer(5)]], constant uint& H [[buffer(6)]], constant uint& HKV [[buffer(7)]], constant uint& win [[buffer(8)]], constant uint& ring [[buffer(9)]],
                         uint2 tg [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
    threadgroup float red[32];
    const uint h = tg.x, t = tg.y, kvh = h / (H / HKV);
    const uint pos = pos0 + t, lo = pos + 1u > win ? pos + 1u - win : 0u, hi = pos0 + T - 1u;   // non-causal: up to the end of the block
    const float scale = 1.0f / sqrt(128.0f);
    float qv = q[ulong(t) * (H * 128u) + h * 128u + tid];
    float m = -INFINITY, l = 0.0f, acc = 0.0f;
    for (uint kp = lo; kp <= hi; kp++) {
        device const half* krow = kring + ulong(kp % ring) * (HKV * 128u) + kvh * 128u;
        float s = qv * float(krow[tid]);
        s = simd_sum(s);
        if (lane == 0) red[tid / 32] = s;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid == 0) { float tot = 0.0f; for (uint i = 0; i < 4; i++) tot += red[i]; red[0] = tot * scale; }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        const float sc = red[0];
        const float mn = max(m, sc), a = exp(m - mn), e = exp(sc - mn);
        l = l * a + e;
        acc = acc * a + e * float((vring + ulong(kp % ring) * (HKV * 128u) + kvh * 128u)[tid]);
        m = mn;
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    out[ulong(t) * (H * 128u) + h * 128u + tid] = acc / l;
}

// Context K/V materialisation for the draft ring: K = rope(k_norm(wk c_t)) at position t, V = wv c_t. grid (2*HKV, T), 128 threads.
kernel void k_draft_ctx_kv(device const float* kproj [[buffer(0)]], device const float* vproj [[buffer(1)]], device const float* knorm [[buffer(2)]],
                           device half* kring [[buffer(3)]], device half* vring [[buffer(4)]], constant uint& pos0 [[buffer(5)]], constant uint& HKV [[buffer(6)]], constant uint& ring [[buffer(7)]],
                           uint2 tg [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]], uint sg [[simdgroup_index_in_threadgroup]]) {
    threadgroup float part[4]; threadgroup float xs[128];
    const bool isV = tg.x >= HKV; const uint hh = isV ? tg.x - HKV : tg.x, t = tg.y; const uint pos = pos0 + t;
    const uint row = (uint)pos % ring;
    if (isV) { vring[ulong(row) * (HKV * 128u) + hh * 128u + tid] = half(vproj[ulong(t) * (HKV * 128u) + hh * 128u + tid]); return; }
    const float v = kproj[ulong(t) * (HKV * 128u) + hh * 128u + tid];
    float ss = simd_sum(v * v); if (lane == 0) part[sg] = ss;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float tot = part[0] + part[1] + part[2] + part[3];
    xs[tid] = v * rsqrt(tot / 128.0f + 1e-6f) * knorm[tid];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float o = xs[tid];
    if (tid < 64) { const float theta = float(pos) * pow(10000000.0f, -float(tid) / 64.0f);
        const float c = cos(theta), s = sin(theta), a = xs[tid], b = xs[tid + 64];
        o = a * c - b * s; kring[ulong(row) * (HKV * 128u) + hh * 128u + tid + 64] = half(a * s + b * c); }
    kring[ulong(row) * (HKV * 128u) + hh * 128u + tid] = half(o);
}

// float -> half conversion (activations handed to k_draft_mm and to the shared PTQ1_0 head).
kernel void k_f2h(device const float* src [[buffer(0)]], device half* dst [[buffer(1)]], constant uint& n [[buffer(2)]], uint i [[thread_position_in_grid]]) { if (i < n) dst[i] = half(src[i]); }

// PQ2_0 probe: decode one 128-weight block to floats for validation against ggml's dequantize_row_pq2_0.
kernel void k_dq_pq2(device const uchar* src [[buffer(0)]], device float* dst [[buffer(1)]], constant uint& K [[buffer(2)]], constant uint& blkIdx [[buffer(3)]], uint e [[thread_position_in_grid]]) {
    if (e >= 128u) return;
    device const uchar* bp = src + ulong(blkIdx) * 34ul;
    const float d = float(as_type<half>(*(device const ushort*)(bp)));
    const uint code = (bp[2 + (e >> 2)] >> (2u * (e & 3u))) & 3u;
    dst[e] = d * float(int(code) - 1);
}

// BF16 matrix times fp16 activations: y[t][n] = sum_k W[n][k] * x[t][k]. One threadgroup per (row, token), 256 threads split over K.
kernel void k_draft_bf16mm(device const ushort* W [[buffer(0)]], device const half* x [[buffer(1)]], device float* y [[buffer(2)]], constant uint& K [[buffer(3)]], constant uint& N [[buffer(4)]], constant uint& T [[buffer(5)]],
                           uint2 tg [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]], uint sg [[simdgroup_index_in_threadgroup]]) {
    threadgroup float part[8];
    const uint row = tg.x, t = tg.y; if (row >= N || t >= T) return;
    device const ushort* w = W + ulong(row) * K; device const half* xx = x + ulong(t) * K;
    float acc = 0.0f;
    for (uint k = tid; k < K; k += 256u) { const uint bits = (uint)w[k] << 16; acc += as_type<float>(bits) * float(xx[k]); }
    acc = simd_sum(acc); if (lane == 0) part[sg] = acc;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) { float s = 0.0f; for (uint i = 0; i < 8; i++) s += part[i]; y[ulong(t) * N + row] = s; }
}
