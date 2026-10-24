// Source: PrismML-Eng/llama.cpp @ 73bfe6cc... ggml/src/ggml-metal/kernels/misc.metal lines 340-540 (FWHT kernels incl. fused sign flip, threadgroup variant)
            atomic_fetch_add_explicit(dst, (int32_t) total, memory_order_relaxed);
        }
    }
}

typedef decltype(kernel_count_equal<int32_t>) kernel_count_equal_t;

template [[host_name("kernel_count_equal_i32")]] kernel kernel_count_equal_t kernel_count_equal<int32_t>;

template <typename T>
kernel void kernel_snake(
        constant ggml_metal_kargs_snake & args,
        device const T     * x,
        device const float * a,
        device const float * inv_b,
        device       T     * dst,
        uint         tgpig [[threadgroup_position_in_grid]],
        uint         tpitg [[thread_position_in_threadgroup]],
        uint         ntg   [[threads_per_threadgroup]]) {

    const int idx = tgpig * ntg + tpitg;
    if (idx >= args.T * args.C) {
        return;
    }

    const int   c  = idx / args.T;  // x is [T, C], a / inv_b collapse to [1, C]
    const float xi = float(x[idx]);
    const float si = sin(a[c] * xi);
    dst[idx] = T(xi + si * si * inv_b[c]);
}

template [[host_name("kernel_snake_f32")]]  kernel void kernel_snake<float>(constant ggml_metal_kargs_snake &, device const float *, device const float *, device const float *, device float *, uint, uint, uint);
template [[host_name("kernel_snake_f16")]]  kernel void kernel_snake<half>(constant ggml_metal_kargs_snake &, device const half *, device const float *, device const float *, device half *, uint, uint, uint);
#if defined(GGML_METAL_HAS_BF16)
template [[host_name("kernel_snake_bf16")]] kernel void kernel_snake<bfloat>(constant ggml_metal_kargs_snake &, device const bfloat *, device const float *, device const float *, device bfloat *, uint, uint, uint);
#endif

// SWIGLU: the transform input is the SwiGLU of src (gate) and up, computed on load with the
// same formula as kernel_swiglu, so the GLU output never round-trips through memory.
template<int N, typename src_t, bool SWIGLU = false>
kernel void kernel_fwht(
        constant ggml_metal_kargs_fwht & args,
        device const src_t * src,
        device float * dst,
        device const float * signs,
        device const src_t * up,
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort sgitg[[simdgroup_index_in_threadgroup]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort3  ntg[[threads_per_threadgroup]]) {

    constexpr int NW = N_SIMDWIDTH;
    constexpr int NE = N / NW;

    const float scale = 1.0f / sqrt((float) N);

    const int sg_per_tg = ntg.x / NW;
    const int64_t r = tgpig.x * sg_per_tg + sgitg;
    if (r >= args.nrows) {
        return;
    }

    // the Hadamard sign flip that precedes the transform in the graph is applied
    // on load when fused in: exact, since the factors are +-1
    signs += (args.n_blk > 0 ? (r % args.n_blk) * N : 0);

    src += r * N;
    dst += r * N;
    if (SWIGLU) {
        up += r * N;
    }

    const int lane = tiisg;

    float reg[NE];
    for (int i = 0; i < NE; i++) {
        const float s = args.n_blk > 0 ? signs[i*NW + lane] : 1.0f;
        float x = float(src[i*NW + lane]);
        if (SWIGLU) {
            x = (x / (1.0f + exp(-x)))*float(up[i*NW + lane]);
        }
        reg[i] = x*s*scale;
    }
    for (int i = 1; i < NW; i *= 2) {
        for (int j = 0; j < NE; j++) {
            const float val = reg[j];
            const float val2 = simd_shuffle_xor(val, i);
            reg[j] = (lane & i) == 0 ? val2 + val : val2 - val;
        }
    }

    for (int i = NW; i < N; i *= 2) {
        const int step = i / NW;
        for (int j = 0; j < NE; j += (2 * step)) {
            for (int k = 0; k < step; k++) {
                const float x = reg[j + k ];
                const float y = reg[j + k + step];
                reg[j + k]        = x + y;
                reg[j + k + step] = x - y;
            }
        }
    }

    for (int i = 0; i < NE; i++) {
        dst[i*NW + lane] = reg[i];
    }
}

// Wide blocks: one row per threadgroup instead of per simdgroup, so each thread
// keeps N/NT values rather than N/32. Butterflies below the simdgroup width still
// shuffle; those up to NT go through threadgroup memory; the rest stay in registers.
template<int N, int NT, typename src_t, bool SWIGLU = false>
kernel void kernel_fwht_tg(
        constant ggml_metal_kargs_fwht & args,
        device const src_t * src,
        device float * dst,
        device const float * signs,
        device const src_t * up,
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort sgitg[[simdgroup_index_in_threadgroup]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort3  ntg[[threads_per_threadgroup]]) {

    constexpr int NW = N_SIMDWIDTH;
    constexpr int NE = N / NT;

    threadgroup float shmem[N];

    const float scale = 1.0f / sqrt((float) N);

    const int64_t r = tgpig.x;
    if (r >= args.nrows) {
        return;
    }

    signs += (args.n_blk > 0 ? (r % args.n_blk) * N : 0);

    src += r * N;
    dst += r * N;
    if (SWIGLU) {
        up += r * N;
    }

    const int tid = sgitg * NW + tiisg;

    float reg[NE];
    for (int i = 0; i < NE; i++) {
        const float s = args.n_blk > 0 ? signs[i*NT + tid] : 1.0f;
        float x = float(src[i*NT + tid]);
        if (SWIGLU) {
            x = (x / (1.0f + exp(-x)))*float(up[i*NT + tid]);
        }
        reg[i] = x*s*scale;
    }

    for (int i = 1; i < NW; i *= 2) {
        for (int j = 0; j < NE; j++) {
            const float val  = reg[j];
            const float val2 = simd_shuffle_xor(val, i);
            reg[j] = (tid & i) == 0 ? val2 + val : val2 - val;
        }
    }

    for (int i = NW; i < NT; i *= 2) {
        for (int j = 0; j < NE; j++) {
            shmem[j*NT + tid] = reg[j];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (int j = 0; j < NE; j++) {
            const float val  = reg[j];
            const float val2 = shmem[j*NT + (tid ^ i)];
            reg[j] = (tid & i) == 0 ? val2 + val : val2 - val;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    for (int i = NT; i < N; i *= 2) {
        const int step = i / NT;
        for (int j = 0; j < NE; j += (2 * step)) {
            for (int k = 0; k < step; k++) {
                const float x = reg[j + k ];
                const float y = reg[j + k + step];
                reg[j + k]        = x + y;
                reg[j + k + step] = x - y;
            }
        }
    }

    for (int i = 0; i < NE; i++) {
        dst[i*NT + tid] = reg[i];
    }
}

typedef decltype(kernel_fwht<64, float>) kernel_fwht_f32_t;
typedef decltype(kernel_fwht<64, half>)  kernel_fwht_f16_t;

template [[host_name("kernel_fwht_f32_64")]]   kernel kernel_fwht_f32_t kernel_fwht<64, float>;
template [[host_name("kernel_fwht_f32_128")]]  kernel kernel_fwht_f32_t kernel_fwht<128, float>;
template [[host_name("kernel_fwht_f32_256")]]  kernel kernel_fwht_f32_t kernel_fwht<256, float>;
template [[host_name("kernel_fwht_f32_512")]]  kernel kernel_fwht_f32_t kernel_fwht_tg<512, GGML_METAL_FWHT_TG_NT, float>;
template [[host_name("kernel_fwht_f32_1024")]] kernel kernel_fwht_f32_t kernel_fwht_tg<1024, GGML_METAL_FWHT_TG_NT, float>;
