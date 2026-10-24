// Source: PrismML-Eng/llama.cpp @ 73bfe6cc... ggml/src/ggml-metal/kernels/mul_mv.metal lines 855-1170 (PTQ1_0 mat-vec incl. multicol variants)
        device       char * dst,
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {
    kernel_mul_mv_q2_0_f32_impl<N_R0_Q2_0, constant ggml_metal_kargs_mul_mv &>(args, src0, src1, dst, nullptr, tgpig, tiisg, sgitg);
}

// Byte-owning dot for PTQ1_0. The previous mapping gave each thread 16 contiguous
// weights, which in base-3 layout means every thread reloads the same qs bytes for a
// different trit index -- five times the byte traffic for the same 24 bytes. Here a
// thread instead owns whole bytes and consumes all five trits of each one, so a block's
// 26 bytes are loaded exactly once across the eight threads that cover it. The cost is
// that the y reads become strided rather than contiguous, which is the cheaper side:
// weight traffic is what decode is bound by.
//
// Byte ownership for thread it in 0..7, matching the CPU codec's element order:
//   qs[2*it], qs[2*it+1]  -> elements n*16 + m        for n in 0..4
//   qs[16 + it]           -> elements 80 + n*8 + it   for n in 0..4
//   qh[it] (it < 2 only)  -> elements 120 + n*2 + it  for n in 0..3
// Dot against coefficients already staged in registers by the caller and reused across
// all nr0 rows. A packed byte is a base-3 fraction of 256: with u = b/256, trit n is
//   t_n = g_{n+1} - 3*g_n,   g_k = floor(3^k * u)
// and 3^k*b <= 61965 is exact in fp32, so the floors are exact and this matches the
// integer recurrence bit for bit over all 256 bytes and all five positions. Summing
// t_n*y_n over a byte's trits then collapses to
//   sum_{k=1..4} g_k*(y_{k-1} - 3*y_k) + g_5*y_4
// whose coefficients depend only on the activations, so the caller stages those in
// place of the raw y. The inner loop is one floor and one fma per trit and never
// leaves the float pipe, which matters because this ISA cannot co-issue integer and
// floating-point work; the recurrence spent four integer ops and a convert per trit.
// sumy is subtracted once for the -1 offset, exactly as before.
inline float ptq1_0_dot_reg(device const block_ptq1_0 * qb, thread const float * yl, float sumy, short it) {
    float acc = 0.f;

    FOR_UNROLL (short k = 0; k < 2; ++k) {
        const float u = (float) qb->qs[2*it + k] * (1.0f/256.0f);
        thread const float * c = yl + 5*k;
        acc += floor(  3.0f*u)*c[0];
        acc += floor(  9.0f*u)*c[1];
        acc += floor( 27.0f*u)*c[2];
        acc += floor( 81.0f*u)*c[3];
        acc += floor(243.0f*u)*c[4];
    }

    {
        const float u = (float) qb->qs[16 + it] * (1.0f/256.0f);
        thread const float * c = yl + 10;
        acc += floor(  3.0f*u)*c[0];
        acc += floor(  9.0f*u)*c[1];
        acc += floor( 27.0f*u)*c[2];
        acc += floor( 81.0f*u)*c[3];
        acc += floor(243.0f*u)*c[4];
    }

    // qh holds 8 elements; every thread takes exactly one, trit it>>1 of byte qh[it&1].
    // The single-trit form is two floors and replaces a lane-divergent loop that
    // stepped the recurrence up to three times.
    {
        const float u  = (float) qb->qh[it & 1] * (1.0f/256.0f);
        const float p0 = yl[16];                      // 3^n for this thread's trit
        acc += (floor(3.0f*p0*u) - 3.0f*floor(p0*u)) * yl[15];
    }

    return (acc - sumy) * (float) qb->d;
}

template<int nr0, typename args_t>
void kernel_mul_mv_ptq1_0_f32_impl(
        args_t args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        threadgroup  char * shmem,
        uint3  tgpig,
        ushort tiisg,
        ushort sgitg) {
    const short NSG = FC_mul_mv_nsg;

    const int nb = args.ne00/QK_PTQ1_0;

    const int r0 = tgpig.x;
    const int r1 = tgpig.y;
    const int im = tgpig.z;

    const int first_row = (r0 * NSG + sgitg) * nr0;

    const uint i12 = im%FC_mul_mv_ne12;
    const uint i13 = im/FC_mul_mv_ne12;

    const uint64_t offset1 = r1*args.nb11 + (i12)*args.nb12 + (i13)*args.nb13;

    device const float * y = (device const float *) (src1 + offset1);

    device const block_ptq1_0 * ax[nr0];
    for (int row = 0; row < nr0; ++row) {
        const uint64_t offset0 = (first_row + row)*args.nb01 + (i12/FC_mul_mv_r2)*args.nb02 + (i13/FC_mul_mv_r3)*args.nb03;
        ax[row] = (device const block_ptq1_0 *) ((device char *) src0 + offset0);
    }

    // 15 collapse coefficients, the qh activation, and the qh trit's 3^n
    float yl[17];
    float sumf[nr0] = {0.f};

    // eight threads cover one 128-weight block; each owns whole bytes, not a
    // contiguous element span, so the block's bytes are read once in total
    const short ix = (tiisg/8);
    const short it = (tiisg%8);

    device const float * yb = y + ix*QK_PTQ1_0;

    {
        const float pow3f[4] = {1.0f, 3.0f, 9.0f, 27.0f};
        yl[16] = pow3f[it >> 1];
    }

    for (int ib = ix; ib < nb; ib += N_SIMDWIDTH/8) {
        // stage this thread's activations once as collapse coefficients, then reuse
        // them for every row: c[k-1] = y_{k-1} - 3*y_k for k = 1..4, c[4] = y_4
        float sumy = 0.f;

        FOR_UNROLL (short k = 0; k < 2; ++k) {
            const short m = 2*it + k;
            float y[5];
            FOR_UNROLL (short n = 0; n < 5; ++n) {
                y[n]  = yb[n*16 + m];
                sumy += y[n];
            }
            FOR_UNROLL (short n = 0; n < 4; ++n) {
                yl[5*k + n] = y[n] - 3.0f*y[n+1];
            }
            yl[5*k + 4] = y[4];
        }
        {
            float y[5];
            FOR_UNROLL (short n = 0; n < 5; ++n) {
                y[n]  = yb[80 + n*8 + it];
                sumy += y[n];
            }
            FOR_UNROLL (short n = 0; n < 4; ++n) {
                yl[10 + n] = y[n] - 3.0f*y[n+1];
            }
            yl[14] = y[4];
        }
        {
            const float v = yb[120 + it];
            yl[15] = v;
            sumy  += v;
        }

        FOR_UNROLL (short row = 0; row < nr0; row++) {
            sumf[row] += ptq1_0_dot_reg(ax[row] + ib, yl, sumy, it);
        }

        yb += QK_PTQ1_0 * (N_SIMDWIDTH/8);
    }

    device float * dst_f32 = (device float *) dst + (uint64_t)im*args.ne0*args.ne1 + (uint64_t)r1*args.ne0;

    for (int row = 0; row < nr0; ++row) {
        const float tot = simd_sum(sumf[row]);

        if (tiisg == 0 && first_row + row < args.ne01) {
            dst_f32[first_row + row] = tot;
        }
    }
}

// Keep the single-vector arithmetic order while sharing decoded weights across columns.
template<int nr1>
inline void ptq1_0_dot_multicol(device const block_ptq1_0 * qb,
        thread const float (&yl)[nr1][17], thread const float (&sumy)[nr1], short it,
        thread float (&sumf)[nr1]) {
    float acc[nr1] = {};
    FOR_UNROLL (short byte = 0; byte < 3; ++byte) {
        const float u = (float) qb->qs[byte < 2 ? 2*it + byte : 16 + it] * (1.0f/256.0f);
        const float g[5] = {floor(3.0f*u), floor(9.0f*u), floor(27.0f*u), floor(81.0f*u), floor(243.0f*u)};
        FOR_UNROLL (short col = 0; col < nr1; ++col) {
            FOR_UNROLL (short n = 0; n < 5; ++n) {
                acc[col] += g[n] * yl[col][5*byte + n];
            }
        }
    }
    const float u = (float) qb->qh[it & 1] * (1.0f/256.0f);
    const float p0 = yl[0][16];
    const float trit = floor(3.0f*p0*u) - 3.0f*floor(p0*u);
    const float d = (float) qb->d;
    FOR_UNROLL (short col = 0; col < nr1; ++col) {
        acc[col] += trit * yl[col][15];
        sumf[col] += (acc[col] - sumy[col]) * d;
    }
}

template<int nr0, int nr1>
kernel void kernel_mul_mv_ptq1_0_multicol(
        constant ggml_metal_kargs_mul_mv & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {
    const short NSG = FC_mul_mv_nsg;

    const int nb = args.ne00/QK_PTQ1_0;

    const int r0 = tgpig.x;
    const int r1 = tgpig.y * nr1;
    const int im = tgpig.z;

    const int first_row = (r0 * NSG + sgitg) * nr0;

    const uint i12 = im%FC_mul_mv_ne12;
    const uint i13 = im/FC_mul_mv_ne12;

    const uint64_t offset1 = r1*args.nb11 + (i12)*args.nb12 + (i13)*args.nb13;

    device const float * y = (device const float *) (src1 + offset1);

    device const block_ptq1_0 * ax[nr0];
    for (int row = 0; row < nr0; ++row) {
        const uint64_t offset0 = min(first_row + row, args.ne01 - 1)*args.nb01 + (i12/FC_mul_mv_r2)*args.nb02 + (i13/FC_mul_mv_r3)*args.nb03;
        ax[row] = (device const block_ptq1_0 *) ((device char *) src0 + offset0);
    }

    // 15 collapse coefficients, the qh activation, and the qh trit's 3^n
    float yl[nr1][17];
    float sumf[nr0][nr1] = {};

    // Eight threads cover one block, with each thread reading whole bytes.
    const short ix = (tiisg/8);
    const short it = (tiisg%8);

    device const float * yb = y + ix*QK_PTQ1_0;

    {
        const float pow3f[4] = {1.0f, 3.0f, 9.0f, 27.0f};
        FOR_UNROLL (short col = 0; col < nr1; ++col) {
            yl[col][16] = pow3f[it >> 1];
        }
    }

    for (int ib = ix; ib < nb; ib += N_SIMDWIDTH/8) {
        // Reuse collapse coefficients across rows: c[k-1] = y_{k-1} - 3*y_k, c[4] = y_4.
        float sumy[nr1] = {};
        FOR_UNROLL (short col = 0; col < nr1; ++col) {
            device const float * yc = (device const float *) ((device const char *) yb + col*args.nb11);

            FOR_UNROLL (short k = 0; k < 2; ++k) {
                const short m = 2*it + k;
                float y[5];
                FOR_UNROLL (short n = 0; n < 5; ++n) {
                    y[n]  = yc[n*16 + m];
                    sumy[col] += y[n];
                }
                FOR_UNROLL (short n = 0; n < 4; ++n) {
                    yl[col][5*k + n] = y[n] - 3.0f*y[n+1];
                }
                yl[col][5*k + 4] = y[4];
            }
            {
                float y[5];
                FOR_UNROLL (short n = 0; n < 5; ++n) {
                    y[n]  = yc[80 + n*8 + it];
                    sumy[col] += y[n];
                }
                FOR_UNROLL (short n = 0; n < 4; ++n) {
                    yl[col][10 + n] = y[n] - 3.0f*y[n+1];
                }
                yl[col][14] = y[4];
            }
            {
                const float v = yc[120 + it];
                yl[col][15] = v;
                sumy[col] += v;
            }
        }

        FOR_UNROLL (short row = 0; row < nr0; row++) {
            ptq1_0_dot_multicol<nr1>(ax[row] + ib, yl, sumy, it, sumf[row]);
        }

        yb += QK_PTQ1_0 * (N_SIMDWIDTH/8);
    }

    device float * dst_f32 = (device float *) dst + (uint64_t)im*args.ne0*args.ne1 + (uint64_t)r1*args.ne0;

    for (int row = 0; row < nr0; ++row) {
        FOR_UNROLL (short col = 0; col < nr1; ++col) {
            const float tot = simd_sum(sumf[row][col]);
            if (tiisg == 0 && first_row + row < args.ne01) {
                dst_f32[(uint64_t) col*args.ne0 + first_row + row] = tot;
            }
        }
    }
}

typedef decltype(kernel_mul_mv_ptq1_0_multicol<4, 2>) mul_mv_ptq1_multicol_t;
template [[host_name("kernel_mul_mv_ptq1_0_f32_mc_c2")]] kernel mul_mv_ptq1_multicol_t kernel_mul_mv_ptq1_0_multicol<4, 2>;
template [[host_name("kernel_mul_mv_ptq1_0_f32_mc_c3")]] kernel mul_mv_ptq1_multicol_t kernel_mul_mv_ptq1_0_multicol<4, 3>;
template [[host_name("kernel_mul_mv_ptq1_0_f32_mc_c4")]] kernel mul_mv_ptq1_multicol_t kernel_mul_mv_ptq1_0_multicol<4, 4>;

[[host_name("kernel_mul_mv_ptq1_0_f32")]]
kernel void kernel_mul_mv_ptq1_0_f32(
        constant ggml_metal_kargs_mul_mv & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {
    kernel_mul_mv_ptq1_0_f32_impl<N_R0_PTQ1_0, constant ggml_metal_kargs_mul_mv &>(args, src0, src1, dst, nullptr, tgpig, tiisg, sgitg);
}

// PQ2_0 and Q2_0 dot against coefficients staged by the caller (same codec, group 128 vs 64). A byte is a base-4 fraction of
// 256: with u = b/256 and g_k = floor(4^k*u), exact in fp32, the floor chain peels the
// fields from the top down, g_1 = t_3, g_2 - 4*g_1 = t_2, g_3 - 4*g_2 = t_1 and
