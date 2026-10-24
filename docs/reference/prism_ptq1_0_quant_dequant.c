// Source: PrismML-Eng/llama.cpp @ 73bfe6cc485554822f80c473a3c0c4f25e80f41b (branch prism), ggml/src/ggml-quants.c lines 2203-2300 (MIT license, ggml authors/Prism)
static const size_t ptq1_0_stages[3] = {32, 16, 8};

void quantize_row_ptq1_0_ref(const float * GGML_RESTRICT x, block_ptq1_0 * GGML_RESTRICT y, int64_t k) {
    assert(k % QK_PTQ1_0 == 0);
    const int64_t nb = k / QK_PTQ1_0;

    for (int64_t i = 0; i < nb; i++) {
        float amax = 0.0f;
        for (int j = 0; j < QK_PTQ1_0; j++) {
            amax = MAX(amax, fabsf(x[j]));
        }

        const float d  = amax;
        const float id = d ? 1.0f/d : 0.0f;

        y[i].d = GGML_FP32_TO_FP16(d);

        size_t j = 0;
        for (size_t s = 0; s < 3; ++s) {
            const size_t c = ptq1_0_stages[s];
            for (; j + c <= sizeof(y->qs); j += c) {
                for (size_t m = 0; m < c; ++m) {
                    uint8_t q = 0;
                    for (size_t n = 0; n < 5; ++n) {
                        int xi = lroundf(x[m + n*c] * id) + 1; // -1, 0, 1 -> 0, 1, 2
                        q *= 3;
                        q += xi;
                    }
                    // ceiling division (243 == pow(3, 5))
                    q = ((uint16_t)q * 256 + (243 - 1)) / 243;
                    y[i].qs[j + m] = q;
                }
                x += 5*c;
            }
        }
        // 4 elements per byte
        for (size_t h = 0; h < sizeof(y->qh); ++h) {
            uint8_t q = 0;
            for (size_t m = 0; m < 4; ++m) {
                int xi = lroundf(x[h + m*sizeof(y->qh)] * id) + 1;
                q *= 3;
                q += xi;
            }
            // shift the first value to the most significant trit
            q *= 3;
            q = ((uint16_t)q * 256 + (243 - 1)) / 243;
            y[i].qh[h] = q;
        }
        x += 4*sizeof(y->qh);
    }
}

void dequantize_row_ptq1_0(const block_ptq1_0 * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k) {
    assert(k % QK_PTQ1_0 == 0);
    const int64_t nb = k / QK_PTQ1_0;

    const uint8_t pow3[6] = {1, 3, 9, 27, 81, 243};

    for (int64_t i = 0; i < nb; ++i) {
        const float d = GGML_FP16_TO_FP32(x[i].d);

        size_t j = 0;
        for (size_t s = 0; s < 3; ++s) {
            const size_t c = ptq1_0_stages[s];
            for (; j + c <= sizeof(x->qs); j += c) {
                for (size_t n = 0; n < 5; ++n) {
                    for (size_t m = 0; m < c; ++m) {
                        uint8_t q = x[i].qs[j + m] * pow3[n];
                        int16_t xi = ((uint16_t) q * 3) >> 8;
                        *y++ = (float) (xi - 1) * d;
                    }
                }
            }
        }
        for (size_t n = 0; n < 4; ++n) {
            for (size_t h = 0; h < sizeof(x->qh); ++h) {
                uint8_t q = x[i].qh[h] * pow3[n];
                int16_t xi = ((uint16_t) q * 3) >> 8;
                *y++ = (float) (xi - 1) * d;
            }
        }
    }
}

size_t quantize_ptq1_0(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrow, int64_t n_per_row, const float * quant_weights) {
    (void)quant_weights; // ternary codes come from the weights themselves; an imatrix has no role
    const size_t row_size = ggml_row_size(GGML_TYPE_PTQ1_0, n_per_row);
    char * qrow = (char *)dst;
    for (int64_t row = 0; row < nrow; ++row) {
        quantize_row_ptq1_0_ref(src, (block_ptq1_0 *)qrow, n_per_row);
        src  += n_per_row;
        qrow += row_size;
    }
    return nrow * row_size;
}

size_t quantize_q4_0(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrow, int64_t n_per_row, const float * quant_weights) {
    if (!quant_weights) {
