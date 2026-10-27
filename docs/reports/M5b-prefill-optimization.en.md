# M5b: prefill optimisation (flash attention, GEMM order, expansion, int8 quality evaluation)

**English** · [简体中文](M5b-prefill-optimization.md)

Date: 2026-10-24 · Commits `00264f7` (attention), `96e048d` (expansion), `a3581ee` (int8), `53b04a8` (GEMM order).

## 1. Results

| scenario | before M5b | now | baseline (Prism llama.cpp) |
|---|---|---|---|
| exact fp16, 512 tokens | — | **245 tok/s** | 208 tok/s (`pp512`) |
| exact fp16, 1024-token chunk | 177 tok/s | **257 tok/s (+45%)** | |
| exact fp16, 7.3K-token prompt | 164 tok/s (had flash attention, old GEMM order) | **213 tok/s (+30%)** | |
| int8 (`--fast`, opt-in), 1024 chunk | 278 tok/s | 270-283 tok/s (no longer a clear advantage) | |
| int8, 7.3K prompt | 176 tok/s (scalar attention) | 232 tok/s | |

(The GPU throttles under sustained load and the same configuration can differ by 10-15% between runs; the table takes the
throttled steady range. On a cold machine the int8 1024-token chunk reaches 330 tok/s.)

The exact path's GEMM is now 3.0 s per 1024 tokens = **16.6 TFLOPS**, essentially at the fp16 matrix unit's peak of 17.3.

## 2. Changes and measurements

### 2.1 Matrix-unit flash attention (`shaders/flash.metal`)

The old attention was a scalar kernel: each query token scanned the whole K/V, O(n^2), unable to use the matrix unit. Of the
41 s taken by a 7.3K-token prompt, attention was ~17 s, and 32K context would have been a disaster.

The new kernel `k_flash_prefill`:
- one simdgroup = 16 query rows x 1 head; the 6 heads that share a GQA group form one threadgroup (192 threads) reading the
  same K/V;
- `S = Q Kᵀ` via `matmul2d_descriptor(16, 64, 256, transpose_right=true)`, with K read straight from the fp16 cache through
  a transposed view (tensor stride 1024);
- mask plus online softmax: `reduce_rows(max / sum)` stores each row's result to threadgroup memory;
- `O += P V`: P goes through threadgroup memory as a half left operand and `multiply_accumulate` accumulates into a
  cooperative tensor (16x256 fp32, 128 registers per thread);
- the tail key block uses the real tensor extent (`dextents(256, nk)`) and MPP zero-pads out of bounds: neither reading past
  the cache nor `NaN x 0 = NaN` (poison test: filling outside the valid region with NaN still gives finite, correct
  results);
- the whole chunk in one dispatch (the old implementation split at 256 tokens and then combined).

Microbenchmark (`tools/probe_flash.cpp`): max error 6.6e-5 against a CPU fp64 reference (values of order 0.28); 3.1-3.3
effective TFLOPS. Per layer, per 1024-token chunk: 4.0 ms at position 0, 62 ms at position 7168. Attention for an entire 8K
prefill is about 4 s (was ~17 s); at 32K about 67 s (still a sizeable cost on top of the 118 s linear part, see §4).
On the real model the exact prefill KL is unchanged (mean 1.39e-7 against the old kernel's 1.13e-7, top-1 72/72).

### 2.2 GEMM grid order (the biggest single effect on the exact path)

Surprise: after changing `dispatchThreadgroups` from a two-dimensional `(N/64, M/32)` to **a one-dimensional linear id with
our own mapping**, N-major (GM=0) became 12.8 s — the original 2D dispatch had been scheduled by the hardware in a
better-locality order, so the earlier "N-major is best" conclusion was really an artefact of that implicit order.

Sweeping the "grouping order" under the linear id (each group of GM adjacent M blocks walks all N blocks together, so one B
strip is reused by GM blocks while those GM blocks' A rows stay in cache):

| GM | exact path, 1024 tokens |
|---|---|
| 0 (N-major) | 12.8 s |
| 4 | 8.2 s |
| 8 | 5.6 s |
| 16 | 4.4 s |
| 24 | 4.9 s |
| **32 (all 1024 rows in one group, i.e. M-major)** | **3.98 s** |

With a 2048-token chunk, GM=32 gives only 190 tok/s (at 8K) and GM=64 only 157: the A matrix (M x K) no longer fits in
cache. Conclusion: **chunk 1024, GM 32**. The int8 path is insensitive to the order (it is limited by the ALU/epilogue, not
by B bandwidth).
Implementation: `tile_order(id, Mt, Nt, gm)`, overridable with `KATA_GM` (default 32).

### 2.3 The weight expansion kernel

The old kernel used about 30 instructions per weight (element-wise `ptq1_elem` decode, transpose through threadgroup
memory, byte-at-a-time stores). The new `k_expand_*_v2`: a thread = (4 rows x 8 consecutive k); an 8-element slice always
shares one trit position `n` (s < 10: `n = s/2`, byte `(s&1)*8..`; s = 10..14: `n = s-10`, byte 16..23; s = 15: `qh`), so the
decode is a uniform `floor(3^(n+1)u) - 3 floor(3^n u)` and the four rows' trits pack into one 32-bit word written directly
(no threadgroup memory, no barrier). **Bit-identical** to the old kernel (identical KL on both the fp16 and int8 paths).
int8 expansion 0.64 -> 0.41 s.

### 2.4 The int8 path: per-group activation scales + a vectorised epilogue

- Activation quantisation moved from "one scale per token" to "one scale per (token, 128-element group)", with the epilogue
  multiplying one extra row factor per group. Over 72 positions: mean KL 2.3e-4 -> 8.2e-5, max KL 3.2e-3 -> 3.1e-4.
- Cooperative-tensor element layout (`tools/layout_probe`): each lane's 32 elements are 4 rows (r, r+8, r+32, r+40) x 2
  column blocks (c0..c0+3, c0+32..c0+35). The epilogue therefore needs only two `float4` reads of weight scale and four of
  activation scale per group (the layout is checked at run time, falling back to the generic path if it does not match).
  10-13% faster than the generic path under the same conditions, bit-identical results.

## 3. int8 quality evaluation (new tool `kata kl-eval`)

On a 24.5K-token mixed corpus (papers, Chinese and English documentation, MSL specification excerpts) the exact and int8
paths prefill the corpus completely (each maintaining its own state), and the full-vocabulary distribution is compared at
the last 64 positions of each chunk (1536 in total):

| version | mean KL | max KL | top-1 agreement |
|---|---|---|---|
| per-token scale | 1.10e-3 | 1.04 | 99.35% |
| **per-group scale (now)** | **4.4e-4** | 0.34 | **99.54%** (1529/1536) |
| the proposed gate | <= 1e-3 | — | >= 99.7% |

## 4. Known limits and next steps

- **Cold start**: the first prefill of each process costs ~2.3 s extra (page faults: the 6.4 GB mapping touched by the GPU
  for the first time, plus the first dispatch). Touching every page from the CPU at startup took 1.7 s and only brought the
  first prefill to 2.1 s, a net loss; a Metal binary archive gave no measurable gain (pipeline creation is < 20 ms). Both
  experiments were reverted.
- **Expansion still costs 0.56 s per chunk (14%)**. Overlapping was tried (concurrent dispatch plus a double-buffered
  staging area, putting "expand the next matrix" and "the current GEMM" in the same barrier region, bit-identical results,
  no races). It was **slower**: 3.9 s became 5.6-5.8 s for 1024 tokens. A concurrent encoder with a full barrier (no
  overlap) is as fast as serial (3.9-4.0 s), so the slowdown comes from the overlap itself: each matrix's expansion writes
  178 MB, evicting the cache reuse the M-major GEMM depends on. Reverted. The remaining option is fusing dequantisation into
  the GEMM (re-decoding per M block, a large rewrite).
- **Long-context attention**: the flash kernel only reaches 3.1-3.3 TFLOPS (small M=16 tiles). Attention for a 32K prompt is
  about 67 s and is worth further tuning (32 rows per simdgroup, keeping P in registers as the left operand).
- Prefill for short prompts (<= 64 tokens) still pays the full expansion (~0.9 s), and is sometimes slower than per-token
  decoding (the stepwise path's threshold is 24 tokens).
- The exact path uses fp16 activations; `verify-check` shows its difference from fp32 decoding is of order 1e-6.
