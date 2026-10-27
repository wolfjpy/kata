# M4: MTP speculative decoding (depth 1, verification width 2)

**English** · [简体中文](M4-mtp-speculative-decoding.md)

Date: 2026-10-24 · Commits `c435af5` (MTP head) and `f7f3796` (speculative decoding).

## 1. Results

| item | plain decode | MTP speculative |
|---|---|---|
| greedy (temp 0), 200 tokens | 20.0 tok/s | **25.0 tok/s (+25%)** |
| sampled (temp 1, top-k 20, top-p 0.95, min-p 0.05) | 19.8 tok/s | **22.7 tok/s (+15%)** |
| predictable text (multi-turn, factual Q&A) | — | 29-31 tok/s (82-90% acceptance) |
| baseline (Prism llama.cpp `tg128`) | 15.6 tok/s | — |

A cycle (greedy, 60% acceptance, 1.60 tokens per cycle): 5.7 ms of drafting plus 58.2 ms of verification = 64 ms.

Quality: `verify2` against exact single-token decoding over 830 positions: **mean KL 4.3e-6, max KL 8.4e-4, top-1 830/830**.
The greedy output is character-identical to plain decoding for the first 690 characters, after which a near-tie flips
between "rolled tight" and "rolled so tightly" (both coherent; see §5).

## 2. The algorithm

Each cycle (state: `n` tokens already fed, `next` = the sampled but not yet fed token at position `n`, `hPrev` = the hidden
state at position `n-1`):
1. **Draft**: the MTP head processes 1 or 2 rows (the second is the row "owed" when the previous cycle accepted) and
   predicts `d`, the token at position `n+1`.
2. **Verify**: the main model processes `[next, d]` in one pass (positions `n` and `n+1`), reading every weight matrix once.
3. **Sample**: draw `s0` from row 0's logits with the normal sampler.
   - `s0 == d`: accept both, then sample `s1` from row 1 as the new `next`.
   - otherwise: commit only `next`, and `s0` becomes the new `next`.

This is **exact speculative sampling from a deterministic draft**: the sampled `s0` follows the target distribution; when it
happens to equal `d` it is accepted, with acceptance probability `p(d)`, and on rejection the resulting `s0` follows `p`
renormalised without `d`. The output distribution is identical to running without speculation, temperature, top-k/p and
min-p all work as usual, and no rejection-correction term is needed.

### State rollback: three buffers, zero copies
The GDN state (48 layers x 3.1 MB = 151 MB) has two possible endpoints during verification (after token 0 and after token
1). `k_gdn_rec_v2` reads the input state once and writes both results into two other buffers; accept/reject is just the
choice of the `stCur` pointer (`commit2`). The conv state works the same way.
The cost is one extra 151 MB write (~+1.1 ms per cycle). The attention K/V needs no rollback: on rejection position `n+1`'s
K/V is dirty, but the next cycle overwrites it before reading.

### Maintaining the MTP cache
- Prefill (>= 25 tokens): after the trunk, three batched GEMMs (`eh_proj`/`wk`/`wv`) write the MTP K/V for every position in
  the block at once (`Engine::encMtpKv`, in the same command buffer as the prefill).
- Stepwise feeding (<= 24 tokens): collect the hidden states and write the MTP K/V once at the end.
- An old problem was fixed: the `<|im_end|>` at the end of an assistant turn (and the last token left unfed when
  `--max-tokens` stops early) is now fed into the cache, so a multi-turn context matches the template.

## 3. The key engineering finding: at T>=2 the verification GEMV is ALU-bound (T=1 is still bandwidth-bound, see M3)

The first `verify2` cost 85 ms per cycle (expected ~52 ms), making MTP slower than plain decoding (17.9 vs 20.0 tok/s).
Ablation:

| skipped | `verify2` time |
|---|---|
| nothing | 85.3 ms |
| all GEMVs | 12.3 ms |
| GDN recurrence | 81.3 ms |
| alpha/beta | 78.9 ms |
| attention | 84.7 ms |

-> The GEMVs account for 73 ms (against 45 ms in the single-token version).

`bench_gemv2` microbenchmark on real tensors (T = tokens processed at once):

| kernel | ffn_gate 5120->17408 |
|---|---|
| fp32 g/c form, T=1 | 154 us (128 GB/s) |
| the same, T=2 | 245 us (80 GB/s) |
| the same, T=3 | 459 us (42 GB/s) |

T=1->2 costs +59% and T=3 almost +200%: **ALU-bound**. Counting operands, T=2 needs about 6 instructions per weight, i.e.
about 2.1 Tops/s, which is already at the GPU's nominal FP32 throughput.

`alu_probe` (`tools/alu_probe.cpp`) measures hardware instruction throughput:

| instruction | throughput |
|---|---|
| fp32 FMA | 1.22 T/s |
| fp16 scalar FMA | 1.66 T/s |
| **half2 FMA** | **2.35 T elements/s (~1.9x fp32)** |
| half4 FMA | 3.12 T elements/s |
| fp32 mul-add + floor | same speed as a plain FMA (**floor is essentially free**) |
| fp32 mul-add + fract | about 2x slower (fract is not free) |
| int32 mul-add | same speed as fp32 FMA |

On that basis I tried "look up the byte's 5 trits as halves + pack two tokens into half2". Ablation (`gemv_exp.metal`, bit
0 = skip the LUT read, bit 1 = skip the activation read, bit 2 = skip the weight read):

| variant | time |
|---|---|
| 2-byte LUT reads (16 per row per block per lane) | 267 us |
| skip the LUT read | **146 us** (= the bandwidth ceiling, 133 GB/s) |
| skip the activation read | 245 us |
| skip the weight read | 193 us |

-> The bottleneck is the random 2-byte reads from threadgroup memory. Changed to **16-byte entries with one vector read per
weight byte** (5 half trits packed into 3 32-bit words, 4 KB of threadgroup memory):

| kernel | three tensors |
|---|---|
| fp32 g/c form, T=2 | 245 / 251 / 147 us |
| **LUT + half2, T=2** | **170 / 174 / 101 us** (-33%, 113 GB/s) |

Partial sums are flushed to fp32 every 5 items (`float2(a0) + float2(a1) + ...`), which is more accurate than accumulating
a whole block in half, at the same speed.

Another finding: alpha/beta via `matmul2d` (M padded to 32, only 2 threadgroups) cost 6.4 ms; a dedicated 2-token bf16 GEMV
(`k_gdn_ab2`) brought it to nearly zero.

The final `verify2` is 57.8 ms: 51 ms of GEMV (113 GB/s) plus 6.8 ms of everything else (3.8 ms of GDN recurrence).

## 4. Numerics and quality

`kata verify-check` compares `verify2`'s logits against the exact single-token path at every position:

| path | positions | mean KL | max KL | max logit difference | top-1 |
|---|---|---|---|---|---|
| LUT + half2 (production) | 66 (72-token prompt) | 2.6e-7 | 1.2e-6 | 0.008 | 66/66 |
| LUT + half2 (production) | 830 | 4.3e-6 | 8.4e-4 | 1.02 | 830/830 |
| fp32 math + fp16 activations (`KATA_VERIFY_V1H=1`) | 830 | 1.6e-6 | 2.4e-4 | 0.44 | 830/830 |

Where the error comes from: fp16 activations (the same as the prefill path) contribute most of it; half accumulation adds
about 2.7x on top. Outliers are unrelated to context length and depend on the content.

All of this is inside the proposed quality gate (KL <= 1e-3, top-1 >= 99.7%), but **this is a lossy option and the gate
needed the owner's confirmation**.

## 5. Known limits

- The verification path differs from plain decoding by ~1e-6 of KL, so greedy output occasionally forks at a near-tie (the
  example above). That is not an error; requiring token-for-token equality would need a 2-token kernel with fp32
  activations (73 ms, slower than no speculation at all), which is not worth it.
- The draft depth is fixed at 1. Analysis: depth 2 needs one more 5.8 ms MTP pass plus a width-3 verification (half2
  cannot hold 3 tokens), which is expected to be a wash against depth 1, so it was not done.
- The sampler still uses `partial_sort` over the 248K vocabulary, about 1 ms per cycle (1.5%).
- The draft's `lm_head` (278 MB) is ~38% of the drafting cost; shrinking it to a frequent-vocabulary subset would save ~2 ms
  per cycle (3%), not done.

## 6. Next steps, by payoff

1. **Matrix-unit flash attention (prefill)**: prefill degrades badly with context (below), making long contexts unusable.
2. A prefill GEMM that fuses dequantisation (the expansion currently writes 24 GB of int8 weights, 16% of the time, and
   the GEMM reaches only 21.6 TOPS on the real shapes against a 36 TOPS peak).
3. Layer-major prefill: expand the weights only once per long prompt.
4. Decode: push the verification GEMV from 113 to ~125 GB/s (-5 ms per cycle), prune the draft vocabulary, move the GDN
   state to fp16.

Appendix: how prefill degrades with context (int8, `--fast`):

| prompt length | speed |
|---|---|
| 1024 | 278 tok/s |
| 3639 | 201 tok/s |
| 7279 | 176 tok/s |

## 7. Addendum: long-context decoding (flash-decoding)

7.3K-token context, greedy, 64 tokens:

| | plain decode | MTP |
|---|---|---|
| short context | 19.8 tok/s | 25.0 tok/s |
| 7.3K context, old scalar attention | 17.8 tok/s | 20.3 tok/s |
| 7.3K context, matrix-unit flash-decoding (default) | 17.6-17.9 tok/s | **21.4-21.7 tok/s** |
