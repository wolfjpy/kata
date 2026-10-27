# M11: int8 KV cache (`--kv int8`, lossy, off by default)

**English** · [简体中文](M11-int8-kv.md)

Date: 2026-10-25 · Corresponds to CHECKPOINT-003 roadmap item M11. Basis: ninfer's default KV formats ("FP8 row-256 /
INT8 group-64").

## 1. Results

| item | fp16 KV (default) | int8 KV (`--kv int8`) |
|---|---|---|
| KV memory | 64 KB/token | **33 KB/token** (K and V, 16 layers x 1024 bytes each, plus one fp16 scale per head-row) |
| KV at 32K / 64K context | 2.1 / 4.2 GB | 1.1 / 2.2 GB |
| decode, 7.3K context (plain / MTP) | 17.5-18.0 / 21.4-21.5 tok/s | 18.1 / 21.3-21.5 tok/s (**no real gain**) |
| **decode, 29K context (plain)** | 13.7 tok/s | **15.6 tok/s (+14%)** |
| **prefill, 29K prompt** | 162 tok/s | **178 tok/s (+10%)** |
| quality: 24.5K-token corpus, 3072 positions, against fp16 KV | — | **mean KL 1.28e-4, top-1 99.71%** (3063/3072), max KL 0.18 |
| needle in a haystack (11.4K tokens, depth 10/50/90%) | 3/3 | 3/3 |
| `verify2` against exact decoding (72 tokens) | KL 2.5e-7 | KL 9.2e-7, top-1 66/66 |

Conclusion: quality just clears the gate (KL <= 1e-3, top-1 >= 99.7%); **memory halves, and there is a speed gain only
at long context (>= ~16K)** — no speed gain at short context. So it stays **off by default**; use `--kv int8` when you
need the long context or the memory. At 1536 positions top-1 is 99.67% and at 3072 positions 99.71%, which says the true
value sits around 99.7% and passes by a hair.

## 2. Design

- **Format**: K and V are each stored as one row of 256 values per (position, KV head), with
  `scale = fp16(amax / 127)` and values `round(x / scale)` clamped to ±127; K stores the post-RoPE values. The scales
  live in an fp16 plane indexed `[layer][position][4 heads]`.
- **Writing**: `k_attn_prep` / `k_attn_prep_b` gained a `cacheOff` parameter; in int8 mode the new rows go to a small fp16
  scratch (`[kMaxChunk][1024]`), and `k_kv_quant` then quantises them into the int8 cache and the scale plane. One extra
  small dispatch per layer when decoding, one per layer when prefilling.
- **Reading**: `k_flash_prefill` and `k_attn_dec_split` are templated on the K/V element type (`half` / `int8_t`);
  `matmul2d` consumes `half x int8_t` directly (a native type in the spec, at roughly fp16 throughput). Every column of
  `S = Q Kᵀ` (that is, every key) is multiplied by its K scale; every column of `P` is multiplied by the V scale before
  multiplying by V (the row sums `l` use the unscaled `p`).
- The single KV layer of the MTP head stays fp16 in all modes (~6% of the total KV).
- Prefix cache: checkpoints are read and written by "plane" (two planes for fp16, four for int8), the numerics tag carries
  `+kv8`, and the format version went to 2 (old checkpoints invalidate themselves).

## 3. Sensitivity: K or V? (quantise-dequantise simulation inside the fp16 cache, same 3072/1536 positions)

The `kv256` simulation (K+V, one scale per row) agrees with the real kernel (1.53e-4 / 99.67% vs 1.77e-4 / 99.67% on the
same 1536 positions), so the simulation can be trusted:

| quantisation scope | mean KL | top-1 (1536 positions) |
|---|---|---|
| K+V, per row | 1.5e-4 | 99.67% |
| K only, per row | 8.4e-4 (one outlier position with KL = 1.2 contributes 7.9e-4 of it) | 99.61% |
| V only, per row | **3.3e-5** | 99.80% |
| K+V, one scale per 64 dims | 1.2e-4 | 99.80% |

The point: **the error comes almost entirely from K** (a score before the softmax is amplified by the exponential); V is
essentially lossless. One scale per 64 dims would lift top-1 to 99.80%, but that difference is inside the measurement
noise (±2 positions), and a group scale for K would require splitting `Q Kᵀ` into four K=64 matmuls with separate scaling
— a large cost for nothing. If more quality margin is needed later, the cheaper option is **int8 for V only, fp16 for
K**: KL 3.3e-5 for a 25% reduction in KV bytes.

## 4. Known limits

- No speed gain below ~7K context (attention there is dominated by latency rather than bytes); only the memory benefit.
- The old scalar attention kernels (`KATA_SCALAR_*`) support fp16 KV only and are ignored in int8 mode.
- The evaluation corpus is a mix of papers, documentation and specifications; a longer (32K+) corpus of real
  conversation or code would narrow the confidence interval.

## Correction (M29e, re-measured properly)

The "KL 1.3e-4 / top-1 99.71%" figure above does not reproduce. The paired `kl-eval` mode is invalid for int8 KV
because `kvInt8` decides the buffer allocation at load time, so running both paths in one process only measures the
layout change. Cross-process measurement instead:

```
KATA_KV=fp16  kata kl-eval ... --save-ref ref.bin
KATA_KV=int8  kata kl-eval ... --ref ref.bin
-> 2048 positions | mean KL 3.1e+01 | max KL 4.7e+04 | max |logit diff| 7.9e+06 | top-1 2038/2048 (99.51%)
```

**The huge KL numbers are a metric artifact, not a broken kernel.** Running the per-position diagnostic
(`KATA_KL_DBG=1`) shows every position with KL > 1 also has enormous *reference* logits:

| position | KL | reference top logit | int8 top logit |
|---|---|---|---|
| 172 | 8.2e+03 | 1.342e+06 | 1.334e+06 |
| 939 | 8.3e+03 | 4.295e+05 | 4.193e+05 |
| 1711 | 4.7e+04 | 7.261e+05 | 7.117e+05 |
| 387 | 2.1e+00 | 226.2 | 225.6 |

A 0.5% relative error at a position whose logits are 1e6 is a ~5000 nat shift, so the two softmax distributions become
effectively disjoint and the KL explodes; the *mean* KL over the corpus is then dominated by a handful of such
positions (`(8243 + 8269 + 47090 + ...) / 2048 = 31`). At ordinary positions the error is small. The honest metric is
top-1, and it is **99.51%**, below the 99.7% gate.

**Root cause: quantisation granularity, not a kernel bug.** One scale covers all 256 elements of a (position, KV head)
vector, so a single outlier costs 1-2 effective bits and the whole head pays. That is the *design* constraint of this
scheme: the MMA path can only apply a scale as a per-key scalar (`cS *= kS[key]`, `P *= vS[key]`), so finer groups
cannot simply be added — a group scale would have to be folded into the dot product itself.

Two fixes are viable, in increasing effort:

1. **Asymmetric (scale + zero-point) quantisation**, which is exact to fold into the MMA path:
   `k = sc * q + mean`, so `S = sc * (Q K_q8) + mean * sum_d(q_d)` — the correction is one scalar per (query row, KV
   head) because `mean` is constant across the head's 256 dims, and `sum_d q_d` is computed once per row. For V the same
   decomposition gives `O += sum_k(p_k * mean_k)`, one scalar per (query row, KV head). This removes the outlier bias
   and typically recovers ~1 bit.
2. **Two scales per head** by splitting the 256 dims into two 128-wide MMAs (`S = sc_lo (Q_lo K_lo) + sc_hi (Q_hi K_hi)`),
   which is also exact and cheap (same FLOPs, two calls).

Until one of those lands, int8 KV stays opt-in and is not a default.

## Asymmetric quantisation: implemented, measured, rejected (M29g)

Fix 1 above (asymmetric `k = sc*q + mean`, folded back into the attention as `+ mean * sum_d(q_d)`) was implemented for
the key side in all three places (quantiser, prefill kernel, decode kernel; `kmean` buffer, per-row `qSum`, A/B switch
`KATA_KV_ASYM`). It measures **worse**, reproducibly:

| quantiser for K | top-1 (2048 positions) |
|---|---|
| symmetric `amax/127` (default) | **99.51%** |
| asymmetric `(min,max)` + mean correction | 99.37% |

The reason is arithmetic rather than a coding error: when the values of one head are roughly symmetric about zero,
`(MX - MN)/254 = 2*amax/254 = amax/127`, i.e. the asymmetric scale is the *same* as the symmetric one, so there is no
extra resolution to gain; the mean term then only adds one more fp16 rounding (`mean` is stored as fp16 and multiplied
by `sum_d(q_d)`, which is O(1) after the 1/sqrt(256) prescale). Fix 2 (two scales per head, 128-dim split MMAs) is
exact but needs a second 16x256 accumulator for V (128 registers/lane), which does not fit.

So the honest status of the int8 KV cache: **99.51% top-1, opt-in, not a default**; the coarse per-(position, head)
scale is a property of what the `matmul2d` path can fold, and passing the 99.7% gate needs either fp8 storage or
per-group scales with one MMA per group.
