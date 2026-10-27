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

## M31: long-context per-position KL, the external cross-check, and the K/V split

Driven by two points from a reader: (1) top-1 agreement can hide drift that only appears deep into a long context, so log
**per-position KL**; (2) if int8 on both K and V hurts, K is usually the touchier one, so run the cheap
**K fp16 / V int8** A/B. Both are right, and here is the data.

### 1. Tooling: per-position KL and depth buckets

`kl-eval` gained `KATA_KL_TRACE=1`, which prints `pos KL ref_logitmax cand_logitmax top1` per position plus a summary
**bucketed by depth** (0-1K / 1-4K / 4K-8K / 8K-16K / 16K-32K / >=32K), and `KATA_KV_DUMP=file`, which writes the first
64 KB of layer 0's K and V planes so two cache modes can be compared byte for byte.

**Control experiment** (the same reference against itself, fp16 KV, 16384 tokens): `mean KL 0.000e+00 | max 0.000e+00 |
4096/4096` — the harness is deterministic, so the depth trend below is signal rather than noise.

### 2. The depth curve: int8 KV against fp16 KV, 16384 tokens, 4096 positions

Quantiles after excluding degenerate positions (reference top logit >= 1e3, about 20% of them):

| depth | n | p50 | p90 | p99 | max | KL>1e-3 | top-1 miss |
|---|---|---|---|---|---|---|---|
| 0-4K | 1021 | 2.45e-05 | 2.32e-04 | 5.88e-03 | 1.01e-01 | 2.6% | 0.8% |
| 4-8K | 1020 | 2.16e-05 | 2.09e-04 | 9.44e-03 | 4.36e-02 | 3.3% | 0.6% |
| 8-12K | 1020 | 1.94e-05 | 2.11e-04 | 1.07e-02 | 4.16e+00 | 3.1% | 1.2% |
| 12-16K | 1020 | 7.36e-06 | 3.13e-04 | 2.21e-02 | 1.01e+00 | 3.9% | 0.8% |

**The median does not grow with depth** (regression of log10(KL) on position: **-0.022 dex per 1K tokens**, i.e. flat), but
**the tail gets heavier**: p99 goes from 5.9e-3 to 2.2e-2 (3.7x) and the fraction over the gate (KL > 1e-3) from 2.6% to
3.9%. So the reader is right that **top-1 (0.8-1.2%) hides the tail worsening**; but the drift is not everywhere — a few
positions get worse deep in the context.

The excluded ~20% degenerate positions (the reference's own logits at 1e3-1e10) behave as before: KL is meaningless there
(a 0.5% relative error on a 1e6 scale is thousands of nats).

### 3. External cross-check: llama.cpp's `--kl-divergence`

Same model, same text (512 tokens), llama.cpp's own fp16 KV as the base, then the KV switched to `-ctk q8_0 -ctv q8_0`:

```
llama-perplexity ... --save-all-logits /tmp/ll_base.bin          # fp16 KV baseline
llama-perplexity ... -ctk q8_0 -ctv q8_0 --kl-divergence --kl-divergence-base /tmp/ll_base.bin
  -> KLD: -0.000017 / -0.000031 / -0.000041 (per mille buckets), Mean dp: -0.000 +- 0.000 %
```

i.e. **|KL| ~ 4e-5**, against our int8 KV at **~1e-4** on the same order (median 2e-5, p90 2e-4). Same order of magnitude, so
**this repository's KL harness is corroborated by an independent implementation**; the 2-3x gap is exactly the granularity:
llama.cpp's q8_0 uses **32-element blocks** (8 scales per 256-dim head) while we use **one scale per 256-dim head**. That
matches §3's simulation (splitting the scale to 64 dims lowers the KL by a tier) and explains why M29g's asymmetric
quantisation could not save it: the problem is the **grouping**, not the zero point.

### 4. The K/V split (K fp16 / V int8): implemented, but **not usable yet**

- Two new kernels, `k_flash_prefill_k16v8` and `k_attn_dec_split_k16v8` (exact fp16 keys, int8 values with one scale per
  (position, KV head)), the engine split into `kvInt8K` / `kvInt8V` throughout (separate layer offsets, separate scale
  planes, a 3-plane prefix-cache layout) and a new mode `KATA_KV=int8v`.
- **What is verified**: V's int8 data plane is **byte-identical** between `int8` and `int8v` (100%), so V quantisation and
  addressing are correct; K is **not written at all** in mixed mode (the first 64 KB of the K plane is all zeros, against
  65421/65536 non-zero bytes in fp16 mode).
- Two real bugs were found and fixed: the prep kernel conflated "write into the cache or into the scratch" with "which side
  is quantised" (`cOff` had to be `kvInt8K`), and mixed mode needs **two different row offsets** (K is bound at the layer
  offset -> 0, V writes into the scratch -> pos0), so the prep kernel gained `cacheOffV`. After those fixes K is still not
  written and the cause is not located.
- Therefore **`KATA_KV=int8v` now refuses to run and says why** rather than returning wrong numbers. The fp16 and int8 paths
  are **numerically identical to before these changes** (2048 positions: fp16 KL 0, 2048/2048; `int8` KL 3.1e+01 / 99.51%,
  the same as M29f), and generation works.

**Next step (the entry point for whoever picks this up)**: in mixed mode, dump `kvScrK` and `kcache` for layer 0 and compare
the prep kernel's actual write addresses, or temporarily bind the prep's buffer 7 to the scratch in mixed mode and copy the
scratch into the cache, to bisect between "writes to the wrong place" and "does not write".
