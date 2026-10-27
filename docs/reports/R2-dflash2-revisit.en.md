# R2: DFlash2 revisited — the earlier conclusion was wrong, so it is reopened

**English** · [简体中文](R2-dflash2-revisit.md)

Date: 2026-10-25 · Supersedes the verdict of `M14-wide-verify-gate` (whose *measurements* remain valid) and rewrites the
DFlash2 line of the "not recommended" section in `REVIEW-01` §4.

## 1. Where I went wrong (three errors, all of method)

**(1) I used T=2 as the reference instead of the bandwidth floor.**
A T=8 verification must read the 5.657 GB of weights at least once, so at 142 GB/s **39.8 ms is an irreducible lower
bound**. The compute demand of T=8 is 8 x 48.7 GFLOP = 390 GFLOP, which at this machine's 14.7 TFLOPS is 26.5 ms — *still
below the memory floor*. Comparing against T=2 (170 us) rather than against that floor made the gate far stricter than the
physics requires.

**(2) I framed it as a pass/fail gate rather than a "how cheap could it be" question.**
The right question is what the cost floor of one round is and how fast it must be to cover the drafting cost. Framed that
way, the bar is much lower (§4).

**(3) The kernel I based the conclusion on had four fixable flaws** (§3), two of which — one `matmul2d` call per K=128
chunk, and M=8 — account for a 3x gap against the production prefill GEMM (14.7-16.6 TFLOPS, the whole K in one call). I
condemned the direction with an afternoon's prototype instead of fixing the prototype.

## 2. The arithmetic redone

| quantity | value |
|---|---|
| weights read per verification round | 5.657 GB |
| bandwidth floor | 39.8 ms (142 GB/s) |
| compute demand of T=8 | 390 GFLOP -> 26.5 ms at 14.7 TFLOPS (still bandwidth-bound) |
| compute demand of T=4 | 195 GFLOP -> 13.3 ms (far below the floor) |
| state: 8 GDN snapshots (instead of triple buffering) | 8 x 147 MB of writes ~ 8.4 ms; 1.2 GB of memory (fine on a 16 GB machine, and cheaper than replaying 5 steps x 2.6 ms) |
| T=8 attention (7.3K context) | about 8-13 ms (8 query rows) |
| one DFlash2 draft forward | ~16 ms as Q8_0 / ~9 ms as Q4_K_M (1.42-2.0 GB of weights) |

So the **theoretical** cost of one T=8 round is `40 + 8.4 + 12 + 3 + draft 9~16 ≈ 72-80 ms`.

## 3. Why my prototype kernel only reached 47 GB/s

`k_wide_sg` (M14): 415 us on ffn_gate, and the ablation shows 134 us of decode plus 281 us of MMA, serialised. Four
mistakes:

| mistake | evidence | fix |
|---|---|---|
| **each `matmul2d` covers only K=128** | the production prefill GEMM covers the whole K=5120 in one call and reaches 14.7-16.6 TFLOPS on the same hardware; each call has ~30 ns of fixed overhead, so at K=128 the compute itself is only ~5 ns — 90% is overhead | cover K=256-512 per call (limited by the 32 KB threadgroup budget), amortising the fixed cost 2-4x |
| **M=8 used an 8-row tile** | in the shape sweep M=16 ingests about 1.2x more weight per unit time than M=8, and M=8 only has 8 real tokens | pad T=8 up to **M=16**: twice the FLOPs but still bandwidth-bound, so it is actually faster; pad T=4 to M=8/16 too |
| **N=R=16 (only 16 rows per simdgroup)** | N=32 is clearly better, N=64 gets worse (it does not fit in threadgroup memory, forcing a smaller K) | N=32 + K=256 (20 KB) or N=16 + K=512 (17 KB); one of the two, measured |
| **decode and MMA serialised** | 134 us + 281 us = 415 us (they add up exactly) | double-buffer the threadgroup tile: decode block k+1 while the matrix unit consumes block k, taking max instead of sum |

Two traps already known are carried over: the threadgroup row stride must be padded (`ld = 128 -> 136` is 1.6x, bank
conflicts), and the tail tile needs a real extent to avoid `NaN x 0`.

Honest note: the shape microbenchmarks I wrote in `/tmp` that day were **unstable** (the same shape measuring 2.19 and 8.40
TFLOPS on two runs; anything above 17.3 TFLOPS means the compiler reordered the dependency chain), so I do not use them as
data — only for qualitative direction (padding, M, K). Real numbers must come from **the real kernel on real tensors**.

## 4. The curve measured that day: acceptance length versus draft width

Using Prism's `llama-speculative-simple` (its Metal speculation path is slow, but the acceptance statistics are valid),
with Bonsai-2-27B as the target, the DFlash2 Q4_K_M drafter and greedy decoding:

| draft width K | verification width T=K+1 | code: tokens/round | prose: tokens/round |
|---|---|---|---|
| 1 | 2 | 1.92 | — |
| 3 | 4 | **3.24** | 1.97 |
| 5 | 6 | **4.68** | — |
| 7 | 8 | **5.20** | **2.07** |

(K=1 is "verification width 2", exactly our MTP depth 1 shape; DFlash2 is slightly better than MTP at the same width.)

Which says: **the payoff grows monotonically with width, while at the bandwidth bound the verification cost is almost
independent of width** -> once verification is cheap, wider is better. K=7 yields 60% more tokens than K=3 for one extra
draft's worth of work (about +3 ms).

## 5. The corrected payoff table (code/reasoning text, draft width 7)

`verify ≈ 51*r + 15` (51 ms is today's T=2 GEMV part, `r` the T=8/T=2 ratio, +15 for state/attention/misc), with a 9 ms
Q4_K_M draft:

| verification kernel state | r | per round | code/reasoning tok/s | prose tok/s | against today's 25 tok/s |
|---|---|---|---|---|---|
| today's T=2 baseline | 1.0 | 75 ms | **69** | 28 | 2.8x / 1.1x |
| four flaws fixed (the target) | 1.2-1.6 | 86-108 ms | **48-59** | 19-24 | 1.9-2.4x / 0.8-1.0x |
| the M14 prototype (unfixed) | 2.3 | 145 ms | 36 | 14 | 1.4x / 0.6x |

Two readings:
- **On code/reasoning even the unfixed prototype is a net win** (36 vs 25). My earlier "not worth it" was wrong.
- **Prose is the only thing to worry about**: once r > 1.6, prose gets slower. The answer is not to abandon DFlash2 but
  **width adaptation** (the DraftPolicy idea from Strata: an online EMA of tokens per millisecond for each width, so prose
  falls back to width-2 MTP or plain decoding automatically). That also explains why Strata's spec depth stops at 2 on
  prose.

## 6. The corrected plan (replacing the M14/M15/M16 closure)

**M25: the wide verification kernel (2-3 days; all of the technical risk lives here)**
1. The real kernel `k_wide_v2`: pad to M=16, N=32, K=256 per call (20 KB of threadgroup memory), double buffered, LUT
   decode, weight tiles exact fp16 (or int8 ternary plus a per-128-group row scale).
2. Acceptance: **r <= 1.6 on ffn_gate/ffn_down/gdn_qkv** (i.e. 170 -> <= 272 us equivalent, target 200 us); numerical error
   <= 1e-5 against an fp32 reference.
3. Along the way: 8 GDN/conv snapshots (replacing triple buffering), T-row KV writes (already there), T-row MTP K/V.

**M26: complete the verification side (1 day, no draft model)**
- Accept arbitrary widths on the target side: `verify(T, toks)` produces T rows of logits/hidden, accepts the greedy
  prefix and commits the matching snapshot; use **dummy drafts** (repeated/random tokens) to measure the real "cost per
  round".
- Acceptance: the per-round cost at T=4/6/8 is <= the table above (<= 108 ms at r=1.6); KL against per-token decoding
  <= 1e-5.

**M27: the MTP chain (half a day, no new model)**
- Use the existing MTP block to draft a chain (feeding the previous row's output hidden as the next row's input), K=3 ->
  T=4 verification. Prism measures 3.13 tokens per round on reasoning.
- Expectation: reasoning text 25 -> 38-45 tok/s. This is **an intermediate gain available without DFlash2**, and it
  validates the correctness of M25/M26.

**M28: the DFlash2 port (3-5 days)**
- Drafter side: 5 layers (plain RMSNorm, QKV, per-head qk norm, scalar RoPE, a non-causal 2048 sliding-window ring KV, the
  two-tap dynamic grouped convolution, SwiGLU), conditioned on the residuals of target layers 5/19/33/47/61, a context
  materialiser, and a top-16 + rank-256 selector; Q8_0 first (we already have Q8_0 GEMV/expansion), Q4_K_M as a later
  optimisation.
- Acceptance: acceptance length within 0.3 of llama.cpp's `draft-dflash`; end to end code/reasoning >= 45 tok/s, prose >= 22.
- Risk: the selector and the dynamic convolution have many details (ninfer's `docs/maintainer/dflash.md` gives the complete
  contract and `include/ninfer/ops/*.h` the exact numerical conventions); Q4_K_M dequantisation needs a new kernel.

**M29: width adaptation (half a day)**
- Estimate tokens/ms online for each width and let prose fall back to a narrower one, avoiding a regression when r is not
  ideal.

## 7. Data and decisions needed

1. Whether **M25's kernel target** accepts "r <= 1.6" as the condition to continue (which maps to 48-59 tok/s on code, with
   prose covered by M29).
2. **Q4_K_M or Q8_0 drafter**: we already have kernels for Q8_0, saving 1-2 days, but the draft costs ~7 ms more per round
   (16 ms as Q8_0 vs 9 ms as Q4_K_M). The recommendation was to get Q8_0 working end to end first and then decide whether
   Q4_K adds enough.
3. Memory: 8 state snapshots add 1.2 GB, and the DFlash2 drafter weights are 1.42-2.0 GB. With 64K context and int8 KV the
   total is about 9-10 GB, still inside the 12.1 GB working set; whether a maximum width should be capped to save memory.
4. If M25 cannot reach r <= 1.6: the r ~ 2 version plus M29 adaptation still works (+44% on code, no prose regression), just
   with half the payoff. Whether that lower-bound line is acceptable.

## 8. The two lessons, restated for next time

- **Before judging whether a direction is worth it, compute its physical lower bound and compare your implementation
  against that**: when the lower bound and the implementation differ by 3x, the conclusion should be "my implementation is
  3x off", not "this direction does not work".
- **Microbenchmarks are unreliable on dependency chains**: the same shape differing by 3.8x between runs, or readings
  above the theoretical peak, mean the measurement itself is broken; only the real kernel on real data with interleaved
  comparisons can settle anything.
