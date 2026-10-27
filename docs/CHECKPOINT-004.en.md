# CHECKPOINT-004: progress after reopening DFlash2 (M25-M26) and what follows

**English** · [简体中文](CHECKPOINT-004.md)

Date: 2026-10-25 · A self-contained recovery document: **after a restart, reading this file (plus the two documents it
points at) is enough to continue.**
Predecessors: `CHECKPOINT-003.md` (results of M8-M17), `reports/R2-dflash2-revisit.md` (which overturned the M14 verdict
and set the reopening plan), `reports/M25-wide-verify-kernel.md`, `reports/M26-verification-side.md`.

## 0. Recovery guide

- Project: `<repo root>`; model `../../models/bonsai-2-27b-mtp-ptq1_0.gguf`; build `make` (~6 s); reference repositories
  `../../refs/{Strata,ninfer,llama.cpp-prism}`; drafter weights
  `../../models/drafters/{dflash2-q4km.gguf,bonsai2-2b-ptq1.gguf}`.
- The owner's four decisions for this round (R2 §7): (1) accept `r <= 1.6` as the condition to continue; (2) go straight
  to the **Q4_K_M** drafter, no Q8 intermediate; (3) **ignore the memory budget** — run as fast as the machine allows
  (16 GB UMA / 256 GB SSD); (4) accept `r ~ 2` as the lower-bound line.
- Working rules: reply in Chinese; documents and reports in Chinese; **commit messages in English only**; one commit per
  milestone with its report attached; only ever commit self-generated files (never the PDFs, PNGs or notes the owner keeps
  in the repository root); lossy switches must clear the quality gate and stay off by default; timing must be interleaved
  A/B (the machine throttles by ~10% after a minute of sustained load).

## 1. What this round completed

| milestone | status | key numbers |
|---|---|---|
| **M25** wide verification kernel | done, commit `b1f984b` | `ffn_gate`: 420 us (T=16) / 170 us (T=2) = **r 2.47** (target 1.6); numerical error 3e-6 |
| **M26** verification side | done, commit `fc95985` | `--vwidth 8` **reproduces plain greedy token for token**; 156 ms per round (T=2) to 167 ms (T=8), **almost independent of T** |

New code: `shaders/wide.metal` (parameterised `k_wide`, double-buffer/ablation switches, `k_wide_hyb`),
`k_gdn_rec_snap` + `k_gdn_conv_snap` in `shaders/prefill.metal`, `Engine::verifyWide/commitWide/dWide`,
`Engine::kMaxVerify=8`, the test entry point `--vwidth T`, and `tools/bench_wide.cpp`.

## 2. M25: why the wide kernel lands at 2.47 rather than 1.2

- Each simdgroup owns N=16 rows with no threadgroup barrier, decodes the weights through the 16-byte ternary LUT into a
  private fp16 tile (column-major, stride KC+8, scale already folded in, exact), then runs `matmul2d(M=16,N=16,K=128)`.
- **Tried and rejected**: K=256/512 (threadgroup memory doubles, occupancy drops to 50%, slower), double buffering
  (639 us, slower), mixing ALU with the matrix unit (473 us, no overlap gain), and a shape sweep (N=8/24/32/64 all worse).
- The hard limit: **at M=16 the matrix unit ingests at only 53% of peak** (MMA alone is 306 us = 291 G weights/s), while
  the production prefill GEMM (M=32/64, the whole K in one call) reaches 14.7-16.6 TFLOPS. The gap is "small M plus only
  K=128 per call".
- External comparison: ninfer reports T=8 at 1.55x a single step on a 5090 (about our r = 1.35); we are 1.8x slower.
- The one direction not tried: **feeding the B operand to `matmul2d` from registers**
  (`get_right_input_cooperative_tensor`), which would need the weight file re-laid out by lane mapping.

## 3. M26: what the verification side costs

- Per round (72-token context, average of 24 rounds): T=2 156 ms, T=4 159, T=6 161, T=8 **167 ms**. Flat cost, which
  confirms the "weights are read once" design.
- T=8 breakdown: wide GEMV 126 ms + attention 10.3 + alpha/beta 6.0 + ~25 for the rest (2 norms, 3 FWHTs, conv/qk/gates
  and the argmax per layer).
- The last two are **small-grid latency**: at 72 tokens the attention has only 4 threadgroups and the alpha/beta GEMM only
  2. About 20 ms should come back (-> 147 ms).

## 4. Corrected DFlash2 payoff, using measured numbers

Measured acceptance length for DFlash2 at K=7 (Prism harness, Kata as the target model): code/reasoning **5.20** tokens
per round, prose **2.07**.

| configuration | per round | code/reasoning | prose |
|---|---|---|---|
| today (K=7, 167 + 9 ms) | 176 ms | **29.5 tok/s** (+18%) | 11.8 (-53%) |
| small-grid overhead fixed | 156 ms | 33.3 (+33%) | 13.3 |
| wide kernel at r=1.6 | 115 ms | 49 (+96%) | 19.5 |

Conclusion: DFlash2 is worth **+18-33% on code/reasoning** today, and prose needs the width-adaptive fallback of M29.
That is short of R2's optimistic estimate (48-59 tok/s) because the wide kernel landed on the lower-bound line and the
fixed per-round overhead measured 41 ms rather than the estimated 16 ms.

## 5. Next steps, in order, each with an acceptance criterion

| milestone | content | acceptance |
|---|---|---|
| **M27a** done (`reports/M27a-small-grid-fixes.md`) | three fixes: alpha/beta as a T-row GEMV, key-segmented attention, state snapshots as record+replay (saves 1.2 GB) | **T=8 at 142 ms per round (was 177, -20%)**; reproduces greedy token for token; DFlash2 ceiling rises to **+38%** |
| **M27b** | MTP chain drafts (K=3 -> T=4, no new model): feed the MTP block's own output hidden back as the next row's input, three serial passes inside the block | reasoning text >= 3.1 tokens per round (Prism's measured value); end to end >= 28 tok/s with the `verifyWide` path unchanged |
| **M28a** done (`reports/M28a-draft-loading-and-quant.md`) | drafter loading + Q4_K/Q6_K decoding (**bit-identical** to ggml: 0.0 difference over 258,048 elements) + drafter matmuls (Q4_K 78-118 GB/s) | one draft round ~21 ms (15 ms once Q6_K is vectorised); DFlash2 payoff revised to **+28-32%** |
| **M28b** done (`reports/M28b-draft-module.md`) | Q6_K vectorised (17 -> 73.5 GB/s) + all drafter ops (conv/qk/attn/ctx_kv/matmuls) + loader and ring buffer (80 MB) | one draft round **14 ms**; the only gap left is "assemble the 5-layer forward + selector + spec integration" |
| **M28c** in progress (`reports/M28c-draft-forward.md`, **5** bugs fixed so far) | forward assembled and running end to end (tap collection / commit / materialise / 5 layers / shared head); two assembly bugs fixed (norm bound to the wrong buffer, Q6_K dtype) and the NaNs are gone; acceptance still 0/7 | drafter switched to Kata-specific files: `kata2-df2-ft5-Q2_K` (regular basis, Q2_K/Q3_K) and `Kata2-ternary` (2-bit prism + its own Hadamard signs, not started); `k_draft_mm` supports four quantisation types. **9 bugs fixed** (delta index, RoPE convention = MROMA sections summing to n_rot/2 -> full 128-dim rotation); **every op verified individually** (matmuls vs ggml, conv/attention/fc vs an independent CPU implementation, row 0 self-consistent); acceptance ~5%, with the problem narrowed to a **structural assumption** rather than an op |
| **M29** | width adaptation: an online EMA of tokens per ms for each width, prose falls back to MTP-1 automatically | prose >= 24 tok/s; no regression on code/reasoning |
| **M30** | (optional) register B operand in the wide kernel + weight re-layout | r <= 2.0 -> code +40% |

**Decision point (needs the owner)**: M28 is a 3-5 day port that buys +18-33% on code/reasoning (prose needs M29 as a
backstop first). The recommendation was to do M27a first (half a day, raises the ceiling to +33%) and then decide, or to
go straight to M28.

## 6. Traps hit in this round (written down so they are not hit twice)

1. **`snapStride` units**: the kernel-side pointer is `float*`, so the stride must be in **elements**; passing bytes writes
   4x out of bounds and shows up as SIGSEGV (with shared memory it corrupts the host heap).
2. **`k_gdn_ab_b` takes alpha and beta as two separate buffers** (stride 48), not a packed layout; the verify path must use
   the prefill `dGemm(wabAll -> pAB)`.
3. **`dWide`'s grid**: each threadgroup of `k_wide` is 4 simdgroups x 16 rows = **64 rows**, not 16.
4. **`prefill` does not reset the state**: call `resetState()` before running several experiments in one process, or the
   previous one leaks into the next (this is why the first `--vwidth` measurement was wrong).
5. **stdout buffering**: on a crash, buffered output is lost; use `setvbuf(_IONBF)` or flush every line when debugging GPU
   code (`main` already sets unbuffered).
6. Microbenchmarks are unreliable on dependency chains (the same shape differing by 3.8x, readings above the theoretical
   peak); trust the real kernel on real tensors.

## 7. Current performance baseline (do not forget it)

| metric | value |
|---|---|
| decode, plain | 19.8-20.0 tok/s (50.1 ms/token) |
| decode, MTP depth 1 | 25.0 (greedy) / 22.7 (sampled) tok/s |
| decode, 29K context | 13.7 (fp16 KV) / 15.6 (int8 KV) tok/s |
| prefill, exact | 255-263 tok/s (1024 chunk), 207-213 (7.3K) |
| prefill, `--fast-ffn` | 274-280, 226.5 tok/s |
| prefix cache | 2.9K system prompt: 15 s -> 0.03 s restore + 21 tokens |
| one wide verification round | T=2 156 ms / T=8 167 ms |

## 8. Environment and versions (checked 2026-10-25)

macOS 27.0.1, Xcode 27.1, Metal compiler 32023.921 (metal4.1 ceiling), metal-cpp `8cdde68`, Prism llama.cpp `b3ebc47`
(prism branch head `dc21b2c`; the newer commits are all CUDA/x86 and do not touch `ggml-metal`), Strata `4b9e05a`, ninfer
`64d10fb`. Details in `VERSIONS`.
