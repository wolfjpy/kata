# M26: completing the verification side (arbitrary-width verification, state snapshots, commit)

**English** · [简体中文](M26-verification-side.md)

Date: 2026-10-25 · Corresponds to M26 in `R2-dflash2-revisit.md`. Prerequisite: M25 (the wide verification kernel, r=2.47).

## 1. What was built

- **`Engine::verifyWide(toks, T, pos, argmaxOut)`**: T <= 8 tokens go through the trunk in one pass. Every PQ1_0 matrix
  (including the shared `lm_head`, 278 MB read once, producing 8 rows of logits) is handled by `k_wide`; the GDN recurrence
  uses the new `k_gdn_rec_snap`, which stores one state snapshot per "accepted prefix length" (9 snapshots, 9 x 147 MB);
  attention uses the matrix-unit flash kernel; everything else reuses the prefill kernels with T rows.
- **`Engine::commitWide(accept)`**: pick the snapshot, copy it back into the live state; rebuild the conv state from the
  window's first `accept` rows of qkv with the new `k_gdn_conv_snap` kernel (no extra snapshot needed); top up the MTP
  cache for rows `pos+1 .. pos+accept-1` and update `hPrev`.
- **Test entry point `--vwidth T`**: verify T drafts per round (dummy drafts here) and accept only row 0, so the output must
  match plain greedy decoding token for token.

## 2. Correctness

| check | result |
|---|---|
| `--vwidth 8` (accept 1) against plain greedy | **token-identical** (550, 220, 16, 13, 561) |
| `--vwidth` against the `verify2` path | identical sequence |
| 9 snapshots + conv rebuild + MTP top-up across rounds | identical (the sequence above spans 5 rounds) |

## 3. Per-round cost (72-token context, greedy, average of 24 rounds)

| T | per round | relative to plain decode (50.1 ms/token) |
|---|---|---|
| 2 | 156 ms | 3.1x |
| 4 | 159 ms | 3.2x |
| 6 | 161 ms | 3.2x |
| 8 | **167 ms** | **3.3x** |

**The cost is almost independent of T**, matching the "weights are read once" expectation.

The T=8 breakdown (ablated with `--skip`):

| item | time |
|---|---|
| wide GEMV (every matrix, `lm_head` included) | 126 ms |
| attention (T=8 rows) | 10.3 ms |
| the alpha/beta GEMM (only 2 threadgroups per layer) | 6.0 ms |
| everything else (2 norms, 3 FWHTs, conv/qk/gates/gnorm and the argmax per layer) | ~25 ms |

Small-grid latency explains the last two: at a 72-token context the attention has only 4 threadgroups and the alpha/beta
GEMM only 2. Both are **fixable** (see §5).

## 4. What this means for DFlash2, using the measured acceptance length

DFlash2 at K=7 (measured in R2: 5.20 tokens per round on code/reasoning, 2.07 on prose), Q4_K_M drafter at ~9 ms:

| configuration | per round | code/reasoning | prose |
|---|---|---|---|
| now (167 + 9) | 176 ms | **29.5 tok/s** (today 25, +18%) | **11.8 tok/s** (-53%) |
| small-grid overhead fixed (~20 ms) | 156 ms | 33.3 (+33%) | 13.3 |
| if the wide GEMV reached r=1.6 | 106 + 9 ms | 49 (+96%) | 19.5 |

That is below R2's estimate (48-59 tok/s) because the wide kernel landed at r=2.47 rather than 1.2-1.6, and because the
fixed per-round overhead measured 41 ms against an estimated 16 ms. **So DFlash2 is worth +18% to +33% on code/reasoning
only; prose has to rely on M29's width adaptation, otherwise it loses 53%.**

## 5. Next steps, ordered by payoff over cost

1. **Fix the small-grid latency (~20 ms per round, 12%)**: turn alpha/beta into a T-row GEMV kernel (96xT threadgroups
   instead of 2); key-segment the attention (like the decode path's flash-decoding) so that 8 rows still have enough
   parallelism; fuse the two norms of each layer into one dispatch.
2. **Push the wide GEMV further**: M25 already swept tile shape, occupancy, double buffering and the mixed path; the only
   untried option is feeding the B operand from registers (`get_right_input_cooperative_tensor`), which needs the weights
   re-laid out by lane mapping and is expensive.
3. **The DFlash2 port (M28)**: a 5-layer drafter plus selector plus Q4_K_M dequantisation, following the contract in
   `refs/ninfer/docs/maintainer/dflash.md`; K=7 only, to start.
4. **M29, width adaptation**: prose falls back to MTP-1 (25 tok/s) automatically.

## 6. An honest assessment

- The verification infrastructure is **correct, measurable and reproducible**: arbitrary width, state snapshots, commit
  and MTP top-up all work, and all of it is validated against the hard criterion "it must reproduce plain greedy".
- But **of the 167 ms per round only 126 ms is the unavoidable weight read**; the other 41 ms is small-grid latency, and
  the 126 ms is itself three times the 40 ms bandwidth floor — the bottleneck is still M25's matrix-unit ingestion rate.
- So DFlash2 is worth "code/reasoning +18% to +33%" today, not the +100% R2 hoped for. Whether to continue into the port
  (M28, 3-5 days) depends on whether that 18-33% is worth it; if not, the advice was to finish §5.1/§5.2 first, since those
  two raise the ceiling to +33% to +96%.
