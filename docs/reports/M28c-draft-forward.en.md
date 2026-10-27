# M28c: the draft forward is assembled and runs end to end; acceptance is still 0 (three bugs located, two fixed)

**English** · [简体中文](M28c-draft-forward.md)

Date: 2026-10-25 · Step three of M28. **This is an honest record of an intermediate state**: the whole chain is wired up
and the NaNs are gone, but the drafter's proposals still disagree with the trunk (0 of 7 accepted). The next round
continues from exactly here.

## 1. What runs now

```
build/kata dflash-eval --model ../../models/bonsai-2-27b-mtp-ptq1_0.gguf \
  --draft ../../models/drafters/dflash2-q4km.gguf --ids "$(cat /tmp/ids72.txt)" --gen 24
```

Output skeleton (measured):

```
[draft] 5 layers, 1079 MB of weights, ring 4096 positions, 80 MB of K/V
prompt 72 tokens via 8-row windows in 4.1 s; anchor 271 at pos 72
  round 0: drafts 5144 6681 91561 91561 10758 10758 10758
            trunk   550 49276 271 271 271 271 271
            accept 0 -> 1 tokens
DFlash2: 24 rounds, 24 tokens, 1.00 tokens/round, 142.6 ms/round
```

**What already works correctly**:
- both the prompt and the continuation go through 8-row wide verification windows, and the trunk's continuation is
  **token-for-token identical** to plain greedy decoding (550, 220, 16, 13, 561, 1330, 33286, 271);
- residual tap collection (layers 6/20/34/48/62, copied to fp16 inside `verifyWide` right after the residual add),
  `commitWide` and `Draft::materialise` (context K/V into the ring buffer) all execute as intended and stay stable across
  rounds;
- the drafter's 5-layer forward is assembled exactly per Prism's `dflash.cpp` (two-sided conv, delta reuse, per-head qk
  norm, partial RoPE, non-causal sliding window, shared lm_head) and costs about **4.5 ms** per round (3% of the ~146 ms
  round).

## 2. The two bugs fixed this round (both in the assembly, committed)

1. **The drafter's norm weights were bound to the trunk's buffer**: `k_rmsnorm_rows*`'s gamma must come from the
   drafter's own GGUF mapping, not the trunk's `wbuf`. After fixing it the logits went from "all zeros" to real values.
2. **Q6_K tensors were decoded as Q4_K**: Q4_K_M mixes in 4 Q6_K tensors (`attn_v` and `ffn_down` of `blk.2/4`), and my
   `mm()` hard-coded the dtype as `false`. It now looks up the real type from the GGUF at that offset. **This was the
   source of the NaNs** (layers 2/4 read wrong bytes -> huge values -> inf -> NaN); they disappeared after the fix.

## 3. What is not solved (next round starts here)

Symptom: the drafter's proposals disagree with the trunk, 0 of 7 accepted. The magnitude profile (`KATA_DBG_STAGE=1`)
shows the first layer's internals at a sane scale (embed 0.003 -> rms 0.37 -> conv0 0.5 -> attn 0.41 -> wo 3.0 -> ffn conv0
0.15 -> gate 3.3 -> swiglu 1.3), but **the `down` output averages 57** (the residual accumulates to 132 per layer, 8.3M
after two layers). So either the `down` input/weight correspondence is still wrong, or this model's activations really are
that large and the problem lies elsewhere.

Suggested narrowing order (every step is checkable):
1. **Establish the magnitude reference first**: print the trunk's own layer-0 residual magnitude (temporarily print the
   sum |.| of `pX[c]` inside `verifyWide`) and compare it with the drafter's 5.4M. If the trunk is the same order of
   magnitude, the problem is not magnitude but *content* (context K/V, masks, positions).
2. **Self-test the context path**: run the drafter once with "zero context + a single anchor" (the ring holding only the
   block's 8 columns) and see whether the proposals become plausible continuations. If they do, the problem is in
   `materialise` (the order of fc/encNorm/rope, or the tap order); if not, it is inside the layers.
3. **A/B the tap layer numbers**: the GGUF says `[6,20,34,48,62]` (0-based), ninfer's documentation says `[5,19,33,47,61]`
   (1-based). Try both and keep the higher acceptance (this is the only place where the documents disagree).
4. **The selector**: currently per-row argmax (DFlash1 style), which does not need the codebook. Get the argmax version
   right first, then add the selector.
5. **End-to-end acceptance**: within 0.3 of `llama-speculative-simple --spec-type draft-dflash` on the same prompt.

**A finding about oracles**: Prism's `LLAMA_MTP_DRAFT_LOG` only covers MTP drafts (DFlash does not go through that
recording path; the log it produces is empty in practice), so **there is no way to get "the correct draft token per
position" as a direct comparison**. The substitutes available:
- end-to-end acceptance length (the only reliable acceptance criterion);
- ninfer's `docs/maintainer/dflash.md` equations plus Prism's `dflash.cpp` graph (already used for the assembly);
- if a per-layer comparison is needed, recompute one layer's 8 tokens on the CPU with ggml's dequant (slow but viable, as
  a one-off);
- check Prism's acceptance at `--spec-draft-n-max 1`: if it is near 100% at K=1, its single-step draft is essentially
  correct and that becomes the weak yardstick "my implementation should reach".

## 4. Debug hooks (kept in the code, off by default)

- `KATA_DBG_STAGE=1`: sum |.| and NaN count of the key buffers per layer (much slower, debugging only).
- `KATA_DBG_DRAFT=1`: print the proposals, the trunk values and the logit scales.
- `--vwidth T`: the wide-verification regression "accepting only row 0 must reproduce greedy".

## 4.5 Which model file to use (the owner's correction)

- `bonsai-2-27b-mtp-ptq1_0.gguf` (Ollama `mannix/Bonsai-2-27B:mtp-ptq1_0`) is PrismML's PTQ1_0 plus an extra `blk.64.*`
  layer (q8_0, ~0.45 GB) for the MTP head; **every other key and tensor is byte-identical to the PrismML file**.
- **When DFlash2 is the drafter, the source file `Ternary-Bonsai-2-27B-PTQ1_0.gguf` (prism-ml) should be used, without
  blk.64**: it saves 0.45 GB resident and is exactly the combination DFlash2 was built for.
- Both files were tested: the trunk tensors are the same, so the trunk numerics and the `--vwidth` regression are
  unchanged, and DFlash2's tap collection and draft behaviour are unaffected (which also cross-validates the "only blk.64
  was added" claim). The file being downloaded: `../../models/bonsai-2-27b-ptq1_0-src.gguf`.

## 5. The full progress of this round (appended)

**Four bugs fixed**, each confirmed by an observable symptom:

| # | bug | symptom | fix |
|---|---|---|---|
| 1 | the drafter's norm weights bound to the trunk's buffer | logits all zero | use the drafter's own GGUF mapping |
| 2 | Q6_K tensors decoded as Q4_K (`attn_v`, `ffn_down` of `blk.2/4`) | NaN in layer 2 | look up the real type at that offset |
| 3 | **the residual tap collection was never inserted** (an earlier batch replace failed silently) | context K/V all zero; results bit-identical with and without context | insert the `k_f2h` copy after the residual add in `verifyWide`; taps then measure sum \|.\| = 200504 and the ring grows with position |
| 4 | `k_draft_qk`'s `isK` condition was written `h >= H + HKV` (the same as the V branch) | K heads treated as Q heads and **K never written into the ring** -> attention scores always 0 -> uniform softmax -> logits crammed within 0.5 nat | change to `h >= H` |

After fixes 3 and 4 the drafter's output went from "flat / all zero" to a structured distribution (row 0's top-5 spread
from 6.5-7.0 to 9.2-10.3), but **acceptance is still 0**: row 0 still does not predict the anchor (row 0's top-1 is 1428
against the anchor 271 at logit 6.33), and the proposals miss the trunk's 550.

**Narrowing tools now in place**:
- the self-consistency metric "rank of the anchor among row 0's predictions" is printed on every run: currently around
  10th (414 before the fixes), usable as a regression metric.
- the position-offset/semantics A/B needs two changes only: `forwardBlock`'s `pos` (+-1) and which rows become proposals
  (rows 1..7 or 0..6).
- if it is still wrong, do the one-off CPU single-layer comparison: recompute layer 0 on the CPU with ggml's dequant (8
  rows, 72 positions of context) and compare buffer by buffer — the only way to name the first operator that diverges.

## 6. The fifth bug (the decisive one this round): the delta index of the dynamic convolution

In `build_dflash2_conv`, `dynamic` is reshaped by `ggml_reshape_4d(dynamic, n_groups, kernel, 2, n_tokens)`, and ggml's
`ne[0]` **is the fastest-varying dimension**, so the linear index is

```
dyn[t][g][tap][side] = dyn_linear[t*1280 + g + 320*(tap + 2*side)]
```

I had been reading it as the natural `[g][tap][side]` (side fastest), which scrambled all 1280 delta coefficients of every
token.

After the fix (the only change is that one index):
- **row 0's top-1 became the anchor itself (271)** — the self-consistency test passes, the first hard evidence that the
  arithmetic inside the layers is right;
- the 0-based tap layer numbers (anchor logit 5.72) beat the 1-based ones (5.17), confirming that the GGUF's
  `[6,20,34,48,62]` is 0-based;
- the proposals are still wrong (row 1's top-1 is 421 against the trunk's 550), and acceptance is 2%.

## 7. A negative result about "filling the block with real tokens" (so it is not repeated)

To separate "mask handling" from "the arithmetic inside the layers", I tried filling rows 1..7 of the block with the real
continuation. **The experiment is invalid**: attention is non-causal, so with real tokens in place row 0 sees the
"future" and its output changes from 271 to 25 — the configuration no longer makes "row i predicts position F+i"
well-defined. The only valid criterion is end-to-end acceptance under the mask.

## 8. The most likely cause now (the first hypothesis for the next round)

Row 0's self-consistency only proves that the identity path (embedding -> 5 layers -> shared vocabulary head) reproduces
the input token; **it does not prove that the context is used correctly**. Row 1 needs the context to predict 550, and it
fails. So the suspicion concentrates on the context path:
1. the input order of `fc`, or the placement of `encNorm` (implemented per the graph, but with no oracle);
2. whether the context K should get `k_norm` + RoPE and the context V should be left alone (implemented per the graph; an
   A/B with "no RoPE on the context K" is worth one run);
3. whether `target_layers` is 0-based (the evidence now favours 0-based);
4. where the context K/V is written (currently an incrementing `frontier`, confirmed to be covered by a non-zero count).

The next round should go straight to **a one-off CPU single-layer comparison** (layer 0, 8 rows, 72 positions of context,
ggml dequantisation plus plain C++ arithmetic), comparing the attention output buffer by buffer — the only way to get a
definitive answer on "is the context being used correctly".

## 9. Operator-by-operator verification (the decisive work of this round)

To stop guessing, I dumped the GPU intermediates and recomputed them with **independent implementations** (see
`tools/dump_attn.cpp` and `/tmp/fc_check.c`):

| operator | comparison method | result |
|---|---|---|
| `k_draft_mm` (Q4_K/Q6_K matmuls) | ggml's `dequantize_row_q4_k/q6_K` plus a double-precision dot product | **passes** (fc: 0.027 = fp16 rounding; ffn_gate/down/q/conv_proj all pass) |
| `k_draft_conv` (dynamic grouped conv) | plain Python recomputation following `build_dflash2_conv` | **passes** (max difference 0.0) |
| `k_draft_attn` (non-causal sliding-window attention) | plain Python per-key online softmax using the GPU's Q and ring | **passes** (max difference 2e-6) |
| `k_draft_qk` (per-head RMSNorm + RoPE + ring write) | verified indirectly through the attention match | **passes** |
| `fc` + `encNorm` (context materialisation) | ggml dequant + double precision | **passes** |
| final norm + shared head | row 0 self-consistency (predicts the anchor) | **passes** |

**Every operator is right, yet end-to-end acceptance is only ~5%** (6/7 correct on a patterned prompt). So the problem is
not an operator but a **structural assumption**. Remaining suspicions, by likelihood:
1. the **layer numbers** of the target residual taps: the GGUF says `[6,20,34,48,62]`, ninfer's documentation
   `[5,19,33,47,61]`; measured, the difference is small (4.8% either way) and neither reaches the expected level;
2. the **residual position** of the tap: currently "after mixer + FFN" (ninfer's contract); "after the mixer only" is worth
   trying;
3. the **selector**: currently per-row argmax (DFlash1 style). The official DFlash2 uses top-16 plus a rank-256 conditional
   path, which may be significantly better;
4. the **mask** inside the block (non-causal vs causal within the block).

## 10. Diagnostic tools (reproducible)

```
make build/dump_attn                     # dump layer 0's embedding/Q/ring/attention output/conv input and output to /tmp
build/dump_attn ../../models/bonsai-2-27b-ptq1_0-src.gguf ../../models/drafters/dflash2-q4km.gguf
KATA_DUMP_FC=1 ...                     # also dump fc's input/output and c_t (written after the commit)
```

One trap I hit repeatedly: **diagnostic dumps must come after `waitUntilCompleted`** (a shared buffer has not been written
by the GPU during encoding yet). Twice I read stale data because of it and wrongly concluded that `fc` was broken.

## 11. The two Kata-specific drafters (files the owner supplied)

| file | size | quantisation | rotated basis | what it needs |
|---|---|---|---|---|
| `Cyronius/Qwen3.8-27B-DFlash2-Kata2-ternary` | 746 MB | **type 142 (2-bit prism) x46**, F32 x32, BF16 x3 | **yes**: ships `prism.hadamard.sign_widths = [4096, 5120, 17408, 25600]`, `sign_mode = explicit`, `sign_values`, `weight_names` (listing which matrices are rotated); `borrowed.signs_folded = output_norm, selector_hidden_cols` | 2-bit prism decoding plus an FWHT on the input of every rotated matrix (with two extra widths, 4096 and 25600, that the trunk GGUF does not have) |
| `Schiltmans/Ternary-Bonsai-2-27B-DFlash2-ft5-Q2_K` | 623 MB | Q2_K x34, Q3_K x10, Q4_K x5 | **no** (no prism metadata, regular basis) | Q2_K + Q3_K decoding (the existing pipeline suffices) |

Both use `target_layers = [6,20,34,48,62]` (1-based boundaries), i.e. 0-based blocks **5/19/33/47/61**; `Cyronius`'s
`config.json` says `[5,19,33,47,61]` explicitly, matching ninfer's documentation -> our `KATA_TAP_SHIFT=-1` is the correct
setting.

**Code changes**: `k_draft_mm`'s template parameter went from `bool Q6` to `int QT` (0=Q4_K, 1=Q6_K, 2=Q2_K, 3=Q3_K),
selecting the decoder from each tensor's own GGUF type; the GGUF loader recognises Q2_K (84 B/256), Q3_K (110 B/256) and
PQ2_0 (type 142).

**Verification status**:
- Q2_K decoding **passes** (against ggml's `dequantize_row_q2_K`, on a 5120-wide ffn_gate).
- Q3_K is still wrong (a 17408-wide tensor's row 0 disagrees with ggml). Two errors were fixed this round: the offset of
  `qs` after `hmask` (+32), and a missing `half_*8` in the per-half super-block scale index. The remaining difference is
  not located.
- Because ft5's `down` matrix is Q3_K, that drafter is still unusable and acceptance is 0%.
- The ternary one needs a new arithmetic path (2-bit prism decoding + FWHT) and was not started this round.

**Next**: (1) compare Q3_K against ggml element by element (a `k_dq_q3k`-style probe that pinpoints it in one shot, as was
done for Q4_K); (2) for the ternary drafter, implement type 142 decoding and insert an FWHT on the inputs of
`fc`/`attn_conv_proj`/`attn_q/k/v/o`/`ffn_*` (using its own sign vectors for widths 4096/5120/17408/25600), and account for
the sign folding of `output_norm`/`selector_hidden`.

## 12. The DFlash2 selector (per the owner's request, no longer DFlash1's per-row argmax)

Implemented exactly per Prism's `build_dflash2_selector`:

1. `gate = selector_hidden @ h` (BF16, not rotated -> the input is the **unrotated** final normalised hidden state); new
   kernel `k_draft_bf16mm` (BF16 weights x fp16 activations).
2. For each mask row take the **top-16** (by descending score, matching `ggml_top_k`) and their unary logits.
3. Gather two rank-256 codebooks as needed (256 BF16 values per row): `W_pred[prev_candidate_id]`,
   `W_succ[candidate_id]`.
4. Conditional walk: `score[c] = u[c] + sum_r W_pred[P_prev, r] * g[r] * W_succ[C_c, r]`, where the first position's
   predecessor is the anchor (a single row) and afterwards the previous position's **selected candidate index** is the
   predecessor row — matching ninfer's `E_i[p,c] = u_i[c] + sum_r W_pred[pred_token(i,p),r] g_i[r] W_succ[C_i[c],r]`.

`KATA_DF_ARGMAX=1` remains as an A/B (DFlash1 style). Measured: the proposal sequence is clearly different from argmax
(e.g. 760 4087 369 264 ...), but **acceptance is still only 0.6%**.

**Which means the gap is not "argmax versus selector"**: the selector constrains the candidates into a coherent path, but
it still builds on the drafter's per-row distribution, and that distribution is not accurate enough (row 0 is
self-consistent, rows 1 and beyond drift). So the problem is back to the **structural assumptions** — most likely the tap
concatenation order/layer numbers of `fc`, or the input semantics of the block (the anchor's position).

## 13. Cumulative state (for whoever picks this up)
- **11** bugs fixed: PQ2_0/rotated basis, two in Q3_K, the delta index, the RoPE convention, the tap position, the missing
  tap collection, the K-head branch, the norm weight binding, the Q6_K dtype, and the fp16->float conversion.
- Every operator verified individually (decoding vs ggml for Q4_K/Q6_K/Q2_K/PQ2_0; conv, attention and fc vs independent CPU
  implementations; row 0 self-consistent with a sharp logit 9.95 vs 8.40).
- End-to-end acceptance is ~1% (it should be ~70%), for both Kata-specific drafters.
- Conclusion: **what remains is structural, not numerical**. To go further, the valuable thing to obtain is an oracle: the
  per-position draft tokens llama.cpp produces for the **same prompt** under
  `llama-speculative-simple --spec-type draft-dflash` (there is no logging hook today), or the exact construction of the
  `target features` in the official conversion script.

## 14. Generalising ALU verification + the MTP chain + adaptive depth (the owner's item 1)

**What was completed**:
1. **The ALU verification path**: `Engine::aluT` (0/2/4) switches `dWide` between the matrix-unit wide kernel and the ALU
   LUT+half2 kernel; two pure permute kernels, `k_ilv2`/`k_ilv4` (row-major fp16 -> interleaved half2), were added for it,
   **without touching the already-verified norm/FWHT kernels**. Measured per round: T=2 **85.9 ms**, T=4 **113.3 ms**,
   T=8 (wide) 146.6 ms, and `--vwidth` + `KATA_ALU=4` still reproduces greedy token for token.
2. **The MTP chain drafter**, `specGenerateChain`: each step feeds the MTP block's own output as the next step's h; depth 1
   takes the fast `verify2` path (67 ms) and depth 3 takes ALU T=4 (113 ms).
3. **Adaptive depth**: an online comparison of the two settings by "expected tokens per millisecond" (a simplified
   version of Strata's DraftPolicy), initialised from the measured round costs.
4. **One correctness bug fixed**: the chain path originally accepted when "draft == trunk argmax", which breaks the
   sampling distribution; it now uses the exact rule (sample from p, accept on a hit, otherwise take the sampled value).

**Results (reasoning text, greedy)**:

| | tok/s | tokens per round | acceptance |
|---|---|---|---|
| depth 1 (default) | **29.2** | 1.90 | 90.5% |
| chain, depth 3 | 21.2 | 2.46 | 48.4% |

**The chain is currently a net loss**: the per-position acceptance of d_2/d_3 is only about 0.4/0.15, whereas Prism's MTP
chain reaches 0.8/0.7 on the same model (3.13 tokens per round measured). The root cause is **the quality of the h
proxy**: the chain's second step needs an estimate of "the target's hidden state at position n", and I used the MTP
block's own output (with an `output_norm` applied), but acceptance is insensitive to that step (47.7% -> 48.4% before and
after changing the norm), which says the h proxy is not the only problem. The default is back at depth 1, with the chain
kept behind `--mtp-chain`.

**Conclusion**: to get +10-20%, the chain drafter's d_2/d_3 must first reach Prism's level (0.8/0.7). The next step should
be to read llama.cpp's `llama_mtp_chain` (mtp.cpp) and see exactly what it feeds into the second step (h, or some fusion
of h and token, and whether it recomputes the normalisation), instead of guessing further.

## 15. Reading llama.cpp's MTP chain (the owner's request) and the conclusion

**The exact semantics read** (the chain graph in `src/models/qwen35.cpp`, all `n_steps` steps inside one `llama_decode`):

```
step s (s = 0: the catch-up row; s >= 1: one row, position r0 = n_lead-1+s):
    h_norm = RMSNorm(h, nextn.hnorm)          # h is the previous step's h_nextn (step 0 uses the trunk's h)
    e_norm = RMSNorm(embed(argmax_token), nextn.enorm)
    cur    = eh_proj @ [e_norm | h_norm]
    cur    = attn(...) * sigmoid(gate) -> wo -> +cur
    cur    = ffn(RMSNorm(cur, attn_post_norm)) + cur
    h_nextn = RMSNorm(cur, head_norm_w)       # head_norm_w = nextn.shared_head_norm (NOT the trunk's output_norm!) = the next step's h
    next token = argmax(head(h_nextn))
```

Also: the MTP cache is written in position order, and the draft is **greedy** (argmax), not the sampler's token.

**Two fixes that followed**:
1. the chain's next-step h was using the trunk's `output_norm`; changed to `nextn.shared_head_norm` (matching llama.cpp);
2. fixed the chain loop **emitting the anchor twice** (which put 64 duplicates inside "160 tokens").

**The chain is still unusable**, and it exposed more problems:

| metric | depth 1 (default) | chain, depth 3 |
|---|---|---|
| reasoning text tok/s | **29.3** | 12.8 (after the emit fix) |
| first-position acceptance p1 | 0.905 | 0.40-0.50 |
| committed token sequence | token-for-token identical to plain greedy | **wrong** (the periodic pattern `A B A B C B` instead of `A B C A B C`) |

The key anomaly: **the same first draft is accepted 0.9 of the time on the depth-1 path and only 0.4 on the chain path**.
Ruled out:
- the T=4 ALU verification: `--vwidth 4` with **mutually distinct** dummy drafts (which would catch a misaligned token row)
  still reproduces greedy token for token;
- the h normalisation (shared_head_norm vs output_norm) — changed, and the numbers did not move;
- the draft call itself (the same `mtpRows(1, ...)`).

So the problem is in the chain loop's **state/commit bookkeeping**, which also explains why the committed sequence is
wrong. The path was withdrawn from the CLI (`--mtp-chain` removed); the code and this section stay for the next round, and
the default remains depth 1.

**Conclusion**: of the owner's item 1 (generalise ALU verification + MTP chain + adaptive depth), **the ALU verification
infrastructure is complete and verified** (three paths at T=2/4/8, a nine-shape sweep), but **the chain drafter has an
unresolved bookkeeping bug**, so it was not adopted. To get the +10-20% on reasoning text, the next step is to audit the
chain loop's `commitWide` / `encMtpKv` / `nextTok` bookkeeping rather than the drafter itself.

## 16. Auditing the chain bookkeeping (the owner's next item)

Forced-depth measurements, after the fixes above:

| | per round | tokens per round | acceptance | tok/s |
|---|---|---|---|---|
| depth 1 through the chain pipeline | 72 ms | 2.00 | 100% | 28.8 |
| depth 3 through the chain pipeline | 113 ms | 1.51 | 50.6% | 12.5 |

The anomaly remains: **p1 = 0.47 at depth 3 against 1.00 at depth 1**, for the same `mtpRows` call. Ruled out: the h
transfer (printed, `sum|h| ~ 4479`, sane), the T=4 verification (already shown to reproduce greedy token for token with
distinct dummy drafts), and the draft call itself.

Three further bugs were found and fixed during the audit:
1. **`bufH` is fp16 but was read as float** — in `Engine::hidden()` and in the stepwise branch of `feed()`, so a short
   suffix (<= 24 tokens) had a garbage `hPrev`;
2. the chain loop **emitted the anchor twice**;
3. the chain's next-step h used the trunk's `output_norm` instead of `nextn.shared_head_norm`.

**Conclusion**: the item was not achieved. The ALU verification infrastructure is done and verified, but the chain
drafter still has an unresolved bug, so the default stays at depth 1 (25.4 tok/s prose, 28.8-29.3 reasoning) and the chain
remains behind `KATA_CHAIN=1`.

## 17. `--fast-ffn` becomes the default; the ceiling of the wide-verification r

**The default was switched**: `--fast-ffn` (int8 FFN prefill) is now the default path, with `--exact` to disable it. A
7.3K prompt went from **207 to 227 tok/s (+10.7%)**; the quality evidence is unchanged (99.87% top-1 over 3072 positions,
KL 7.5e-5). Stacking `--kv int8` on that default measured 99.58% (below the 99.7% gate), so that combination falls back to
the exact prefill with a printed note.

**Pushing r down (a negative result)**: I implemented `k_wide2`, "two independent row groups per simdgroup", hoping to
overlap the MMA of group A with the decode of group B (N=8 x 2, same threadgroup memory as N=16). The numerics are
exactly right (identical max |diff| to `k_wide`), but it is **slower**: 663 us against 421 us on the 5120->17408 matrix,
because MPP's `op.run()` retires synchronously for the issuing warp ("decode B overlaps MMA A" does not happen) while the
extra barrier per chunk is real.

**The hard floor for r**: the `k_wide` ablation puts 306 of the 420 us in the MMA (9.4 TFLOPS, 54% of peak at M=16) and
~110 us in the decode. Even with the decode fully hidden, r can only reach 306/170 = **1.8** (verify(8) ~ 51 x 1.8 + 15 =
107 ms, DFlash2 at about 40 tok/s). Reaching r ~ 1.6 would need the MMA at 11.7 TFLOPS (69% of peak), which is a shape
limitation of MPP at M=16 rather than something the code can work around. So "wide verification 2.47 -> 1.6" is **not
achievable on the M6**; 1.8 is the reachable ceiling and requires a different overlap mechanism.
