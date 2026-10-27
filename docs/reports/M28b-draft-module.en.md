# M28b: the draft module (loading, ops, geometry) and the exact to-do list for M28c

**English** · [简体中文](M28b-draft-module.md)

Date: 2026-10-25 · The second step of M28. The owner's instruction: slot in M27a first to raise the ceiling (done,
`reports/M27a-small-grid-fixes.md`), then go straight to **Q4_K_M** (decoding and the forward ops are done, see §2).

## 1. The authoritative graph, taken from Prism's source (no more guessing from prose)

`refs/llama.cpp-prism/src/models/dflash.cpp` is the only authority for this implementation (the drafter is an ordinary
5-layer model; there is no dedicated Metal kernel):

**Context (encoder)**: `c_t = plain_rmsnorm(fc(concat(5 target residuals)), enc.output_norm)`; `fc` is a 25600->5120 Q4_K;
`enc.output_norm` is an RMSNorm applied **after** `fc`.

**Per layer (decoder, il = 0..4)**:

```
n = rmsnorm(inpL, attn_norm)
dyn = attn_conv_proj @ n ;  n = conv(n, dyn, attn_conv_base, side=0)
Q, K, V = wq/wk/wv @ n            # the same K/V weights are also used for the context injection
Q = per_head_rmsnorm(Q, q_norm);  K = per_head_rmsnorm(K, k_norm)
Q = rope(Q, pos); K = rope(K, pos)          # only the first 64 dims, dimension_sections [64,0,0,0]
cur = attn(Q, K, V, non-causal, sliding 2048) @ wo
cur = conv(cur, dyn, attn_conv_base, side=1)          # the output side reuses the dyn computed on the input side
ffn_inp = cur + inpL
n2 = rmsnorm(ffn_inp, ffn_norm)
dyn2 = ffn_conv_proj @ n2 ; n2 = conv(n2, dyn2, ffn_conv_base, side=0)
cur = ffn_down(silu(ffn_gate(n2)) * ffn_up(n2))
cur = conv(cur, dyn2, ffn_conv_base, side=1)
inpL = cur + ffn_inp
out = rmsnorm(inpL, output_norm); logits = target_lm_head(out)
```

**The exact conv semantics** (`build_dflash2_conv`): `out[t][c] = sum_tap (base[side][tap][c] + delta[t][g(c)][tap][side]) *
X[t-tap][c]` with `X[-1] = 0`; `delta` is conv_proj's output, 1280 values per token, ordered
`[group(320)][tap(2)][side(2)]`; `base` is laid out `[side][tap][c]` (c fastest). This matches ninfer's documentation.

**Inputs and outputs**: the drafter uses the target model's `tok_embd` (**the plain, de-Hadamarded embedding**, i.e. our
`k_embed_b` output) and the target's `lm_head` (PTQ1_0 + Hadamard, so sign + FWHT then the wide GEMV).

## 2. Completed this round

| item | status |
|---|---|
| Q4_K / Q6_K decoding | done — **bit-identical** to ggml (258,048 elements, difference 0.0) |
| Q6_K vectorised | done — 17 -> **73.5 GB/s** (draft round 21 -> **14 ms**) |
| drafter matmul `k_draft_mm` | done — Q4_K 76-118 GB/s, row 0 matches a CPU dequantisation |
| dynamic convolution `k_draft_conv` | done — exact semantics above (side 0/1, delta layout, zero padding) |
| drafter QKV prep `k_draft_qk` | done — per-head RMSNorm (32/8 heads, 128 dims) + partial RoPE (first 64, pairs (i,i+32)) + ring write |
| non-causal sliding-window attention `k_draft_attn` | done — one threadgroup per (head, row), window 2048, online softmax, ring 4096 |
| context K/V `k_draft_ctx_kv` | done — written into the ring after k_norm + rope |
| loader `src/draft.{h,cpp}` | done — all 81 tensors bound, ring buffers (5x4096x1024x2x2 B = 80 MB), scratch; `build/draft_probe` prints the geometry |
| **forward assembly + selector + spec integration** | M28c |

## 3. The M28c to-do list, in order, each step checkable

1. **`Draft::materialise`**: `catH` (fp16 [n][25600]) -> `k_draft_mm` (fc, Q4_K, MT=8) -> `k_rmsnorm_rows_h` (encNorm) ->
   `cH`; per layer `k_draft_mm` (wk/wv, Q4_K, 5120->1024) -> `k_draft_ctx_kv`. Acceptance: it cannot be compared directly
   against llama.cpp's `enc_norm_out`, so the end-to-end acceptance of the whole draft is the criterion.
2. **`Draft::forwardBlock`**: assemble per the graph in §1 (4 `k_draft_mm` + `k_rmsnorm_rows_h` + `k_draft_qk` +
   `k_draft_attn` + `k_draft_conv` twice per block, twice per layer + `k_swiglu` + `k_add2`), ending with
   `k_add_rms_fwht_t` (outNorm, flags=0) + `eng.xWideGemv(outW)` to get 8 x 248320 logits.
3. **Proposals**: get the whole chain running with **per-row argmax** first (DFlash1 style, skipping the selector) and
   measure acceptance; add the selector afterwards (two codebooks + a rank-256 walk, about +0.3-0.7 tokens per round).
4. **Spec integration**: add a draft callback to `specGenerate`; accept prefixes with the existing greedy/sampling rules;
   after `commitWide`, call `materialise` to fill in the context of the newly committed positions.
5. **Acceptance**: within 0.3 of Prism's `llama-speculative-simple --spec-type draft-dflash` on the same prompt; end to end
   >= 31 tok/s on code/reasoning (today 25).

## 4. The arithmetic now (M27a plus the measured drafting cost)

| configuration | per round | code/reasoning | prose |
|---|---|---|---|
| today (plain decode / MTP-1) | 50.7 / 62 ms | 20 / 25 tok/s | 25 |
| M27a verification + 14 ms draft, argmax proposals | 156 ms | **33 tok/s** (at 5.2 tokens per round) | 13 |
| the same + selector (if it adds 0.5 tokens per round) | 156 ms | 36 | 13.5 |
