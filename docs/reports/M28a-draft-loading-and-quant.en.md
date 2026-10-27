# M28a: loading the DFlash2 drafter and decoding Q4_K / Q6_K

**English** · [简体中文](M28a-draft-loading-and-quant.md)

Date: 2026-10-25 · The first step of M28 (loading + quantised decoding + drafter matmuls), preparing the ground for the
5-layer forward and the selector. The weights come from an existing file: `Q4_K_M` from
`incoai/Qwen3.8-27B-DFlash2-GGUF` (locally `../../models/drafters/dflash2-q4km.gguf`).

## 1. First, how much work is "porting" actually

In `refs/llama.cpp-prism` the DFlash2 drafter is `src/models/dflash.cpp`, and there is **no DFlash-specific Metal
kernel** — it is an ordinary 5-layer model running through standard ggml ops. So what is needed is loading plus a handful
of conventional ops, not reverse-engineering a new runtime.

GGUF metadata (measured):

```
dflash.block_count = 5            dflash.block_size = 8            (W=8, K=7)
dflash.embedding_length = 5120    dflash.feed_forward_length = 17408
dflash.attention.head_count = 32  head_count_kv = 8  key_length = 128  causal = False
dflash.attention.sliding_window = 2048, pattern = [T,T,T,T,T]
dflash.conv_kernel_size = 2, conv_group_size = 16
dflash.selector_rank = 256, selector_top_k = 16
dflash.target_layers = [6, 20, 34, 48, 62]        tokenizer.ggml.mask_token_id = 248070
```

81 tensors: **45 Q4_K + 4 Q6_K + 32 F32**, 1132 MB in total (Q4_K 976 + Q6_K 155 + F32 1.1). Q6_K appears only in `attn_v`
and `ffn_down` of `blk.2`/`blk.4`. Each layer has attn_conv_base/proj, q/k/v/output, q/k norm, attn_norm,
ffn_conv_base/proj, gate/up/down and ffn_norm; plus `fc.weight` (25600->5120, projecting the concatenation of the 5 target
residuals), two `output_norm`s and `selector_*`.

## 2. What was completed (all with numerical comparisons)

**(1) GGUF loading**: the existing loader gained the byte sizes for Q4_K (144 B/256) and Q6_K (210 B/256); tensor
type/shape/offset support was already there.

**(2) Q4_K / Q6_K decoding** (`shaders/draft.metal`): **bit-identical to ggml's own `dequantize_row_q4_K/q6_K`** —
`tools/dq_probe` compares 8 real tensors x 3 rows x every K block, **258,048 elements, max difference 0.0**. Three mistakes
of my own were fixed on the way: only 32 bytes are consumed per 64-element group (I had written 32 groups), the low and
high nibbles inside one 64-element group are **two different** (scale, min) pairs, and the low nibble of the top byte
needs `& 0x0F`.

**(3) The drafter matmul `k_draft_mm`**: reuses `k_wide`'s "decode into a private fp16 tile -> single-simdgroup
`matmul2d`" structure, instantiated with MT=8 block columns, N=16 rows per simdgroup, KC=128 and 4 simdgroups; Q4_K
processes 16 bytes at a time (16 low nibbles + 16 high nibbles = 32 weights), while Q6_K is decoded element by element
(correct but slow).

Correctness and speed on real tensors (`tools/bench_draft`, row 0 against a CPU-dequantised dot product):

| tensor | type | size | time | bandwidth |
|---|---|---|---|---|
| ffn_gate 5120->17408 | Q4_K | 50 MB | 0.64 ms | **78.5 GB/s** |
| ffn_down 17408->5120 | Q4_K | 50 MB | 0.61 ms | 82.7 |
| attn_q 5120->4096 | Q4_K | 11.5 MB | 0.10 ms | 117.7 |
| conv_proj 5120->1280 | Q4_K | 3.6 MB | 0.06 ms | 58.2 |
| ffn_down 17408->5120 | Q6_K | 73 MB | 4.28 ms | **17.1** (needs vectorising) |

## 3. Corrected drafting cost and the DFlash2 payoff

- One draft round reads 1132 MB. At today's speeds: Q4_K 976 MB at ~80 GB/s = 12.2 ms, Q6_K 155 MB at 17 GB/s = 9.1 ms ->
  **about 21 ms**. Vectorising Q6_K to ~60 GB/s gives **about 15 ms**.
- Combined with M27a's verification cost (142 ms):

| configuration | per round | code/reasoning (K=7, 5.20 tokens) | prose (2.07) |
|---|---|---|---|
| M26 + 9 ms draft (old estimate) | 176 ms | 29.5 tok/s | 11.8 |
| **M27a + 21 ms draft (measured)** | **163 ms** | **31.9 tok/s (+28%)** | 12.7 |
| M27a + 15 ms draft (Q6_K vectorised) | 157 ms | 33.1 (+32%) | 13.2 |

So the DFlash2 expectation is revised to **+28% to +32%, on code/reasoning only**; prose still needs M29's width-adaptive
fallback.

## 4. The remaining M28 work, in order

1. **Vectorise Q6_K** (half a day, saves 6 ms): follow Q4_K's 16-byte job idea, taking 16 bytes of `ql` and `qh` -> 32
   6-bit values plus two int8 scales per group.
2. **Context materialisation** (0.3 days): the trunk's 5 residual taps (layer-block outputs 6/20/34/48/62) -> `fc`
   (25600->5120) -> each layer's `W_ck/W_cv` projection -> ring KV (2048 window x 5 layers x 8 kv heads x 128 x 2 x 2 B =
   40 MB).
   Note: the GGUF has only `attn_k`/`attn_v` per layer, with no separate context K/V projection — per ninfer's
   documentation, "Source K/V parameters are used for both query-block and context projections" (the official recipe
   shares the same physical data), so the context K/V reuses the `attn_k`/`attn_v` weights and follows `dflash.md`'s
   `k_ctx = rope(plain_head_rmsnorm(W_ck c_t))`.
3. **The 5-layer forward** (0.5 days): plain RMSNorm, QKV, per-head qk norm, scalar RoPE (sections [64,0,0,0], i.e. rotate
   only the first 64 dims in pairs (i, i+32)), **non-causal sliding-window attention** (`|p_key - p_query| < 2048`,
   including the block's 8 columns), the two-tap dynamic grouped convolution, and SwiGLU.
4. **The selector** (0.3 days): each mask row's top-16 comes from the **target vocabulary head** (one `k_wide` call
   computes 8 rows), 16 rows gathered from each of `selector_predecessor`/`selector_successor` (256 Q4_K values per row),
   `g_i = W_hidden h_i` (5120->256), and a conditional walk by
   `E_i[p,c] = u_i[c] + sum_r W_pred[pred,r] g_i[r] W_succ[C,c,r]`.
