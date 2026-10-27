# Milestone M5: batched prefill (the exact fp16 path and the int8 path)

**English** · [简体中文](M5-prefill.md)

Date: 2026-10-24

## Design

- Each chunk (up to 1024 tokens) is processed layer by layer. For every ternary matrix, the PTQ1_0 weights are first
  expanded on the GPU into a tile-major matrix layout and then multiplied with MPP `matmul2d`. Expansion and GEMM are
  serialised in the same command buffer, and the weights only ever live on the GPU at 1.75 bpw.
- **Path A (exact)**: expand to fp16 (`w = scale * trit`; fp16 times ±1/0 is exact), `half x half -> float`, tile 32x64,
  one simdgroup, N-major.
- **Path B (int8)**: expand to int8 trits plus a per (row, 128-group) fp32 scale; activations quantised to int8 with one
  scale per token; `int8 x int8 -> int32` with **an exact scale epilogue per K=128 group** (the int32 accumulator lives in
  a cooperative tensor, is multiplied element-wise by that group's weight scale, and is accumulated into an fp32
  accumulator); tile 64x64, 4 simdgroups.
- Every other operator got a batched version: embedding, fused residual-add + RMSNorm + signs + FWHT (one threadgroup per
  token), alpha/beta fused into one 128-column GEMM, conv (sliding window), q/k L2 norm, gating, GDN recurrence, gated
  RMSNorm, and attention (batched RoPE + KV write + split flash-decoding-style attention).
- After prefill, the GDN state, conv state and KV cache are fully compatible with per-token decode, so decoding can
  continue immediately.

## What the cooperative-tensor API turned out to do (see `docs/probes/int8_epilogue_api_debug.cpp`)

- `op.get_destination_cooperative_tensor<A, B, int>()` and `<..., float>()` have the **same layout**, so they can be mixed
  element by element: `cAcc[i] += float(cInt[i]) * scale`.
- `get_multidimensional_index(i)` returns `[column n, row m]`.
- **A broadcast `cS.load(broadcast tensor)` gives wrong results** (measured), so the scales are indexed element-wise, with
  the column and row numbers computed once outside the loop.
- Tiles 64x128 and 32x64 spill registers and reach only 4-5 TOPS; **64x64 with 4 simdgroups reaches 26 TOPS** including
  the exact epilogue, 2.1x the sustained fp16 rate of 12.6.

## Quality (72 positions, against per-token fp32 as the gold standard)

| path | mean KL | max KL | top-1 agreement |
|---|---|---|---|
| the llama.cpp reference | 7.1e-7 | 3.5e-6 | 72/72 |
| A: exact fp16 | 1.1e-7 | 5.6e-7 | 72/72 |
| B: int8 | 1.9e-4 | 7.7e-4 | 72/72 |

- A is essentially equivalent to fp32 (closer to it than llama.cpp is).
- B's mean KL of 1.9e-4 is below the proposed 1e-3 threshold; for reference that is the magnitude of a Q8 quantisation of
  the weights, on a model that is already 1.58 bit.
- **Only 72 positions were sampled**, so B cannot become the default until it is confirmed on a larger, more varied corpus
  (see the to-do list).

## Speed (1024-token chunk, warm, this machine)

| path | time | tok/s | against the baseline (llama.cpp `pp512` 208) |
|---|---|---|---|
| A: exact fp16 | 5.60 s | **183** | 0.88x |
| B: int8 | 3.68 s | **278** | **1.34x** |

Breakdown of the int8 path (1024 tokens, 4.28 s before optimisation): GEMM 2.29 s, weight expansion 0.60 s, quantisation
0.04 s, everything else (GDN, attention, conv, FWHT) 1.48 s.

## Findings along the way

- **The first timing included pipeline compilation** (MPP's GEMM kernel takes about 2 s to compile), so the benchmark has
  to warm up first (`--repeat 2`).
- The original GDN recurrence held 128 state registers per thread and **spilled to memory**, costing 0.83 s; giving each
  state column to 4 lanes (32 registers per thread) and reducing with `simd_shuffle_xor` brought it to about 0.14 s.
- Turning alpha/beta from a per-token GEMV (which re-read 47 GB of weights per token) into one 128-column GEMM took it
  from 0.34 s to nearly zero; the norm went from 5 threadgroups per token re-reading, to 1.

## To-do (further prefill optimisation)

1. int8 weight expansion 0.60 s -> ideally 0.2 s (the threadgroup transpose has bank conflicts and the decode uses a
   generic slow function).
2. GEMM 2.29 s -> the microbenchmark says 26 TOPS (1.9 s): auto-tune the tile and grid order per shape.
3. Attention is currently a per-token split decode, O(N^2) and without K/V reuse; replace it with matrix-unit flash
   attention.
4. Layer-major prefill, so that the weight expansion happens once per prompt, and longer prompts become possible.
5. Quality: validate the int8 path on a varied corpus of >= 100k tokens before making it the default.
6. Per-group activation scales (rather than per-token) would lower the int8 error further, at the cost of one more memory
   access in the epilogue.
