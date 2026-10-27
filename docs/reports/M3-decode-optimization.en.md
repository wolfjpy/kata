# Milestone M3: decode optimisation

**English** · [简体中文](M3-decode-optimization.md)

Date: 2026-10-24

## Results

| stage | per token | tok/s | note |
|---|---|---|---|
| M2 reference kernels | ~500 ms | ~2 | per-lane element-wise decode |
| + fast GEMV (`k_gemv_v1`) | 52.4 ms | 19.1 | operating directly on the GGUF layout |
| + key-major GDN state, fused alpha/beta, fused residual-add and norm, split attention | **49.6 ms** | **20.1** | numerics unchanged: 5/5 top-1 on a 5-token prompt, max logit difference 0.0005 |
| baseline (Prism llama.cpp `tg128`) | 63.9 ms | 15.64 | |

**+29% over the baseline.**

## The key idea in the fast GEMV

- Each 128-weight block is handled cooperatively by 8 lanes: lane `it` owns bytes `qs[2it]` and `qs[2it+1]` (elements
  `n*16+2it+{0,1}`), `qs[16+it]` (elements `80+n*8+it`) and one trit of `qh[it&1]`.
- The decode runs in a pure-float pipeline: a byte is the base-3 fraction `u = b/256` with
  `t_n = floor(3^(n+1) u) - 3 floor(3^n u)`, hence
  `sum_n t_n y_n = sum_{k=1..4} g_k (y_{k-1} - 3 y_k) + g_5 y_4` with `g_k = floor(3^k u)`. The coefficients depend only on
  the activation, and each byte costs five `floor` plus five `fma` operations. (The idea comes from the comments in the
  Prism kernel; the implementation is ours.)
- Each simdgroup handles 4 adjacent output rows, so the activation coefficients in registers are reused 4 times.
- Microbenchmark (`tools/bench_gemv`, real model tensors, rotating through 8 different layers to avoid caching):

| shape | reference kernel | `v1_nr4_sg4` | share of the 142 GB/s sustained bandwidth |
|---|---|---|---|
| ffn gate/up 5120->17408 | 12.6 GB/s | **131.6 GB/s** | 93% |
| ffn down 17408->5120 | 16.3 GB/s | **127.0 GB/s** | 89% |
| gdn qkv 5120->10240 | 12.5 GB/s | **129.1 GB/s** | 91% |
| lm_head 5120->248320 | 12.8 GB/s | **134.1 GB/s** | 94% |

- Conclusion: **the GEMV is not ALU-limited and needs neither a re-layout nor a LUT.**

## Ablation, item by item (per token, against a 49.6 ms baseline)

| item | cost |
|---|---|
| all GEMVs (including lm_head) | 45.0 ms (5.657 GB / 45 ms = 126 GB/s) |
| ffn gate / up / down | 9.1 / 9.7 / 10.0 ms |
| gdn qkv / z / ssm_out | 4.3 / 1.9 / 2.6 ms |
| attention q / k+v / o | 1.0 / 0.3 / 0.8 ms |
| lm_head | 1.3 ms |
| GDN recurrence step | 2.6 ms (48 layers x 54 us; the state read/write floor is 48 us) |
| fused residual + norm | 0.5 ms |
| conv, alpha/beta, 3 FWHTs, attention (short context) | each < 0.1 ms |
| all non-GEMV work together | 5.4 ms |
| idle (embed + submit) | 0.4 ms |

## Remaining headroom and the next step

- Pure GEMV inside the real context runs at 126 GB/s (130-134 in isolation); the tail/launch losses between kernels cost
  about 3 ms.
- The GDN state read/write floor is about 2.3 ms (fp32); an fp16 state would save about 1.2 ms but has to clear the
  quality gate.
- Grinding further on this path is worth about 10% more (to ~22 tok/s). **The bigger levers are MTP speculative decoding
  (~1.6-1.8x) and prefill (today it is the same speed as decode, ~20 tok/s, against a baseline of 208).**

## Other changes

- Added `--skip name,...` (kernel-name ablation) and `--gen N` (which prints the decode speed).
- The lm_head is only computed at the last prompt position.
- Attention moved to flash-decoding: `k_attn_split` (4 KV heads x S splits, with the 6 Q heads of a GQA group sharing the
  K/V reads) plus `k_attn_combine`.
