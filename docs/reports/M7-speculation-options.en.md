# M7: the other speculation options (DFlash2, a 2B drafter, deeper MTP) and the cost of wide verification

**English** · [简体中文](M7-speculation-options.md)

Date: 2026-10-24 · All numbers measured on the same Mac mini M6.

## 1. The question

Today we have MTP depth 1 with a verification width of 2 (25 tok/s). Could a stronger drafter (the 2B drafter, DFlash2, a
deeper MTP chain) go faster?

## 2. Measured acceptance (Prism llama.cpp's `llama-speculative-simple`, Bonsai-2-27B as the target, greedy)

Prism's own Metal speculation path is slow (5-9 tok/s — ignore the speed), but the acceptance statistics are valid.
"Tokens per round" = decoded tokens / verification rounds (each round = accepted tokens + 1 extra).

| drafter | prose (no thinking) | code (no thinking) | reasoning (thinking) |
|---|---|---|---|
| MTP depth 1 | 1.6-1.64 (measured in our engine) | — | 1.86 |
| MTP depth 3 (chain) | 1.87 | 3.2 | 3.13 |
| **DFlash2** (7 draft tokens, Q4_K_M) | 2.2 | **5.1** | **5.1** |
| 2B drafter model (an ordinary draft model, 4 draft tokens) | 1.23 | 2.07 | — |

Points that matter:
- **Acceptance depends strongly on how predictable the text is**: the same MTP chain gives 1.87 on prose and 3.2 on code.
  Thinking mode is predictable reasoning text, and the default mode sits on the high-acceptance side.
- **DFlash2 transfers to Kata very well**: `z-lab/Qwen3.8-27B-DFlash2` (1.92B parameters, 1.14 GB as GGUF Q4_K_M) was
  trained for the original Qwen3.8-27B, yet on the ternary-quantised Kata it still reaches an acceptance length of 5.1 on
  code/reasoning (the official H200 numbers: GSM8K 5.46, MTP chain 5.02).
- **A 2B draft model is not worth it**: acceptance length only 1.2-2.1, below the MTP chain, and each draft token costs
  0.44 GB of reads.
- DFlash2 is block-diffusion: one forward produces the whole block (8 tokens), so the bandwidth cost per draft token is
  only ~0.14 GB (Q4_K_M), far below a per-token draft model — but it needs the target model's hidden states for KV
  injection, so the draft must wait for verification and cannot run in parallel on the CPU/ANE.

## 3. The binding constraint: the cost of a wide verification kernel (`tools/bench_gemv2`, ffn_gate 5120->17408, per matrix)

| kernel | T (tokens verified at once) | time |
|---|---|---|
| LUT + half2 (in use) | 2 | 170-185 us |
| same (2 half2 pairs) | 4 | 296 us (+60%) |
| same | 6 | 397 us |
| same (4 half2 pairs per matrix, NR=2) | 8 | 590-598 us (**3.2x**) |
| the matrix-unit prototype `k_wide_gemm` (fp16 threadgroup block + `matmul2d`, M=16) | 16 | 831 us |

- The existing multiply-add kernel slows down roughly linearly with T (ALU-bound), so a T=8 verification costs about
  160 ms per round -> one DFlash2 round of ~170 ms producing 5.1 tokens is ~30 tok/s, **no better than today's MTP depth 1
  (~29 tok/s on reasoning text)**. Depth 3 MTP is the same (~30 tok/s). So with the current kernels depth 1 really is the
  optimum, which matches the earlier estimate — now with data.
- The matrix-unit prototype is **numerically correct** (max error 3e-6, far better than half accumulation, with the weights
  decoded exactly to fp16) but **slow**: each K block serialises "load -> decode -> barrier -> matmul2d -> barrier", and
  each threadgroup uses 20 KB of threadgroup memory, so only one threadgroup can reside per core and there is no latency
  hiding; roughly 3.6 us per step against a budget of ~0.8 us. In principle a T=16 verification could approach the
  bandwidth limit (decode the weights once, multiply-adds nearly free on the matrix unit), but it needs software
  pipelining, smaller tiles (so that 2+ threadgroups fit per core) and asynchronous copies — not a small change.

## 4. Conclusion and recommendation

| option | expectation | prerequisite | recommendation |
|---|---|---|---|
| keep MTP depth 1 | 25 tok/s | — | keep |
| MTP depth 3 / adaptive depth | reasoning/code ~+0-5%, prose -20% | current kernels | no |
| a 2B draft model | below MTP | — | no |
| **DFlash2 + wide verification** | reasoning/code ~60 tok/s (estimate: draft ~9 ms + wide verification ~65 ms = ~75 ms per 5.1 tokens); prose about break-even | (1) a wide verification kernel at T=8-16 within ~1.5x the T=2 cost; (2) DFlash2 implemented in Metal (block-diffusion transformer, two-tap dynamic convolution, selector, KV injection, Q4_K_M dequantisation); (3) extracting hidden states from several target layers; (4) lossless sampling for block drafts | **the only direction that can significantly raise decode speed, but multi-day and high risk**; the advice was to spend 1-2 days getting the wide verification kernel to target (the decision gate) first |
| a CPU/ANE drafter | — | — | under UMA the bandwidth is shared, a drafter's DRAM traffic contends with the GPU and only latency can overlap; DFlash2 additionally depends on the target's hidden states so it cannot run in parallel; and the ANE cannot natively run ternary weights. Not recommended |
