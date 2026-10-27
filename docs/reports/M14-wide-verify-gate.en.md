# M14: the wide-verification microbenchmark gate — not passed, DFlash2 (M15/M16) closed

**English** · [简体中文](M14-wide-verify-gate.md)

> **This verdict has been overturned (2026-10-25).** The measurements here remain valid, but the "closed" judgement was
> wrong: it compared against T=2 rather than the bandwidth floor, and the prototype kernel was flawed on K=128, M=8, N=16
> and serialisation. See **`R2-dflash2-revisit.md`**: DFlash2 was reopened (M25-M29).

Date: 2026-10-25 · Corresponds to CHECKPOINT-003 roadmap item M14. The owner's decision: since DFlash2 is a multi-day,
high-risk project, gate it behind a wide-verification microbenchmark first. **Gate: T=8 must cost <= 1.5 x T=2 per matrix
(~255 us at ffn_gate).** Result: **not passed**, DFlash2 not done.

## 1. Design (taking Strata's structure, using the M6's matrix hardware)

`shaders/wide.metal`, `tools/bench_wide.cpp`. Each simdgroup owns R output rows independently (no threadgroup-level
barrier, only `simdgroup_barrier`): for each 128-wide K block it decodes its own R x 128 weights through the 16-byte LUT
into a **private** threadgroup tile (row stride padded by 8 elements to avoid bank conflicts), then a single-simdgroup
`matmul2d` (M = token rows, N = R, K = 128) multiplies the fp16 activations (read straight from device memory) by that
tile, accumulating in a cooperative tensor. The weights are decoded once and shared by all tokens. Two variants:
- `k_wide_sg`: the tile is fp16 (scale already applied, exact);
- `k_wide_q8`: the tile is int8 ternary (exact, no scale applied), `half x int8 -> float`, with the per-128-block row scale
  applied to the MT x R partial results before accumulating (4 elements per lane).

Both problems of yesterday's prototype (`k_wide_gemm`, 831 us per 16 tokens) were solved: 4 simdgroups sharing a 20 KB
block became a private 4-8 KB tile per simdgroup, and threadgroup-level barriers became simdgroup-level ones. **The
numerics are correct** (max error 1e-6 to 1e-5 against an fp32 reference, two orders of magnitude better than the
half-accumulating LUT kernel); **M=8 `matmul2d` works**, and M=8 costs the same as M=16 (the token dimension is
essentially free).

## 2. Results (per matrix, us; compared within one run)

| shape | T=2 LUT (in use) | T=4 FMA-LUT | T=8 FMA-LUT | **T=8 wide fp16** | **T=8 wide int8** | wide int8 / T=2 |
|---|---|---|---|---|---|---|
| ffn_gate/up 5120->17408 | 170 | 297 | 589 | 416 | **367** | **2.16x** |
| ffn_down 17408->5120 | 174 | 307 | 615 | 441 | **388** | **2.23x** |
| gdn_qkv 5120->10240 | 101.5 | 180 | 351 | 256 | **233** | **2.30x** |

The 1.5x gate is missed on all three shapes (2.2-2.3x). The wide kernel is 1.4-1.6x faster than the T=8 FMA version but
slower than the FMA version below T=4, so it only makes sense at T >= 6.

## 3. Why it is stuck here

Ablation (M=8, R=16, fp16 variant): 416 us total; skipping the decode gives 281 us, so **`matmul2d` itself needs ~280 us**
and the decode contributes ~135 us. Raising the K block from 128 to 256/512 scales the time proportionally (620/908 us),
which says it is not per-call fixed overhead but **how many weight bytes per second the matrix unit can ingest from a
threadgroup-memory B operand**: about 318 G weights/s (ffn_gate's 89M weights / 280 us), half of the 640 G weights/s
bandwidth floor. No amount of decode optimisation can take a T=8 matrix below ~280 us. The int8 variant halves the tile
bytes and buys only 12%.

## 4. The economics (why "closed" was the right call at the time)

At 367 us: the GEMV part of a T=8 verification is 51 ms x 367/170 = 110 ms, plus about 15 ms of extra attention/GDN cost at
T=8, so one verification round is ~125 ms; the DFlash2 draft is ~9 ms.

| text | DFlash2 tokens per round | round time | tok/s | today's MTP depth 1 |
|---|---|---|---|---|
| code / reasoning | 5.1 | ~134 ms | **~38** | ~29 (1.8 tokens / 64 ms) |
| prose | 2.2 | ~134 ms | ~16 | ~25 |

Even on the most favourable text that is only ~+30%, prose is actually 35% slower (needing an adaptive fallback), and the
price is a multi-day, high-risk port (block-diffusion transformer, selector, dynamic convolution, context materialisation,
ReplaySSM replay, Q4_K_M dequantisation). **Not worth it.**

## 5. What would change this conclusion

- Feeding the B operand to `matmul2d` from registers instead of threadgroup memory (the spec has
  `get_right_input_cooperative_tensor`, letting each lane decode the elements it owns with arithmetic, skipping the tile
  and the barriers). **This is the only untried variant**; if the matrix unit's B ingest rate is the limit it could double,
  but the elements a lane owns are scattered, so the decode becomes per-element arithmetic (~3-4 instructions each) and the
  payoff is uncertain.
- A newer OS or hardware (M6 Pro/Max have more matrix units) changing `matmul2d`'s small-tile throughput.
- If a stable acceptance length of >= 5 and a T=8 verification at <= 255 us per matrix ever coexist, re-evaluate DFlash2.

## 6. Artifacts

`shaders/wide.metal` (both wide kernels; experimental, unused by the engine) and `tools/bench_wide.cpp` (the gate
benchmark, `make build/bench_wide`; e.g.
`build/bench_wide $M build/kata.metallib k_gemv_lut2_nr4_sg4 k_wide_q8_m8_r16_g4 ...`). The R=32/64 int8 variants run at
similar speed but give wrong results (cause not investigated) and are not used in the report.
