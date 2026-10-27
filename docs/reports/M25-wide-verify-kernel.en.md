# M25: the wide verification kernel (paving the way for DFlash2)

**English** · [简体中文](M25-wide-verify-kernel.md)

Date: 2026-10-25 · Corresponds to M25 in `R2-dflash2-revisit.md`. Prerequisite: R2 overturned M14's "closed" verdict and
decided to reopen.

## 1. Goal and result

| item | target | result |
|---|---|---|
| r = T=16 / T=2 on `ffn_gate` | <= 1.6 | **2.47** (420 us vs 170 us) |
| numerics | <= 1e-5 against an fp32 reference | **2.98e-6** (two orders of magnitude better than the half2 path's 5.6e-4) |
| verdict | — | missed 1.6, landed on the "lower-bound line" (r ~ 2) of R2's §5 table |

Projected at r = 2.47, one DFlash2 round (K=7) costs `51 x 2.47 + 16 ~ 142 ms` of verification plus 9 ms of Q4_K_M
drafting = **151 ms for 5.20 tokens = 34 tok/s** on code/reasoning (today: 25, so +38%); prose at 2.07 tokens per round
would be 14 tok/s (-45%, which is what M29's width adaptation has to catch).

## 2. What was built, and why each step

The kernel `k_wide` (`shaders/wide.metal`): each simdgroup owns N output rows with no threadgroup-level barrier; every
128-weight group is decoded through the 16-byte ternary LUT into a **private** fp16 tile (column-major, row stride KC+8 to
break bank conflicts, scale already folded in, exact), and then a single-simdgroup `matmul2d(M=16, N, K=KC)` multiplies it
by the MT token rows.

Shape sweep on real tensors (relative comparison within one cooled-down run):

| shape | time | note |
|---|---|---|
| `k_gemv_lut2` (in use, T=2) | 170 us | baseline |
| M16 N16 K128 G4 | **420 us** | best; 4 sg per core = full occupancy, 17.4 KB |
| M16 N24 K128 G4 | 569 | 26 KB, occupancy unchanged but N=24 is inefficient |
| M16 N32 K128 G2 | 471 | 17.4 KB but only 2 sg per core |
| M16 N32 K128 G1 | 556 | 1 sg per core |
| M16 N16 K256 G2 | 657 | 16.9 KB, 2 sg per core |
| M16 N8 K256 G4 | 686 | two jobs per lane |
| M16 N64 K128 G1 | 425 (MMA only) | large N but only 1 sg per core |

## 3. Five directions that were tried and did not work (with the data, so they are not repeated)

1. **Larger K, 256 or 512**: in the microbenchmark larger K always wins (each `matmul2d` has ~30 ns of fixed overhead),
   but **in the real kernel it is slower** (657 vs 420): doubling the threadgroup memory leaves only 2 simdgroups per core,
   dropping occupancy from 100% to 50%. Occupancy beats call overhead.
2. **Double buffering** (decode the next block while the MMA consumes the current one): 639 us, 52% *slower* than
   single buffering. Doubling the threadgroup memory and changing the barrier structure more than cancels the theoretical
   overlap.
3. **MMA-only ablation**: 306 us (291 G weights/s, 9.4 TFLOPS), independent of shape (N=8/16/32/64 and K=128/256 all land
   in 306-430). **At M=16 the matrix unit ingests at only 53% of peak**, whereas the production prefill GEMM (M=32/64, the
   whole K=5120 in one call) reaches 14.7-16.6 TFLOPS. The difference is the K and M covered per call: a token dimension
   this small cannot amortise anything.
4. **Mixing ALU with the matrix unit** (split the output rows 2:1 by throughput, each path reading its own rows): 473 us,
   between MMA-only (420) and ALU-only (590), with **no overlap gain**. On this chip the ALU and the matrix unit compete
   for issue slots across warps rather than complementing each other.
5. **Feeding B from device memory** (skipping the threadgroup hop): requires expanding the weights, which is 12.8 GB of
   writes per round = a 90 ms floor, more than twice the 40 ms bandwidth floor. Not viable.

## 4. Comparison with an external implementation

On an RTX 5090, ninfer reports one DFlash2 round (T=8) at **1.55x** a single decode step, which maps to roughly r ~ 1.35
against our T=2 baseline. We reached 2.47, i.e. **1.8x slower than a known implementation**. The gap is concentrated in
"the matrix unit at M=16", and everything within our control (tile shape, occupancy, barriers) has been swept; going
further needs information inside MPP (for instance the register-resident B operand path), which is high risk and
uncertain.

## 5. Decision

- Adopt `k_wide_m16_n16_k128_g4` (420 us) as the verification kernel and move on to M26.
- Payoff expectations are computed at r = 2.47: **+38% on code/reasoning, with prose relying on M29's width adaptation to
  fall back to today's behaviour**.
- If it is ever pushed further, the one untried option is "feed the B operand to `matmul2d` from registers"
  (`get_right_input_cooperative_tensor`), which requires re-laying the weights out by lane mapping. It does not contradict
  M12's conclusion (re-layout is useless at T=1) but it is expensive; recorded as an optional experiment.

## 6. Artifacts

`shaders/wide.metal` (`k_wide` parameterised over M/N/KC/NSG, double-buffer and ablation switches, plus the mixed-path
`k_wide_hyb`) and `tools/bench_wide.cpp` (shape sweep on real tensors plus a per-token correctness comparison). Both are
production-path candidates and evidence, not one-off scripts.
