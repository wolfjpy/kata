# M27a: small-grid fixes in the verification window (-20% per round, ceiling raised to +38%)

**English** · [简体中文](M27a-small-grid-fixes.md)

Date: 2026-10-25 · Corresponds to M27a in `CHECKPOINT-004` §5. The goal was to push the "fixed overhead per round" down
from 41 ms so that DFlash2's ceiling rises.

## 1. Result

| configuration | T=2 | T=4 | T=8 |
|---|---|---|---|
| M26 (before) | 156 ms | 159 | 167 |
| **M27a (now)** | **141 ms** | **140** | **142 ms** |

The plain decode baseline in the same period was 50.7 ms/token (throttled state), and the same state measured 177 ms per
round before the fixes, so this is **-20%**, better than the 147 ms target set in CHECKPOINT-004. At 7.2K context T=8 costs
176 ms with split attention against 173 ms with flash — equivalent.

Impact on DFlash2 (K=7, 5.20 tokens per round measured, Q4_K_M drafter at ~9 ms):

| configuration | per round | code/reasoning | prose |
|---|---|---|---|
| M26 | 176 ms | 29.5 tok/s (+18%) | 11.8 |
| **M27a** | **151 ms** | **34.4 tok/s (+38%)** | 13.7 |
| fixed + Q8_0 drafter (16 ms) | 158 ms | 32.9 | 13.1 |

## 2. The three fixes

**(1) alpha/beta: 2 threadgroups -> 96xT.**
It reused the prefill `dGemm(wabAll -> pAB)`, which with Mpad=32 launches only 2 threadgroups, so every layer was pure
latency (6-12 ms per round measured). The new kernel `k_gdn_ab_t` gives one simdgroup one (output row, token): grid
(96, T) = 768 threadgroups, computing dot products directly from the bf16 weights and the fp16 normalised hidden state,
writing `ab[t][0..47] = alpha` and `[48..95] = beta`.

**(2) Attention: flash -> key-split (1-32 splits depending on context length).**
`k_flash_prefill` launches one threadgroup per KV head, so at T=8 with a short context the whole GPU sees only 4
threadgroups (17 ms per round measured, pure latency). Switched to `k_attn_split_b` + `k_attn_combine_b`: grid (S, 4, T)
= 32 threadgroups on a short context and 1024 at 7K. At 7.2K context the two are equivalent (176 vs 173 ms), so short
contexts use the split path and **int8 KV still uses flash** (the split kernels read only the fp16 cache). Both paths were
verified: with T=8 accepting only row 0, the output reproduces greedy decoding token for token (550, 220, 16, 13).

**(3) GDN state: 9 snapshots -> record + replay (saves 1.2 GB and 5 ms).**
M26's `k_gdn_rec_snap` wrote 8 copies of the 147 MB state per round = 1.2 GB (8.4 ms) plus 1.36 GB of VRAM. Now
verification **does not touch the state** (`k_gdn_rec_snap` with `snapStride = 0`), recording each layer's conv/qk/gate
into `vRec` (48 layers x 8 rows x 14332 floats = 22 MB); once the acceptance length is known, `k_gdn_rec_b2` replays the
accepted prefix from the committed state **reading once and writing once** (the state stays in registers across the F
tokens). The replay costs 2 x 151 MB of traffic plus the record reads, about 3 ms — cheaper than writing 8 snapshots, and
1.36 GB of VRAM lighter.

## 3. The per-round breakdown now (T=8, ~142 ms)

| item | time | note |
|---|---|---|
| wide GEMV (FFN 92.5 / GDN 26 / attention 13 / lm_head 6) | ~126 | M25's hard limit (46 GB/s) |
| GDN recurrence + replay | ~4 | record + replay |
| attention (split) | ~3 | was 17 |
| alpha/beta | ~1 | was 6-12 |
| everything else (norm/FWHT/conv/qk/gates/gnorm/argmax) | ~8 | |

**89% of what remains is the wide GEMV**, i.e. M25's matrix-unit ingestion problem. The only way to go faster is to attack
that (r from 2.47 to 1.6 would take the round to ~107 ms and code/reasoning to 48 tok/s).

## 4. Reproduce

```
build/kata --model $M --ids "$(cat /tmp/ids72.txt)" --vwidth 8 --gen 16   # per-round cost + token-for-token greedy
KATA_KV=int8 build/kata ... --vwidth 8                                   # int8 KV takes the flash path
KATA_SKIP=v_ffn|v_gdn|v_ab|v_attn_only                                     # per-item ablation
```
