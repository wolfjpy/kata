# CHECKPOINT-003: what Strata and ninfer taught us, the measurements, the confirmed decisions, the roadmap

**English** · [简体中文](CHECKPOINT-003.md)

Date: 2026-10-25 · A self-contained recovery document: **after a restart, reading this file (and what it points at) is
enough to keep working.**
Predecessors: `CHECKPOINT-001.md` (design stage), `CHECKPOINT-002.md` (state of the implemented engine, file map,
reproduce commands, hypotheses overturned by measurement), `reports/` (one report per milestone; M7 is the drafter
analysis).

## 0. Recovery guide (read this first)

- Repository root: `<repo root>`. Model: `../../models/bonsai-2-27b-mtp-ptq1_0.gguf`. Build: `make`.
- Reference repositories (outside the project, persistent): `../../refs/Strata` (commit `4b9e05a`), `../../refs/ninfer`
  (`64d10fb`), `../../refs/llama.cpp-prism` (`b3ebc47`, with `llama-speculative-simple` additionally built). Drafter
  weights: `../../models/drafters/` (`dflash2-q4km.gguf`, 1.14 GB; `bonsai2-2b-ptq1.gguf`, 0.43 GB).
- If the reference repositories are gone: `git clone https://github.com/Niko1221/Strata` and
  `https://github.com/Neroued/ninfer` into `../../refs/`.
- The owner keeps untracked notes in the repository root: these are saved copies of their own messages, not engine
  output. Do not commit them and do not delete them.
- Working rules (the owner's requirements): reply in Chinese; only commit self-generated files (`docs/ shaders/ src/
  tools/ Makefile VERSIONS config/ third_party/ README.md`), and **never** the PDFs, PNGs or notes the owner leaves in the
  root; commit messages in English; one commit per milestone with its report attached; the latest-version rule (re-check
  upstream versions at the start of every phase and record them in `VERSIONS`); every lossy option must clear the quality
  gate (KL <= 1e-3 nat, top-1 >= 99.7% on a corpus of >= 20K tokens, measured with `kata kl-eval`) and must default to
  off.
- The roadmap in §7 is the current plan and §8 lists the confirmed decisions. The next step is the first unfinished
  milestone in the roadmap.

## 1. Current performance baseline (M6, Mac mini, one minute of sustained GPU load, no throttling)

| metric | value | reference |
|---|---|---|
| decode, plain | 19.8 tok/s | Prism llama.cpp 15.6 |
| decode, MTP depth 1 (greedy / sampled) | 25.0 / 22.7 tok/s | |
| decode, MTP, 7.3K context | 21.4-21.7 tok/s | |
| prefill exact fp16, 512 / 1024 tokens | 245 / 257 tok/s | llama.cpp 208 (`pp512`) |
| prefill exact fp16, 7.3K tokens | 213 tok/s | |
| cold start | ~3 s (first prefill 3.3 s, steady state 0.92 s) | chat mode already warms up on a background thread |

## 2. The two reference repositories

### 2.1 Strata (Niko1221/Strata, MIT, C++/CUDA/HIP/SYCL/**Metal**)

A productised local inference platform (desktop app, terminal, OpenAI/Anthropic API). The engine contains a standalone
Metal implementation (`metal/`), running Qwen3.8-Flash-Next (MoE, Q2_0 37.6 GB) on an **M5 Max 128 GB**. Key documents:
`docs/MAC.md` (all Metal measurements and switches), `docs/DETAILS.md`, `docs/MESSAGE_BOUNDARY_CACHE.md`,
`docs/PROMPT_CACHE_TAIL.md`. Key sources: `metal/kernels/{mma,gdn,gdn_prompt,dense_mm,gemv,units,quant,alu_bench,
gridbar}.metal`, `metal/src/gpu/runtime.cpp`, `src/spec/{suffix_drafter,draft_policy,controller}.cpp`.

Measurements (M5 Max): decode without MTP 87.4 tok/s (llama.cpp 56.6); `--spec 2` + MTP + suffix drafter 119.5/108.2
(prose/story); the suffix drafter contributes about +5% (114 -> 119.5); window sweep: spec 2 is best for prose, spec 3 for
mixed Q&A, >= 4 is worse; prefill about 1430 tok/s (llama.cpp 887).

The parts of its design that matter to us:

1. **The wide verification kernel `mma.metal`**: "a GEMV for a 1-8 token window on the simdgroup matrix unit". Each
   simdgroup owns 8 output rows and, for each 128-wide K block, decodes its own 8x128 weights into a **private**
   threadgroup tile (`wt[SG][8*128]`, no cross-simdgroup barrier), then runs 16 steps of
   `simdgroup_multiply_accumulate` (A = an 8x8 weight block, B = 8 (k) x 8 (token)). The cost of a verification window is
   nearly independent of its size.
2. **GDN verification does not write state**: in `gdn.metal` a verification window with `commit=0` writes nothing; after m
   tokens are accepted, the same kernel (`GDN_NTOK=m, commit=1`) replays them. The `0x100|k` mode commits in place when the
   acceptance count is known.
3. One command buffer per verification window; **the whole MTP draft round in one command buffer**
   (`STRATA_METAL_MTP_CHAIN`).
4. **Suffix drafter** (98 lines): a trigram hash -> the last 4 end positions, take the longest match (<= 32), no model
   needed, microseconds. **`DraftPolicy`**: an online EMA that learns the measured round cost for each window size and the
   acceptance rate in 4 buckets by match length, then picks between the MTP window and the lookup window by "expected
   committed tokens per millisecond" (`E = 1 + q + ... + q^k`, margin 3%), with a cost table bucketed by context length.
5. Prefill: `dq_f16` (expand the whole matrix to half, once per chunk) + MPP `matmul2d` (`gemm_f16<TM,TN,SG>`, with
   `transpose_right` for B), chunk 2048 (4096 is only +2-4%). `gdn_prompt.metal`: three phases (pre in parallel, scan
   state in registers, post).
6. Runtime: every buffer `storageModeShared`, hazard tracking `Untracked`, one `MTLResidencySet` attached to each queue;
   re-encoding a graph costs about 0.17 us per dispatch. `gridbar.metal` is only a feasibility probe; the engine does not
   use persistent kernels.
7. Context cache: conversation files, **message-boundary checkpoints** (taken just before the last user message, so an
   agent editing that message does not have to re-read the whole history), `PROMPT_CACHE_TAIL`.
8. Streaming MoE experts (SSD), n-gram tables, the Gyro format — irrelevant to us.

### 2.2 ninfer (Neroued/ninfer, Apache-2.0, C++/CUDA, RTX 5090 only)

A single-GPU engine written from scratch: Qwen3.5/3.6/3.8 + MTP + DFlash/DFlash2 + vision + prefix cache + scheduler. Its
value is in the documentation (`docs/maintainer/`): `dflash.md` (the complete maths and implementation contract of
DFlash/DFlash2), `replayssm-gdn.md` (GDN replay), `linear-tuning.md` (a methodology for tuning linear layers),
`storage-layouts.md` (split-plane weight layouts), `paged-kv-cache.md`, `engine-architecture.md`,
`resource-scheduling-and-context-cache.md`, `docs/performance/qwen3.8-27b.md` (measurements).

Measurements (RTX 5090, Qwen3.8-27B, FP8 E4M3 row-256 KV, CUDA graphs):
- no speculation: 84.1 tok/s (groupwise-int, 15.9 GB -> 1.34 TB/s = 75% of nominal bandwidth); NVFP4 74.1.
- MTP3: code 193.5 (3.29 tokens/round), story 123.4 (2.10), structured 214.7 (3.65); DFlash2 K=7: code 258.6 (4.75),
  structured 348.9 (6.41), story 115.0 (2.11), short reasoning 305.7 (5.61; MTP3 196.1), long reasoning 177-189 (3.4-3.6;
  MTP3 149-155).
- **One DFlash2 round costs about 1.55x a single decode step** (code: 4.75 tokens / 258.6 tok/s = 18.4 ms against 11.9 ms
  per step).
- Prefill: groupwise-int 3331 tok/s (7.7K prompt, ~160 TFLOPS = 77% of the BF16 peak); NVFP4 12.8K tok/s (FP4 tensor
  cores).
- Linear-layer tuning methodology: the "hot range" T=1..128, priority T=1 > T=4,8 > the rest; measure every T; use the
  "sustained read ceiling" (1674.5 GB/s, measured) as the reference.

Design points:

1. **DFlash2**: see §4.
2. **ReplaySSM**: see §5.
3. **Split-plane weight layout `row_split_k128_v1`** (Q4/Q5/Q6/Q8): the code plane, the high-bit plane and the scale plane
   are separate, each 256-byte aligned, ordered within a plane by "row 0 group 0, row 0 group 1, ...". Read with 16-byte
   vectors. Prism's CUDA backend also has a planar PTQ1_0 (but it reorders the **activations**: Q8_1 -> eight 16-byte
   planes plus one scale plane), because CUDA uses dp4a; Apple GPUs have no dp4a.
4. **KV formats**: BF16, INT8 (group-64), FP8 E4M3 (row-256, i.e. one scale per head per token), NVFP4, K8V4; with a
   perplexity harness (`docs/perplexity.md`). Qwen3.8 defaults to FP8 row-256.
5. **Optimised draft head `--lm-head-draft`**: the 131072 rows ranked by token frequency (53% of the 248320 vocabulary),
   used only for proposing drafts; verification still uses the full head.
6. **Prefix / context cache**: "a reusable checkpoint = the KV plus the complete continuation state at that token
   boundary (GDN state and so on)"; device and host capacities shared, retained resources reclaimed under pressure, and a
   suspended request resumes from a snapshot or by replay.
7. CUDA graphs with exact B topology; the speculative unit is split into a Forward and a Finish graph, and the CPU prepares
   the constraint mask between them.

## 3. Measurements on drafting and verification from this round (M7 has more)

### 3.1 Acceptance (Prism `llama-speculative-simple`, Bonsai-2-27B as the target, greedy)

| drafter | prose | code | reasoning (thinking) |
|---|---|---|---|
| MTP depth 1 | 1.6 (our engine) | — | 1.86 |
| MTP depth 3 | 1.87 | 3.2 | 3.13 |
| DFlash2 (Q4_K_M, 7 drafts) | 2.2 | 5.1 | 5.1 |
| Kata-2 2B (an ordinary draft model, 4 drafts) | 1.23 | 2.07 | — |

(Units: tokens produced per verification round.) These agree with ninfer's numbers on Qwen3.8, which says Kata transfers
well as a DFlash2 target.

### 3.2 The cost of wide verification (`tools/bench_gemv2`, `ffn_gate` 5120->17408, per matrix)

LUT + half2 GEMV: T=2 170-185 us, T=4 296, T=6 397, T=8 590-598 (3.2x). The matrix-unit prototype (`k_wide_gemm`: an fp16
threadgroup block + `matmul2d`, M=16, 4 simdgroups sharing one 20 KB threadgroup block, threadgroup-level barriers) takes
831 us: numerically correct (max error 3e-6) but too slow, because only one threadgroup can reside per core and each K
block serialises "load -> decode -> barrier -> matmul -> barrier". **Strata's structure (a private tile per simdgroup) is
exactly the fix for that.**

### 3.3 Throughput of the matrix paths (M6, new this round, probes in `docs/probes/`)

| path | throughput |
|---|---|
| classic `simdgroup_matrix` 8x8x8 (half/float alike, goes through the ALU) | **5.2 TFLOPS** |
| `matmul2d` half x half -> float | 15.7-16.0 TFLOPS (64x128 / 4 SG) |
| `matmul2d` half x int8 -> float / half | 14.3-15.1 TFLOPS (**about fp16, no int8 speedup**) |
| `matmul2d` int8 x int8 -> int | 33.8 TOPS |

Meaning: Strata's `simdgroup_matrix` approach is not fast enough if copied directly onto the M6 (T=8 would spend ~80 ms in
the MMA alone); what is needed is "private tile per simdgroup + `matmul2d`", and an M=8 tile (otherwise T <= 16 is billed
as 16 tokens). The T=8 MMA at 12.6 TFLOPS is about 33 ms, overlapping the ~40 ms bandwidth floor, so verification should
land at ~55-70 ms.

### 3.4 Runtime options (measured in a throwaway copy; no steady-state gain)

Encoding one decode step on the CPU costs 0.7 ms (1.3%); `MTLResourceHazardTrackingModeUntracked` and residency sets are
inside the noise at steady state; a residency set brings the first prefill down to 0.92 s but the process total stays at
~3.9 s (the same ~2.4 s simply moved into the load phase, where `requestResidency()` blocks on faulting in 6.4 GB).
**Cold start is the OS faulting in a 6.4 GB mapping: it can be hidden, not removed.**
(`docs/probes/residency_untracked_experiment.diff`)

### 3.5 The int8 prefill Pareto front (24.5K-token corpus, 1536 positions, `kl-eval`)

| configuration | mean KL | max KL | top-1 |
|---|---|---|---|
| all int8 (per-group activation scale) | 4.4e-4 | 0.34 | 99.54% |
| **FFN gate+up only (~47% of FLOPs)** | **6.1e-5** | 9.1e-3 | **99.80%** |
| gate+up + qkv + z + wq | 5.9e-4 | 0.76 | 99.67% |

The error comes mainly from the input projections that feed the GDN recurrent state (qkv/z/wq); the mean KL is dominated
by a few outlier positions, and top-1 is the more stable metric. The int8 GEMM measures only ~21.5 TOPS (epilogue
overhead), so "gate+up only" is worth about 8-10% prefill.

## 4. DFlash2 specification summary (for a possible port; original in `refs/ninfer/docs/maintainer/dflash.md`)

- Draft configuration (official Qwen3.8-27B): H=5120, 5 layers, FFN 17408, 32/8 Q/KV heads, head_dim 128, all local
  attention with sliding window S=2048, RoPE theta 1e7, **target layer-block IDs `[5,19,33,47,61]`**, a two-tap dynamic
  grouped convolution (group width 16), and a candidate selector with top-16 and rank 256. 1.92B parameters (3.85 GB in
  BF16; 1.14 GB as GGUF Q4_K_M, about 2 GB as Q8_0).
- Conditioning: for every processed position t, take the **full residual** of the target's layer-block output `r_t^l`
  (after the mixer and FFN, before the next layer's input norm), concatenate `s_t = concat(r_t^l)`, and
  `c_t = plain_rmsnorm(W_feature s_t, context_norm)`; each draft layer projects `c_t` into its own context K/V
  (`k_ctx = rope(plain_head_rmsnorm(W_ck c_t))`, `v_ctx = W_cv c_t`). **Context prefill only does the feature projection
  and the K/V projections — prompt tokens never go through the drafter's residual/MLP stack.** Only the last 2048 positions
  need to be kept (a ring KV, 5 layers x 40 MiB in total).
- One round: committed frontier F, anchor token `a`; draft input = `[a, MASK x K]` (positions F..F+K, K=7 -> block width
  8); non-causal attention (context plus within-block), local layers with `|p_key - p_query| < S`; one forward produces K
  proposals. Head: per-column top-16 candidates -> the DFlash2 selector
  `E_i[p,c] = u_i[c] + sum_r W_pred[pred,r] * g_i[r] * W_succ[C_i[c],r]` doing a conditional path walk (sampled from the
  softmax when temperature > 0, keeping `q_i`).
- Verification: the target processes `[a, d_1..d_P]` in one pass; greedy takes the matching prefix plus a correction/bonus
  token; sampled verification accepts with `min(1, p/q)` and resamples from `max(p-q, 0)` on rejection, which preserves the
  target distribution.
- For every committed position, its features become "context to materialise", projected only before the next draft.
- The drafter's 5 layers are Qwen3-style; `dynamic_grouped_conv`, `candidate_selector`, `context_kv_materialize` and
  `sliding_window_attention` have exact numerical contracts in `refs/ninfer/include/ninfer/ops/*.h`.

## 5. ReplaySSM summary (original in `refs/ninfer/docs/maintainer/replayssm-gdn.md`)

Verification starts from the committed checkpoint `S_0`, **produces outputs and a raw transition record `R_t` per token,
but writes no state**; once m tokens are accepted, `R_1..R_m` are replayed in order from `S_0` to obtain `S_m`. For 27B:
one state is 144 MiB (ours is 151 MB) plus 2.8 MiB of conv history; the per-token record is 1.705 MiB (raw k/v plus g/beta)
— 1/86 of a snapshot. The key constraint: **replay must perform exactly the same finite-precision transitions as
verification** (same normalisation, gating, reduction order and storage boundaries), or the state drifts across rounds. We
currently write a full state per position at T=2 (triple-buffered), which is unacceptable at T=8 (1.2 GB per round).

## 6. Conclusions distilled from this round (each backed by data)

1. Decode GEMV: T=1 is bandwidth-bound; it is T>=2 verification that is ALU-bound (corrected in CHECKPOINT-002 §4.1).
2. ~~The remaining 8-10% at T=1 comes from the read pattern~~ (**corrected by M12**): in a pure read probe the native
   small reads reach 125-132 GB/s and 16-byte vector reads 141-142 GB/s, but once the decode compute is added inside the
   GEMV, the staged wide-read prototype runs at the same speed as the original (125-135 vs 127-134 GB/s). Read width is
   not the limiting factor.
3. The exact prefill GEMM is at ~96% of peak; the remaining non-GEMM overhead (0.56 s of expansion plus 0.4 s of other
   work) totals about 1 s, and removing all of it would be worth only +28%.
4. `half x int8` gets no int8 speedup on the M6; the only matrix mode faster than fp16 is int8 x int8, and that needs
   activation quantisation.
5. Cold start cannot be removed (page faults); it can be hidden (background warm-up, or a residency set requested on a
   background thread).
6. Throttling: from cold, one minute of sustained load (14 runs of the 1024-token prefill) stays flat; throttling appears
   after several minutes and on the power-hungrier pure-matmul probes.
7. With the current ALU kernels, depth-1 MTP is the optimum; deeper speculation needs a cheap wide verification first.

## 7. Roadmap (milestones; each one committed with its report)

Every milestone follows: re-check upstream versions; write a data-driven acceptance criterion; implement; validate the
numerics with `verify-check` / `kl-eval` / `--cmp-logits`; measure (interleaved A/B, cooled down); write
`reports/Mxx-*.md`; update the status table in `README.md` and the CHECKPOINT; commit.

| milestone | content | acceptance | expected payoff |
|---|---|---|---|
| **M8** | this document (the learning and decisions summary) | — | — |
| **M9** done (`reports/M9-prefix-cache.md`) | **persistent prefix cache**: write the GDN state, conv state and KV (including the MTP K/V) to SSD at user-message boundaries, indexed by a token-prefix hash; reused by a new process or session on long prompts; LRU plus a disk budget | the restored logits match a from-scratch prefill bit for bit (or KL <= 1e-6); first token of a 20K-token prompt ~80 s -> <= 5 s | 10x+ first-token in agent scenarios |
| **M10** deferred (`reports/M10-sampling-aware-draft.md`: the offline estimate says +5 points, gate not met) | **sampling-aware drafts**: when temperature > 0, sample the draft from the MTP distribution q and verify with `min(1,p/q)` plus residual resampling; estimate `sum min(p,q)` offline first | sampled acceptance 49% -> >= 58%, distribution unchanged (frequency test over many samples) | ~+8% in sampled mode |
| **M11** done (`reports/M11-int8-kv.md`; note the later correction to 99.51% top-1) | **int8 KV** (one scale per head-row; native `half x char` matmul2d; the K scale multiplies the columns of S, the V scale those of P) | KL <= 1e-3 and top-1 >= 99.7% on a 24K corpus; 32K decode +8-10%; half the memory | long-context decode, 64K memory |
| **M12** dropped (`reports/M12-weight-layout.md`: the staged wide-read prototype gained nothing, gate not met) | **split-plane weight re-layout** (`kata-pack` producing a planar file; GEMV/expansion/embedding read 16-byte vectors) | bit-identical numerics; plain decode 19.8 -> >= 21.2 tok/s | decode +8-10% |
| **M13** done (`reports/M13-fast-ffn.md`: KL 7.5e-5, top-1 99.87%, prefill +7-9.5%) | **int8 prefill switch for the FFN gate+up** (`--fast-ffn`, lossy, off by default) | top-1 >= 99.7% on a 24.5K corpus (measured 99.80%), prefill +8-10% | prefill +8-10% |
| **M14** verdict overturned (`reports/R2-dflash2-revisit.md`) | **wide verification microbenchmark gate**: `matmul2d` with a private tile per simdgroup, M=8, weights LUT-decoded into the private tile; measure T=4/8 | passed only if **T=8 is <= 1.5x T=2 (~260 us at ffn_gate)**; otherwise stop DFlash2 and close it in the report | gate |
| **M15-M16** rescheduled (see M25-M29 in `reports/R2-dflash2-revisit.md`) | ReplaySSM replay + wiring wide verification into the engine + sampling compatibility | verify-check: KL <= 1e-5; a T=4/8 verification round <= 1.5x T=2 | prerequisite for DFlash2 |
| DFlash2 port (Q8_0 or Q4_K_M drafter weights + wide GEMV; context materialiser, dynamic convolution, selector, sliding-window attention) | acceptance length matching llama.cpp `draft-dflash` (code ~5.1); lossless; code/reasoning >= 50 tok/s | 2x decode on code/reasoning |
| **M17** dropped (`reports/M17-small-items.md`: QoS and NEON sampling had no measurable effect; single command buffer ~1.7%, vocabulary subset ~1.4%, and the suffix drafter is pointless without wide verification — all deferred) | small items: suffix drafter + online cost policy (only meaningful once wide verification exists); draft and verify in one command buffer; draft vocabulary subset (131K rows) | keep only what is worth >= 1% | <= 5% |

Ordering rationale: M9-M13 are data-backed, low-risk gains; M14 is the decision gate for DFlash2, placed after M13 but
moved earlier if the earlier milestones stall.

## 8. Confirmed decisions (owner, 2026-10-25)

1. **Build the prefix cache** (M9). The context coverage defaults were mine to choose: up to the engine's context limit; a
   disk budget of 40 GB (256 GB SSD), LRU eviction; each checkpoint = 151 MB of GDN state + 6 MB of conv + 64 KB x tokens
   of KV (1.3 GB at 20K, 2.1 GB at 32K, 4.2 GB at 64K); all configurable.
2. **DFlash2 is a multi-day, high-risk project, gated behind the wide verification microbenchmark** (M14 -> M15 -> M16).
3. **Add the lossy FFN gate+up int8 switch** (M13), off by default, with the quality data documented.
4. The owner's general instruction: build per plan unless a question comes up; checkpoint every milestone and commit
   following git best practice (small focused commits, English commit messages with a body, one report per milestone,
   never commit the owner's files).

## 9. Risks and cautions

- The GPU throttles after several minutes of sustained load: timing must be interleaved A/B, repeated, and cooled for 1-2
  minutes when in doubt.
- A 72-position test hides error accumulation: quantisation changes must be evaluated on a corpus of >= 20K tokens.
- The correctness object of the prefix cache is "the complete continuation state"; having the KV pages is not a hit — the
  GDN state, conv state, MTP K/V and `hPrev` must be restored too. The chat template strips older thinking content on
  later turns, so checkpoints belong at **user-message boundaries** and the key must include the rendered token sequence.
- 64K context already sits near 12.7 GB resident; int8 KV (M11) relieves that.
- DFlash2 verification inherits ReplaySSM's constraint that replay must match verification exactly in finite precision, so
  the current `k_gdn_rec_v2` and the replay kernel must share the same arithmetic.

## 10. Quick reference

| topic | location |
|---|---|
| Strata Metal wide verification kernel | `refs/Strata/metal/kernels/mma.metal` |
| Strata GDN verification/replay | `refs/Strata/metal/kernels/gdn.metal`, `gdn_prompt.metal` |
| Strata suffix drafter / policy | `refs/Strata/src/spec/suffix_drafter.cpp`, `draft_policy.cpp`, `include/strata/spec/*.hpp` |
| Strata Metal runtime | `refs/Strata/metal/src/gpu/runtime.cpp` |
| Strata message-boundary cache | `refs/Strata/docs/MESSAGE_BOUNDARY_CACHE.md` |
| ninfer DFlash/DFlash2 | `refs/ninfer/docs/maintainer/dflash.md`, `refs/ninfer/include/ninfer/ops/*.h`, `refs/ninfer/src/models/qwen3_5/execution/draft.cpp` |
| ninfer ReplaySSM | `refs/ninfer/docs/maintainer/replayssm-gdn.md`, `include/ninfer/ops/gdn_replay.h` |
| ninfer weight layout / tuning | `refs/ninfer/docs/maintainer/storage-layouts.md`, `linear-tuning.md` |
| ninfer context cache | `refs/ninfer/docs/maintainer/resource-scheduling-and-context-cache.md`, `paged-kv-cache.md` |
| probes from this round | `docs/probes/simdgroup_mma_probe.*`, `matmul2d_mixed_type_probe.cpp`, `residency_untracked_experiment.diff`; `tools/{weight_read_probe,kv_read_probe,bench_gemv2,probe_flash,alu_probe,layout_probe}.cpp` |
| draft acceptance logs | `/tmp/ss_*.log` (temporary); summarised in `reports/M7-speculation-options.md` |

## 10.4 DFlash2 reopened (2026-10-25)

The "closed" verdict of `reports/M14` was overturned: it compared against T=2 instead of the bandwidth floor, whereas a
T=8 verification is in fact still bandwidth-bound (~40 ms), and the prototype kernel was flawed on K=128, M=8, N=16 and
the serialisation of decode and MMA. Today's acceptance curve (K=1/3/5/7 -> 1.92/3.24/4.68/5.20 tokens per round) shows
the payoff growing monotonically with width. The reopening plan is in **`reports/R2-dflash2-revisit.md`** (M25 wide
verification kernel -> M26 complete the verification side -> M27 MTP chain -> M28 DFlash2 port -> M29 width adaptation).
M25 is done (`reports/M25-wide-verify-kernel.md`, r=2.47 against a 1.6 target) and M26 is done
(`reports/M26-verification-side.md`): verification at arbitrary width plus 9 state snapshots and commit, with `--vwidth 8`
reproducing plain greedy token for token at 167 ms per round, almost independent of T.

## 10.5 Follow-up (2026-10-25): repository assessment and the next phase

The full read-through assessment, the three things newly measured that day (the combined lossy switches land at 99.58%
top-1 and fail, a one-shot process necessarily pays 3.3 s of weight page-in, and the three forward paths disagree
arithmetically), and the M18-M24 plan for the next phase are in **`reports/REVIEW-01-repo-assessment.md`**. That document
is now the source of the plan; this one preserves the results of M8-M17.

## 11. Final outcome of the roadmap (2026-10-25)

| milestone | result |
|---|---|
| M8 | done — this document |
| M9 | done — persistent prefix cache: a 2.9K-token system prompt goes from 15 s to 1.15 s (first chat turn); a repeated 3.6K prompt from 18.6 s to 0.03 s; output byte-identical |
| M10 | deferred — sampling-aware drafts: the offline estimate gains only 5 points against a gate of 9 (optional, +2-3%) |
| M11 | done — `--kv int8`: KL 1.3e-4, top-1 99.71%, needle 3/3; half the memory; 29K-context decode +14%, prefill +10%; no speed gain at short context; off by default. (Quality later corrected to 99.51% top-1; see the report.) |
| M12 | dropped — weight layout: the staged wide-read prototype gained nothing (GEMV is not limited by read width) |
| M13 | done — `--fast-ffn`: prefill +7-9.5%, KL 7.5e-5, top-1 99.87% (passes the gate); off by default |
| M14 | failed — wide verification gate: T=8 is 2.2-2.3x T=2 against a 1.5x gate, DFlash2 closed. (Later overturned by R2.) |
| M15, M16 | cancelled (M14 did not pass) |
| M17 | dropped — every small item fell below the 1% keep line |

### Where the numbers stand now (deltas against CHECKPOINT-002)
- Decode: plain 19.8, MTP 25.0 (greedy) / 22.7 (sampled) tok/s; 29K context 13.7 tok/s, 15.6 with `--kv int8`.
- Prefill: exact 255-263 tok/s (1024 chunk), 245 (512), 207-213 (7.3K); `--fast-ffn` 274-280 (+7-8%), 226.5 at 7.3K
  (+9.5%); `--fast` (all int8, fails the gate) 281.
- Agent / chat: the same long system prompt is restored from the second run on, taking the first token from ten-odd
  seconds to about 1 s.

### Files added this round
`src/ckpt.{h,cpp}` (prefix cache), `shaders/wide.metal`, `tools/{bench_wide,test_cache.sh}`, `reports/M9-M14,M17`;
`kl-eval`'s `--save-ref/--ref/--fast-ffn/--i8`; `kata cache`, `kata mtp-accept`; and the switches `--kv int8`,
`--fast-ffn`, `--i8 list`, `--no-cache/--cache-dir/--cache-gb/--cache-min`.

### Directions still open (by expected payoff)
1. The remaining decode time: verification GEMV at 113 GB/s (~51 ms) and drafting (~6 ms). M12 showed read width is not
   the limit; untried is halving the ALU instruction count at T=1/T=2 with a LUT + half2 that changes the read/decode
   overlap at the same time.
2. Prefill non-GEMM overhead of about 1 s per 1024 tokens (0.56 s of expansion plus 0.4 s of other work): fuse the
   dequantisation into the GEMM (a large change).
3. Sampling-aware drafts (+2-3% in sampled mode), draft and verify in one command buffer (+1.7%).
4. Segmented, chained prefix-cache storage (removing the ~2x redundancy between the "message boundary" and "end of prompt"
   checkpoints).
5. The one variant not tried in M14: feeding the B operand from registers (`get_right_input_cooperative_tensor`).
