# REVIEW-01: a read-through assessment of the repository, the open issues, and the plan

**English** · [简体中文](REVIEW-01-repo-assessment.md)

Date: 2026-10-25 · Discussion only, no code changed. Companion reading: `CHECKPOINT-003.md` (design, decisions,
milestone results), `reports/M0-M17`, `MODEL_SPEC.md`.

## 0. How this document came about

I re-read all 21 documents, the 11 source files in `src/`, the 14 MSL files in `shaders/` and the 8 programs in `tools/`,
and checked every key number in the reports against the code; **for any number I would use to make a decision**, I either
re-measured it on today's machine or traced it to the code that produces it.

Three things measured today changed my judgement (§3.4): combined lossy switches fail the quality gate, a one-shot
process necessarily pays 3.3 s of weight page-in, and the arithmetic of the different forward paths does not agree. Those,
plus the existing performance account, produce the plan in §4 and the milestones in §5.

## 1. The repository at a glance

### 1.1 Code (~3,750 lines of our own code plus vendored metal-cpp)

| file | lines | role |
|---|---|---|
| `src/engine.cpp` | 517 | **four forward paths**: `step` (single token), `prefill` (batched), `verify2` (2-token verification), `mtpRows` (the MTP head); plus `feed`/`encMtpKv`/`warmup`/`dKvQuant` |
| `src/engine.h` | 117 | buffer handles, per-layer weight offsets, the public API, raw-pointer accessors for the prefix cache |
| `src/cli.cpp` | 257 | `run`/`chat`/`tokenize`/`cache`/`mtp-eval`/`verify-check`/`kl-eval` |
| `src/ckpt.cpp` | 166 | prefix-cache format, restore, async save, LRU |
| `src/main.cpp` | 107 | the development entry point (per-layer comparison, `--gen`, `--verify`, `--skip` ablation) |
| `src/spec.cpp` | 47 | the MTP speculative main loop (exact speculative sampling) |
| `src/tokenizer.cpp` + `pretok.mm` | 128 | BPE + ICU regex (the only Objective-C++ file) |
| `src/chat.cpp` | 69 | chat template + sampler |
| `src/gguf.cpp` / `gpu.cpp` | 124 | GGUF parsing (whole file mmapped, zero copy) / the Metal context |
| `shaders/*.metal` | 1,484 | ~1,050 of which are production; `gemv_exp` (299), `layout_probe` (59) and `wide` (137) are experimental probes |
| `tools/*` | 336 | reference-runtime wrappers, six microbenchmarks, the cache regression script |

The style is consistent: MSL and the host share `config/kata_m6.h` constexprs; the GGUF metadata is validated at load; all
buffers are allocated once in `load()`; there is no dynamic allocation; a full `make` takes 6 seconds.

### 1.2 Documentation (21 documents, in three layers)

- **Authoritative layer**: `CHECKPOINT-003.md` (current state, decisions, overturned assumptions, roadmap results),
  `MODEL_SPEC.md` (the maths), `reports/M9-M17` (this round).
- **Historical layer**: `CHECKPOINT-001.md` (1209 lines of design-stage record, including all the hardware probing and
  upstream research), `CHECKPOINT-002.md`, `reports/M0-M7`.
- **Evidence layer**: `docs/probes/` (28 probe source files), `docs/data/` (GGUF header dumps), `docs/reference/`
  (Prism/HF reference code).

The historical layer has **two misleading places** (they do not affect correctness, but they affect trust):
1. `reports/M3` says "the GEMV is not ALU-limited; it needs neither a re-layout nor a LUT". That is true for T=1, but M4/M12
   show that T>=2 *is* ALU-limited and that a re-layout genuinely does not help (for a different reason). The conclusion
   held, the argument was later replaced.
2. `reports/M5`'s 278 tok/s int8 and `CHECKPOINT-002` §7's prefill degradation table (3639 -> 201, 7279 -> 176) are both
   states from before M5b; today's numbers are 255-263 and 226 (`--fast-ffn`). Anyone reading the historical reports gets
   the old conclusions.
Recommendation: put a one-line "superseded by X" banner at the top of M3, M5 and CHECKPOINT-002. One line of work, a
significant gain in reading experience.

## 2. Where it stands

### 2.1 What is right (and what backs it)

| judgement | evidence |
|---|---|
| zero copy plus a single resident weight copy is correct | 5.657 GB/token, matching the theoretical floor; there is no second copy of the weights |
| the PTQ1_0 decode is the right choice | T=1 GEMV runs at 126-134 GB/s (89-94% of sustained bandwidth), essentially the same as the "read without decoding" native-layout ceiling (124-132) -> nothing is wasted |
| decode has already captured ~80% of its roofline | 50.6 ms x 142 GB/s = 7.19 GB of "equivalent reads", against 5.657 GB of weights + 0.3 GB of state = 5.96 GB -> 83% |
| the prefill GEMM is at the matrix unit's peak | 16.6 TFLOPS measured on a 1024-token chunk (peak 17.3, so 96%); the non-GEMM part is only 1.0 s |
| the exact path's quality is fine | KL 1.4e-7 over 72 positions, top-1 72/72 |
| the speculative decoding is exact | deterministic draft plus `p(argmax q)` acceptance, distribution unchanged; `verify-check` gives top-1 100% over 830 positions |
| the int8 conclusions have boundaries | all-int8 fails -> FFN only (99.87%) -> +14% at 29K context |
| every conclusion records how to falsify it | the probes, the ablation switches, `kl-eval` and `verify-check` can all be re-run |

### 2.2 Technical debt in the code, by impact

1. **The layer loop is implemented four times.** `step` and `verify2` are two copies of the same structure (~60 lines
   each), `prefill` is a third variant and `mtpRows` a fourth. Any change (an fp16 state, ReplaySSM, a new attention
   variant) has to be made in 3-4 places, and missing one produces a silent numerical inconsistency — which is exactly the
   class of accident that already happened this round (`main.cpp` set `int8Mode` but not the mask, so all three modes ran
   the exact path and both the timings and the conclusions were briefly wrong).
2. **Mode selection is scattered through the hot path**: `getenv()` (`KATA_SCALAR_DEC`, `KATA_I8_MODE`, `KATA_OLD_EXPAND`,
   `KATA_DEC_CHUNK`, `KATA_GM`, `KATA_KV`) is read on every encode (static-cached, so the cost is negligible), but those
   switches **are not in `numericsTag()`** — only the int8 set and `+kv8` made it in. In other words there are several
   routes between the environment and the code that can change the numerics without invalidating the cache.
   `KATA_SCALAR_*` is currently blocked by `!kvInt8`, but that is a coincidence, not a design.
3. **`sk()` does ablations with string comparisons**: about 600 calls per token, each a linear scan of a
   `vector<string>`. The cost is negligible (~6 us), but putting a profiling-only mechanism into the production encoding
   path is pollution.
4. **There is no reproducible numeric baseline.** `/tmp/ids72.txt`, `/tmp/golden72.bin` and `/tmp/ref128.bin` are not in the
   repository, so once `/tmp` is cleared every "KL 1.4e-7 / top-1 72/72" acceptance criterion becomes impossible to re-run.
   The only automated test in the repository is `tools/test_cache.sh` (11 checks).
5. **`engine.cpp` is a 517-line single file**, and `prefill` mixes encoding, planning and logging. There is no intermediate
   representation and no "one layer = one function" boundary.
6. Small inconsistencies: the `Makefile` uses `-std=c++20` while the docs say C++17; and the metal-cpp commit in `VERSIONS`
   cannot be verified from the vendored copy (no `.git` there).

### 2.3 Feature gaps, directly relevant to the stated purpose

The owner's goal is "chat and agent use", and three things are missing:
1. **No resident service**: every `run` is a new process, which **necessarily** pays 3.3 s of weight page-in (measured
   today: three consecutive processes at 3.55 / 3.33 / 3.40 s; the file pages are not kept in the unified buffer cache).
   If an agent starts a process per step, that 3.3 s is pure overhead.
2. **No tool-call template**: `chat.cpp` renders only system/user/assistant, with no `<tool_call>`/`<tool_response>`
   sections and no argument JSON parsing. An agent scenario has to render it itself with `--raw`.
3. **No stop strings, no context rolling**: only EOG stops the generation; when the context fills up it errors out instead
   of dropping the oldest non-system turns.

### 2.4 Details confirmed while reading (not bugs, but worth knowing)

- `resetState()` clears only the current GDN state; the other two (triple buffering) keep garbage from last time, which is
  safe because `verify2` always writes before reading. That invariant exists only outside the comments; it should be
  written down.
- A cache checkpoint written with `--no-spec` contains neither the MTP K/V nor `hPrev`, so `--no-spec` and default
  (speculation on) checkpoints are **not interchangeable**. The MTP K/V could in fact be rebuilt afterwards with
  `encMtpKv` (three matrix GEMMs, about 0.05 s for 3K tokens).
- Cache integrity relies only on `pread` return values and a token-sequence comparison, with no checksum: a truncated file
  is caught, a silent bit flip is not.
- `verify2`'s MTP cache row is left dirty when a draft is rejected, but the next cycle always overwrites it before
  reading — correct, but again only in my memory.

## 3. The performance account, re-checked today

### 3.1 Decode (64-token context, MTP off, three interleaved rounds, ms/token)

| round | baseline | without `gemv_gate` | without `k_gdn_step` | without `k_attn_split` |
|---|---|---|---|---|
| 1 | 50.6 (three consecutive runs all 50.6) | -9.6 | -2.8 | -0.1 |
| 2 | 51.4 / 52.5 / 52.6 | -11.2 | -2.8 | -0.3 |
| 3 | 52.3 / 52.6 / 52.9 | -10.1 | -3.4 | -0.6 |

Breakdown (using the 50.6 ms cold-machine value):

| item | time | floor | note |
|---|---|---|---|
| weight reads (5.657 GB) | ~45 ms (GEMV measured at 126 GB/s) | 39.8 ms (142 GB/s) | the remaining 3.5-5 ms is ALU/latency/tails |
| GDN state reads and writes (302 MB) | ~3 ms | 2.1 ms | pure bandwidth; only compression helps |
| everything else (norm/FWHT/attention/conv/argmax) | ~2.5 ms | — | the serial part of the dependency chain |
| **total** | **50.6 ms = 19.8 tok/s** | **~42 ms = 24 tok/s** | realistically ~46 ms (21.7 tok/s) |

**One wrong expectation to correct**: running the 19 ablations sequentially, the items summed to 58 ms against a 53 ms
baseline. That is not a mistake in the arithmetic — it is the GPU throttling by ~10% after a minute of sustained load. Any
single ablation needs an adjacent baseline, which is also why two "apparently effective" optimisations were rejected in
M17: interleaved, there was no difference.

The MTP cycle: verification 58.4 ms (5.657 GB -> **97 GB/s**, lower than decode's 112 because the T=2 LUT kernel is ALU-bound
at 113 GB/s, plus 1.2 ms for the doubled state write and the attention) + drafting 5.7 ms (729 MB at 128 GB/s) = 64.1 ms for
1.60-1.66 tokens -> 25 tok/s.
-> **If the T=2 GEMV reached T=1's efficiency (126 GB/s), the cycle would drop to ~55 ms, i.e. 29 tok/s.** T=2 is currently
6 instructions per byte (the half2 LUT), which is the ALU ceiling, and M12 showed read width is not the bottleneck. So this
route ends here.

### 3.2 Prefill (1024-token chunk, two interleaved rounds today)

| configuration | total time | of which GEMM | GEMM efficiency |
|---|---|---|---|
| exact fp16 | 4.48 / 4.39 s | 3.46 / 3.39 s | 14.7 TFLOPS |
| `--fast-ffn` | 3.99 / 3.87 s | 3.10 / 3.01 s | — |

### 3.3 Long context

| context | fp16 KV | int8 KV |
|---|---|---|
| 7.3K, decode (plain / MTP) | 17.5-18.0 / 21.4-21.5 | 18.1 / 21.3-21.5 |
| 29K, decode (plain) | 13.7 | **15.6 (+14%)** |
| 29K, prefill | 162 | 178 (+10%) |

The conclusion is unchanged: KV compression only has a speed meaning at >= ~16K context; elsewhere it is about memory.

### 3.4 Three problems found today

**(a) Lossy switches cannot be combined freely.** Each passes the gate alone, the combination does not:

| configuration (against fp16 weights + fp16 KV, 3072 positions) | mean KL | top-1 |
|---|---|---|
| `--fast-ffn` | 7.5e-5 | **99.87%** (3068/3072) |
| `--kv int8` | 1.3e-4 | **99.71%** (3063/3072) |
| **both at once** | 2.0e-4 | **99.58%** (3059/3072) FAIL |

The errors add roughly linearly and the gate is 99.7%, so any combination of two lossy switches must be evaluated
separately, and the CLI should either forbid it or warn when both are on. This is a quality problem rather than a speed
problem, but it decides whether `--fast-ffn` can become the default.

**(b) A one-shot process always pays 3.3 s.** Across three consecutive processes, the first prefill of `run --no-cache` is
always 3.3-3.6 s and the file pages are not retained — meaning **every agent step pays the weight page-in again**. The
cache saves compute, not this 3.3 s (measured: a full-prompt hit restores in 0.03 s, but the first decode takes ~2.7 s
longer to page the weights in).

**(c) The arithmetic of the different paths does not agree.** Quantified (72 positions):

| comparison | mean KL |
|---|---|
| prefill (fp16 activations) vs per-token (fp32 maths) | 1.4e-7 (max 7.4e-7) |
| verification path (half2 accumulation) vs per-token | 2.5e-7 |

Both tiny, but at a near-tie they flip the greedy output (seen at character 690 of a story). **"The same prompt gives
different answers depending on length or path"** is user-visible inconsistency, and the cache, `--kv` and `--fast-ffn` all
change which path is taken.

## 4. Improvement plan, ordered by value over risk

Each item states its goal, approach, expectation, acceptance and risk. Items marked with a star are the ones I recommend.

### * A. Unify the numeric paths and build a reproducible test baseline (debt, zero speed cost)

- **Approach**: (1) switch `step` to fp16 activations (the same precision as `prefill`), using the already-measured
  same-speed `k_gemv_v1h_t1` (154 us vs 148, inside the noise); (2) extract the layer encoding into
  `encodeLayer(e, il, Mode)` (`Mode` = {decode, verify2, prefill-tok}) shared by `step`/`verify2`/`prefill`; (3) add
  `tools/make_golden.sh` (generating the baseline with `ref_logits`) and `tools/test_all.sh` running: golden KL,
  `verify-check`, the cache regression, tokenizer agreement, and cross-path greedy agreement (20 prompts x 3 paths).
- **Expectation**: no speed change; the "different path, different answer" class disappears; every future change gets a
  machine-runnable regression.
- **Acceptance**: `make test` green; cross-path greedy agreement 20/20; KL between `step` and `prefill` <= 1e-6.
- **Risk**: low. This is the precondition for everything else.

### * B. A quality baseline: task-level evaluation (decides whether lossy switches can be default)

- **Approach**: fix a task set (20 multi-step arithmetic/reasoning problems, 8 code completions, 20 long continuations),
  decode greedily, and run it under exact / `--fast-ffn` / `--kv int8` / the combination, comparing **final answers** rather
  than logits. About 30 minutes per mode.
- **Expectation**: user-legible evidence of whether the model got worse; only with it is changing a default defensible.
- **Acceptance**: answer agreement >= 95% per mode, with every difference listed.
- **Risk**: low (it only runs the model); but the task set has to be agreed first (§7).

### * C. A resident service plus an agent interface (the main use case)

- **Approach**: `kata serve`: JSONL requests from stdin/socket (`system`/`prompt`/`raw`/`max_tokens`/sampling/tools),
  reusing the Engine and PrefixCache, responses carrying timings; single-threaded queuing (this is a single-user machine).
  In the same milestone: Qwen3.8 tool-call template rendering plus argument JSON parsing, `stop` strings, and a rolling
  policy when the context fills (drop the oldest non-system turn and rebuild the state).
- **Expectation**: an agent step goes from `3.3 s + prefill` to just `prefill` (~0.03 s on a cache hit); tools become
  usable through something better than `--raw`.
- **Acceptance**: 100 consecutive requests with no leak; the same request run three times serially, with the 2nd and 3rd
  first token at <= 0.2 s; tool-call JSON round-trips correctly (checked against three examples of the official template).
- **Risk**: medium (template details, cache invalidation when rolling).

### * D. Layer-major prefill (+12% on long prompts)

- **Approach**: loop over layers on the outside and chunks on the inside, so each matrix is expanded once for the whole
  prompt; keep the activations resident per chunk (8K: 164 MB of residuals + 279 MB of fp16 x'). Start with a cheap
  preliminary experiment: chunk 2048/4096 with the `GM` sweep (16/24/48) redone, to check whether "bigger chunk is slower"
  was really just a grid order that was not retuned.
- **Expectation**: a 7.3K prompt goes from 207 to ~235 tok/s (+13%); linear benefit beyond 16K.
- **Acceptance**: output bit-identical to the per-chunk version; 8K prefill >= +8%; +0.5-1 GB resident.
- **Risk**: medium (the GDN recurrence must run continuously across chunks and attention must process the whole prefix;
  not a small amount of work).

### * E. In-situ int8 GEMM efficiency (+7% prefill)

- **Approach**: in situ it only reaches ~20 TOPS (33.8 in the microbenchmark, 64%), because every 128-group block needs an
  "int32 -> fp32 scaled accumulate" epilogue. Options: (1) prefetch all K-group scales into registers per lane; (2)
  double-buffer the K loop so that reading `cInt` overlaps updating `cAcc`; (3) the aggressive option — requantise each
  row's weights by the row maximum to int8 (so the scale is applied once), at the cost of introducing a **weight**
  quantisation (unlike the existing activation one), which needs a quality evaluation.
- **Expectation**: in-situ TOPS x1.3 -> `--fast-ffn` from +8% to +15%.
- **Acceptance**: in-situ >= 26 TOPS (1.3x); bit-identical numerics (1, 2) or the same KL order (3).
- **Risk**: medium (3 carries quality risk; 1 and 2 are pure performance work).

### F. fp16 GDN state (+2.4% decode)

- **Approach**: first a fake-quant probe (quantise the state to fp16 after `k_gdn_step`/`k_gdn_rec_*` and write it back)
  through `kl-eval` and the needle test; only if it passes, change the kernels and the checkpoint format (version +1).
- **Expectation**: state traffic 302 -> 151 MB/token, decode -1.5 ms, MTP cycle -1.2 ms.
- **Acceptance**: KL <= 1e-3 and top-1 >= 99.7% over 3072 positions; needle 3/3.
- **Risk**: low (the probe gates it), and the payoff is small.

### G. Cache engineering (size, portability, integrity)

- Chained incremental checkpoints (removing the 2x redundancy between the "message boundary" and "end" checkpoints);
  rebuild missing MTP K/V with `encMtpKv` so speculative and non-speculative checkpoints become interchangeable;
  an optional `--cache-verify` (xxhash).
- **Expectation**: roughly twice as many checkpoints for the same budget; better hit rate and cross-mode reuse.
- **Risk**: low.

### H. Small items (1.5-3% each, last)

Sampling-aware drafts (+2-3%; the model is written, only the wiring is missing), draft and verify in one command buffer
(+1.7%), a draft vocabulary subset (131072 rows, +1.6%). M17 already recorded each estimate and its reasoning.

### Not recommended (with the data, so it is not re-litigated)

| idea | data |
|---|---|
| re-laying the weight layout / 16-byte-wide reads | the staged prototype runs at the same speed as the original (125-135 vs 127-134 GB/s); the pure-read 141 GB/s does not carry into the GEMV (M12) |
| ~~wider verification + DFlash2~~ | **overturned**: M14 compared against T=2 instead of the bandwidth floor, and the prototype kernel had four flaws; see `R2-dflash2-revisit.md`, reopened as M25-M29 |
| persistent megakernel / ICB / graph replay | encoding is 0.7 ms/token (1.3%), the dispatch gap 0.61 us |
| using CPU/ANE/SME to help decode or expansion | CPU writes to DRAM cut the GPU matmul to 70%; the ANE serialises the MTP dependency chain (M0/CHECKPOINT-001) |
| overlapping expansion with the GEMM | measured slower (3.9 -> 5.6 s), the cache reuse is destroyed |
| binary archive / pre-touching pages / residency set | no steady-state gain; the cold start is merely moved elsewhere |

## 5. Next steps (milestones M18-M24)

| milestone | content | acceptance |
|---|---|---|
| **M18** | unify the numeric paths (`step` with fp16 activations) + the `encodeLayer` refactor + `make test` (in-repo golden, cross-path greedy agreement, cache regression) | `make test` green; cross-path agreement 20/20; no speed regression |
| **M19** | the quality baseline: task-level evaluation (arithmetic/code/continuation), producing an agreement table for four modes | >= 95% per mode; differences listed |
| **M20** | `kata serve` (JSONL) + tool-call template + stop strings + context rolling | 100 requests with no leak; hot request first token <= 0.2 s; three tool examples round-trip |
| **M21** | layer-major prefill (chunk/GM preliminary experiment first) | 8K prompt >= +8%; output bit-identical |
| **M22** | in-situ int8 GEMM efficiency (options 1, 2; option 3 subject to quality) | in-situ >= 26 TOPS; `--fast-ffn` >= +12% |
| **M23** | combination rules for lossy switches + the default decision (whether `--fast-ffn` becomes default, whether combinations are forbidden) | a combination matrix; defaults backed by task-level evidence |
| **M24** | cache: chained increments, cross-mode portability, optional checksums; fp16 GDN state (if M19's criterion passes) | three tables: size / hit rate / quality |

Ordering rationale: M18 is the foundation (without it every later step is on sand); M19 decides whether M22/M23 can land;
M20 is the functional gap for the owner's main use case; M21/M22 are the two remaining real performance items; M24 closes
out.

## 6. Decisions needed from the owner

1. **Should `--fast-ffn` become the default?** Alone it is 99.87% (passes), but combined with `--kv int8` it is 99.58%
   (fails). Three routes: (1) default on and make the combination mutually exclusive or warned; (2) keep it off (the status
   quo); (3) wait for M19's task-level evaluation. I lean to (3).
2. **The shape of M20's interface**: JSONL over stdin (simplest, no dependencies) or a unix socket + HTTP (easier to wire
   into agent frameworks)? I lean to stdin JSONL plus an optional `--listen` unix socket.
3. **Is +0.5-1 GB of resident memory acceptable for M21?** (At 64K context the total goes from ~7 GB to ~8 GB, still inside
   the 12.1 GB working set.)
4. **M19's task set**: I propose 20 arithmetic/reasoning problems + 8 code completions + 20 continuations (~30 min per
   mode). If you care about a specific benchmark (a GSM8K subset, your own agent tasks), specifying it now is cheapest.
