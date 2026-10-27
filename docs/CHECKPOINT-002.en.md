# CHECKPOINT-002: the engine exists (M0-M6 + MTP + prefill optimisation)

**English** · [简体中文](CHECKPOINT-002.md)

Date: 2026-10-24 · A self-contained recovery document. The full record of the design stage is in `CHECKPOINT-001.md`;
the details of each milestone are in `reports/`.

## 1. State

Project `kata-m6`: a Metal-only inference engine for Ternary Bonsai-2-27B (Qwen3.8-27B, `qwen35` architecture, PTQ1_0
Hadamard-folded ternary weights plus an MTP layer), C++17 + MSL, target machine Mac mini M6 / 16 GB. The GGUF is mapped
zero-copy; no Python anywhere.

| metric | this engine | baseline (Prism llama.cpp, same machine) |
|---|---|---|
| decode, plain | 19.8 tok/s | 15.6 tok/s (`tg128`) |
| decode, MTP speculative (greedy / sampled) | **25.0 / 22.7 tok/s** | |
| decode, MTP, 7.3K context | 21.4-21.7 tok/s | |
| prefill, exact fp16, 512 tokens | **245 tok/s** | 208 tok/s (`pp512`) |
| prefill, exact fp16, 1024-token chunk | **257 tok/s** | |
| prefill, exact fp16, 7.3K-token prompt | 213 tok/s | |
| prefill, int8 (`--fast`, not recommended) | 270-283 tok/s | |

(A fanless machine throttles under sustained load; the same configuration can differ by 10-15% between runs, so A/B
comparisons must be interleaved.)

Quality (all against the exact fp32 path):
- MTP verification path (fp16 activations + half2 LUT GEMV): 830 positions, mean KL 4.3e-6, max 8.4e-4, top-1 100%. The
  output *distribution* is unchanged (exact speculative sampling from a deterministic draft), but greedy output
  occasionally forks at a near-tie.
- exact prefill: mean KL 1.4e-7.
- int8 prefill: 1536 positions of a 24.5K-token corpus, mean KL 4.4e-4, **top-1 99.54%, which misses the 99.7% gate**, so
  it stays off by default.

Git: `a6020e4` (M0) -> `4318bf0` (M2) -> `89af761`/`d97f58a` (M3) -> `fbf5ccb` (M5) -> `8d21717` (M6) ->
`c435af5`/`f7f3796` (M4 MTP) -> `00264f7` (flash attention) -> `96e048d` (expansion) -> `a3581ee` (int8) -> `53b04a8`
(GEMM order) -> `9d8fe1d` (flash decoding).

## 2. Source map

| file | role |
|---|---|
| `config/kata_m6.h` | constants shared by host and MSL (shapes, tile sizes); validated against the GGUF metadata at load |
| `src/gguf.*` | GGUF parsing, whole file mmapped |
| `src/gpu.*` | Metal device/queue/pipeline cache (metal-cpp; `*_PRIVATE_IMPLEMENTATION` only here) |
| `src/engine.*` | the forward pass: `step` (1-token decode), `prefill` (batched), `verify2` (2-token verify), `mtpRows` (MTP head), `feed` (the common entry point, maintains the MTP cache), `encMtpKv`, `dDecAttn` |
| `src/spec.*` | the MTP speculative main loop (`specGenerate`) with triple-buffered state rollback |
| `src/tokenizer.*`, `src/pretok.mm` | BPE tokenizer (ICU regex; the only Objective-C++ file), token-for-token identical to `llama-tokenize` |
| `src/chat.*` | Qwen3.8 chat template and sampler (top-k -> top-p -> min-p -> temperature) |
| `src/cli.*`, `src/main.cpp` | subcommands `run`, `chat`, `tokenize`, `mtp-eval`, `verify-check`, `kl-eval`, plus the subcommand-less development entry point (per-layer comparison, KL, `--gen`, `--verify`, `--skip`) |
| `shaders/gemv_fast.metal` | decode GEMV (the fp32 g/c form) |
| `shaders/mtp.metal` | Q8_0 GEMV/expansion, the MTP input, `k_gemv_lut2` (2-token LUT + half2), `k_gdn_rec_v2`, `k_gdn_conv_v2`, `k_gdn_ab2` |
| `shaders/prefill.metal` | expansion (v2), fp16/int8 GEMM (MPP `matmul2d`), batched GDN/RMSNorm/FWHT kernels |
| `shaders/flash.metal` | prefill flash attention, decode flash-decoding |
| `shaders/gemv_exp.metal`, `layout_probe.metal` | ablation and probe kernels (not used by the engine) |
| `tools/` | `ref_logits` (Prism reference), `bench_gemv`/`bench_gemv2`, `alu_probe`, `probe_flash`, `kv_read_probe`, `layout_probe` |
| `docs/MODEL_SPEC.md` | the verified formulas, MTP included (§8) |

## 3. Build and reproduce

```sh
make                                   # build/kata + build/kata.metallib
M=../../models/bonsai-2-27b-mtp-ptq1_0.gguf
build/kata run --model $M --prompt "..." --temp 0 --max-tokens 200         # MTP on by default; --no-spec turns it off
build/kata --model $M --ids ... --prefill --repeat 2 [--int8]              # prefill timing (first run includes ~2 s cold start)
build/kata --model $M --ids ... --prefill --nlogits 72 --cmp-logits golden72.bin   # against saved fp32 logits
build/kata --model $M --ids ... --prefill --verify 20 [--skip a,b]         # 2-token verify timing and ablation
build/kata verify-check --model $M --ids ...                               # verification path vs exact decode
build/kata kl-eval --model $M --file corpus.txt --tokens 24576             # int8 vs exact prefill
build/bench_gemv2 $M build/kata.metallib k_gemv_lut2_nr4_sg4 ...           # GEMV microbenchmark
```

Environment variables (development only): `KATA_SKIP=a,b` (kernel ablation at run time), `KATA_GM` (GEMM group size,
default 32), `KATA_SCALAR_ATTN=1` / `KATA_SCALAR_DEC=1` (the old scalar attention, for A/B), `KATA_DEC_CHUNK` (keys per
decode split), `KATA_OLD_EXPAND`, `KATA_VERIFY_V1H`, `KATA_I8_EXCLUDE=down,wo,...` (matrices that stay exact fp16 on the
int8 path). The test fixtures `/tmp/ids72.txt`, `/tmp/golden72.bin`, `/tmp/big1024.ids` are regenerated by
`build/ref_logits` (which needs the Prism reference runtime, see `VERSIONS`); they are not in the repository.

## 4. Hypotheses that measurement overturned (each with its data, in the report named)

1. **The GEMV bottleneck depends on how many tokens one weight read serves** (M3, M4 §3; an earlier phrasing of this,
   "decode GEMV is ALU-bound, not bandwidth-bound", was wrong and has been corrected):
   - **T=1 (plain decode) is bandwidth-bound**: 126-134 GB/s, 89-94% of the 142 GB/s sustained bandwidth; removing the
     decode step from an ablation makes it only 3-7% faster. M3's "not ALU-limited" holds for T=1.
   - **T>=2 (verification) is ALU-bound**: every extra token adds a full set of multiply-adds while the weight bytes stay
     the same. In the fp32 g/c form T=1->2 costs +59% and T=3 almost +200%; switching to a threadgroup-memory trit
     lookup (16-byte entries, one read per weight byte) with half2 packing of two tokens made T=2 33% faster.
   - Tooling notes: `floor` is essentially free (it fuses with the preceding FMA), `fract` is not; half2 FMA has ~1.9x the
     element throughput of fp32; random 2-byte threadgroup reads are the bottleneck of the LUT scheme.
   - **The remaining ~8-10% gap at T=1 comes from the read pattern itself** (`tools/weight_read_probe`): with no compute
     at all, the 2-byte/1-byte small reads on the native 28-byte block layout reach only 125-132 GB/s, while the same
     byte count as 16-byte vector reads reaches 141-142 GB/s (= the DRAM ceiling). Recovering that requires re-laying the
     weights into a 16-byte-aligned layout (see §5).
2. **A persistent megakernel is not needed**: the dispatch gap is 0.6 us; a global barrier costs 0.45-4.7 us, and
   persistence made things 2-11% *slower*.
3. **GEMM grid order matters most** (M5b §2.2): one-dimensional linear ids with "M-first grouping" (GM=32) is 39% faster
   than the original two-dimensional dispatch; one-dimensional N-first is 3x slower. The hardware schedules the 2D
   dispatch in a locality-friendlier order. Chunks above 1024 rows get *slower* (the A matrix no longer fits in cache).
4. **The exact fp16 GEMM is already at the matrix-unit peak** (16.6 TFLOPS). The int8 path, because of its epilogue and
   7-bit activation quantisation, is only ~10% ahead and fails the quality gate.
5. **Overlapping expansion with the GEMM is slower** (the 178 MB of expansion writes evict the GEMM's cached reuse).
6. **The decode attention kernel is already at ~65% of the KV read bandwidth**; the K/V layout (strided vs contiguous
   per head) makes no difference; K+V is 4 KB per key per layer (64 KB per token). Long context can only come from
   compressing the KV.
7. **The ~3 s cold start can be hidden but not removed**: ~2 s is the OS faulting in the mapped 6.4 GB (single-threaded
   ~3.4 GB/s, and 8 threads in parallel are slower), ~1 s is the first dispatch. Pre-touching pages and Metal binary
   archives gave no net gain; a small warm-up forward at load time merely moves the ~3 s into the load phase. What was
   implemented: `chat` warms up on a background thread while the user is typing (first-turn prefill 3.4 s -> 1.1 s). A
   one-shot `run` cannot hide it; a long-lived process (chat, server) pays it once.
8. **MPP / cooperative tensor rules**: tensor element types must not be `const`; with a real extent on the tail tile MPP
   zero-pads out of bounds (which avoids `NaN x 0`); `cS.load(broadcast)` gives wrong results, so index scales
   element-wise via `get_multidimensional_index`; the 64x64 / 4-simdgroup destination layout is fixed (4 rows and 2
   column blocks per lane); `matmul2d` wants `int8_t`, not `char`, for int8 operands.
9. **A 72-position test hides error accumulation**: any quantisation change must be evaluated with `kl-eval` on a corpus
   of at least 20K tokens.
10. **Throttling**: from cold, the same 1024-token exact prefill run 14 times in a row (~56 s) stays at 3.87-4.02 s with
    no throttle; throttling appears after several minutes of sustained load (10-30% slower observed) and on the
    power-hungrier pure-matmul probes (17.3 -> 12.6 TFLOPS within seconds). Software cannot pin the GPU clock, so A/B
    comparisons must be interleaved and repeated, and the machine allowed to cool for 1-2 minutes when in doubt.

## 5. Not done yet (ordered by expected payoff)

| item | expected payoff | note |
|---|---|---|
| a prefill GEMM that fuses dequantisation | prefill +10-14% | expansion still costs 0.56 s per 1024 tokens; needs per-M-block re-decoding, a large change |
| KV compression (fp8/int8) | 32K-context decode +20-30% | lossy, needs a quality gate |
| **re-lay the weights into a 16-byte-aligned layout** (`kata-pack`; 8 rows x 1024 K per tile, 14 cache lines with no padding) | decode +8-10% (read probe ceiling 141 vs 127 GB/s); the verification path likewise | needs a packing tool plus rewritten GEMV/expansion/embedding kernels; the file grows by 6.4 GB and is still zero-copy mmappable |
| verification GEMV (T=2) from 113 to ~125 GB/s | MTP cycle -5 ms (+8%) | the LUT reads remain the bottleneck; can be solved together with the re-layout above |
| **wide verification + a DFlash2 drafter** (see `reports/M7-speculation-options.md`) | code/reasoning decode possibly 2x+ (measured acceptance length 5.1); prose about break-even | requires the new wide verification kernel (the current kernel costs 3.2x at T=8; the matrix-unit prototype takes 831 us per 16 tokens, ~5x slower) plus DFlash2 and Q4_K_M dequantisation in Metal; multi-day, high risk |
| int8 KV (`half x char` is a natively supported `matmul2d` type; scale the columns of S by the K scale per key, and P by the V scale per key) | 32K-context decode +8-10%, half the memory at 64K | lossy but far more accurate than fp8; simpler than fp8 (no MTLTensor) |
| draft vocabulary pruning (run lm_head on a frequent subset only) | MTP cycle -2 ms (+3%) | small risk of lowering the acceptance rate |
| prefill for short prompts (25-200 tokens) | chat turn latency 0.9 s -> ~0.3 s | a full expansion currently costs 0.9 s; a T=8 LUT-verification-style batch would do |
| sampling on the GPU | tiny | the CPU currently reads back 1 MB of logits per token |
| tool-call template, `<think>` budget | feature | not implemented |
| ANE / SME / Metal 4 ICB | no expected payoff | the conclusion from the design-stage analysis; the dispatch gap is already 0.5% |

## 6. Questions that needed the owner's decision

1. **Quality gate**: the proposal is KL <= 1e-3 nat, top-1 >= 99.7% (on >= 20K tokens). By that measure int8 prefill fails
   (already off by default) and the MTP verification path passes (KL 4.3e-6). Is the fp16/half2 numerics of the MTP
   verification path acceptable, given that greedy output occasionally forks at a near-tie?
2. Should KV compression (lossy) be done in exchange for long-context decode speed?
3. Default thinking mode (on today, `--no-think` turns it off) and the thinking budget.
4. 64K context runs, but the resident footprint is ~12.7 GB (6.4 GB of weights + 6.3 GB of buffers), right at the limit of
   a 16 GB machine; should `--ctx` be capped at 48K?
