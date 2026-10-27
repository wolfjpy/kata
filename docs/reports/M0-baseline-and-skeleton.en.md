# Milestone M0: baseline, reference tooling and the project skeleton

**English** · [简体中文](M0-baseline-and-skeleton.md)

Date: 2026-10-24

## What was done

1. **Pinned and checked the latest versions** (see `VERSIONS` in the repository root): macOS 27.0.1, Xcode 27.1, Metal
   compiler 32023.921 (language version ceiling 4.1), metal-cpp (`apple/metal-cpp` main, macOS 27.0.1), Prism llama.cpp
   `b3ebc47` (the newest commit as of 2026-10-24, newer than the previous `73bfe6cc`).
2. **Downloaded and verified the model**: `mannix/Bonsai-2-27B:mtp-ptq1_0`, 6,397,969,920 bytes, SHA-256 matching the
   Ollama blob name (`174c9c65...`). It lives outside the repository, in `../../models/`.
3. **Built the reference runtime**: Prism llama.cpp (Metal, Release), outside the repository in `../../refs/llama.cpp-prism`.
4. **Reference tool** `tools/ref_logits.cpp`: links against Prism's libllama and dumps, for a given token-id sequence or
   text, the logits at every position, the per-layer residual outputs (`l_out-<il>`) and each layer's norm/attention/FFN
   outputs, plus greedy continuation. Build with `make ref_logits`.
5. **Project skeleton**: `Makefile` (plain make), `config/kata_m6.h` (compile-time constants shared by C++ and MSL),
   `third_party/metal-cpp` (pinned), `shaders/`, `src/`. The offline Metal toolchain was verified (`metal -std=metal4.1`
   -> `metallib`), that the shared header compiles inside MSL, and that a kernel actually runs on the GPU.
6. **Model specification** `docs/MODEL_SPEC.md`: the exact formulas verified against the Prism sources (the GDN
   recurrence, attention, the order and permutation of the Hadamard transform, and the inverse transform used by the
   embedding).

## Baseline (measured on this machine)

`llama-bench -ngl 99 -r 3`, Prism llama.cpp `b3ebc47`, model `Ternary-Bonsai-2-27B-PTQ1_0-MTP`:

| test | throughput |
|---|---|
| **pp512** | **208.08 ± 7.12 tok/s** |
| **tg128** | **15.64 ± 0.06 tok/s** |

- Effective decode bandwidth is about 15.64 x 5.66 GB = 88.5 GB/s, only 62% of the sustained 142 GB/s.
- A sample of the reference output: `The capital of France is` -> ` Paris.\nThe capital of Germany is Berlin.\nThe`, token
  ids `760 6511 314 9338 369`, last-position top-1 `11751` (logit 14.7327).

## Targets, relative to the baseline

- decode: about 25 tok/s plain autoregressive (+60%), and about 36-40 tok/s with MTP (+130% to +155%).
- prefill: about 440-590 tok/s on the int8 path (2.1x to 2.8x), and about 160-215 tok/s on the exact fp16 path (level with
  the baseline, so the exact path is only used when quality demands it).
