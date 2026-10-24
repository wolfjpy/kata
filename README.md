# bonsai-m6

A Metal-only inference engine specialised for **Ternary Bonsai 2 27B** (Qwen3.8-27B, PTQ1_0 with Hadamard-folded ternary weights) on a **Mac mini M6, 16 GB**. C++17 + Metal Shading Language, no Python, no frameworks. Speed is the only goal.

## Build

```sh
make            # build/bonsai and build/bonsai.metallib (needs Xcode 27 with the Metal toolchain)
```

The model file is expected outside the repository (see `VERSIONS`): `../../models/bonsai-2-27b-mtp-ptq1_0.gguf` (ollama `mannix/Bonsai-2-27B:mtp-ptq1_0`).

## Use

```sh
M=../../models/bonsai-2-27b-mtp-ptq1_0.gguf
build/bonsai run  --model $M --prompt "Explain unified memory in one paragraph." --max-tokens 300
build/bonsai chat --model $M --system "You are concise."          # interactive; the KV cache and GDN state are reused between turns
build/bonsai run  --model $M --prompt "..." --no-think --temp 0    # thinking off, greedy
build/bonsai run  --model $M --raw --file prompt.txt               # pre-rendered prompt (special tokens parsed)
build/bonsai run  --model $M --prompt "..." --fast                 # int8 prefill path (see docs/reports/M5-prefill.md for the quality status)
build/bonsai tokenize --model $M --text "hello"                    # token ids, identical to llama-tokenize
```

Defaults follow the model card: temperature 1.0, top-k 20, top-p 0.95, min-p 0.05, thinking enabled with `xhigh` effort.

## Development tools

```sh
make ref_logits        # reference runtime wrapper (needs the Prism llama.cpp checkout, see VERSIONS)
build/ref_logits --model $M --text "..." --out ref.bin --dump-dir dump/ --gen 12
build/bonsai --model $M --ids 1,2,3 --ref-dir dump/ --ref-logits ref.bin     # layer-by-layer comparison
build/bonsai --model $M --ids ... --prefill [--int8] --nlogits 72 --cmp-logits golden.bin   # KL / top-1 against saved logits
build/bonsai --model $M --ids ... --gen 64 --skip gemv_gate,k_gdn_step       # per-kernel ablation timing
make build/bench_gemv  # GEMV microbenchmark on real tensors
```

## Status

See `docs/CHECKPOINT-001.md` (design and measurements), `docs/MODEL_SPEC.md` (verified formulas) and `docs/reports/` (one report per milestone).

| | this engine | reference (Prism llama.cpp, same machine) |
|---|---|---|
| decode | **20 tok/s** | 15.6 tok/s |
| prefill, int8 path (1024-token chunk) | **278 tok/s** | 208 tok/s (pp512) |
| prefill, exact fp16 path | 183 tok/s | |
