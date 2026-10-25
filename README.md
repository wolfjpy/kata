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
build/bonsai run  --model $M --prompt "..." --no-spec              # plain decoding (MTP speculative decoding is on by default and exact)
build/bonsai run  --model $M --prompt "..." --fast-ffn             # int8 prefill for the FFN gate+up only: +7-9% prefill, passes the quality gate (KL 7.5e-5, top-1 99.87%; docs/reports/M13-fast-ffn.md)
build/bonsai run  --model $M --prompt "..." --fast                 # int8 prefill for every matrix: +10%, does NOT pass the quality gate (docs/reports/M5b-prefill-optimization.md)
build/bonsai run  --model $M --system "$(cat long_system_prompt.txt)" --prompt "..."   # the second run restores the system prompt from the disk cache
build/bonsai run  --model $M --prompt "..." --kv int8             # int8 KV cache: half the KV memory, +14% decode / +10% prefill at 29K context (lossy: KL 1.3e-4, top-1 99.71%)
build/bonsai cache [--clear]                                        # list / delete the persistent prefix cache (default ~/.cache/bonsai, 40 GB budget)
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
make build/bench_gemv  # GEMV microbenchmark on real tensors (fp32 activations); build/bench_gemv2 for the two-token variants
build/bonsai mtp-eval     --model $M --ids ... --gen 96     # how often the MTP head predicts the main model's greedy token
build/bonsai verify-check --model $M --ids ...              # two-token verification pass vs the exact decode path (KL, top-1)
build/bonsai kl-eval      --model $M --file corpus.txt --tokens 24576   # int8 prefill vs exact prefill on a corpus
sh tools/test_cache.sh                                       # regression test of the prefix cache (needs the model)
make build/probe_flash && build/probe_flash                 # prefill attention kernel: correctness vs CPU and speed per context length
```

## Status

See `docs/CHECKPOINT-001.md` (design and measurements), `docs/MODEL_SPEC.md` (verified formulas) and `docs/reports/` (one report per milestone).

Measured on the Mac mini M6 (the GPU throttles under sustained load, so repeated runs can differ by 10 to 15 percent).

| | this engine | reference (Prism llama.cpp, same machine) |
|---|---|---|
| decode, plain | **19.8 tok/s** | 15.6 tok/s (tg128) |
| decode, MTP speculative (greedy / sampling) | **25.0 / 22.7 tok/s** | |
| prefill, exact fp16, 512 tokens | **245 tok/s** | 208 tok/s (pp512) |
| prefill, exact fp16, 1024-token chunk | **257 tok/s** | |
| prefill, exact fp16, 7.3K-token prompt | **213 tok/s** | |
| prefill, `--fast-ffn` (opt-in, lossy, within the gate) | 274 to 280 tok/s (1024 chunk), 226 tok/s (7.3K) | |
| prefill, `--fast` (opt-in, outside the gate) | 281 tok/s | |

Persistent prefix cache (M9): a prompt whose token prefix was seen before restores its complete state from `~/.cache/bonsai` instead of prefilling it. A 2.9K-token system prompt: 15 s of prefill becomes 0.1 s of restore plus the 21 new tokens; a repeated 3.6K-token prompt: 18.6 s becomes 0.03 s. Outputs are byte-identical to the uncached run. Options: `--no-cache`, `--cache-dir`, `--cache-gb`, `--cache-min` (see `docs/reports/M9-prefix-cache.md`; checkpoints contain the prompt's KV, so use `--no-cache` for sensitive prompts).

Quality: speculative decoding keeps the output distribution unchanged (verification pass vs exact decode: mean KL 4.3e-6, top-1 100 percent over 830 positions). The exact prefill path matches the fp32 golden logits (mean KL 1.4e-7). The int8 path does not meet the proposed gate (mean KL 4.4e-4, top-1 99.54 percent on 1536 positions), so it is off by default.
