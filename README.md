# Bonsai

A from-scratch **Metal inference engine** for [Bonsai 2 27B](https://huggingface.co/prism-ml) — a 27B hybrid-attention
model whose weights are stored at **1.58 bits** (ternary ±1 with fp16 group scales) — running entirely on-device in a
16 GB Apple laptop.

No llama.cpp, no MLX, no MPS graph: every kernel is hand-written Metal Shading Language, including the on-the-fly 2-bit
weight decoder, the gated DeltaNet recurrence, the attention path, a tensor-core (`matmul2d`) prefill GEMM, and the MTP
speculative-decoding head.

| | |
|---|---|
| **Weights on disk / in memory** | 6.4 GB single GGUF (27B parameters: ternary weights with fp16 group scales, plus a Q8_0 MTP head) |
| **Prefill** | 280 tok/s (1K chunk) · 227 tok/s (7.3K prompt) |
| **Decode** | 25.4 tok/s prose · 28–29 tok/s reasoning · 31.6 tok/s structured code (greedy) |
| **Context** | 32K (fp16 KV ≈ 2.1 GB, int8 KV ≈ 1.1 GB, opt-in) |
| **Hardware** | Apple M6, 16 GB unified memory, Metal 4 |

---

## Highlights

- **Ternary weights, decoded in the kernel.** PTQ1_0 packs 5 weights into 2 bytes as base-3 digits. The GEMV kernel
  turns a 256-entry lookup table of 8-wide ternary patterns into one dispatch, so no dequantised copy of the model ever
  exists in memory.
- **Hybrid attention, both kinds.** 48 of the 64 layers are gated DeltaNet (conv state + delta-rule recurrence, one
  kernel per token, state kept in registers); the other 16 are GQA attention with per-head QK-norm, partial RoPE and a
  sigmoid output gate. The two paths are fused end to end in a single command buffer per layer.
- **Speculative decoding with the model's own MTP head** (blk.64, Q8_0): a one-token draft per step, verified in the
  same command buffer as the target forward, so a draft is never paid for twice.
- **Prefix cache.** A prompt whose token prefix was seen before restores its complete state (KV, conv state, recurrence
  state) from `~/.cache/bonsai` instead of prefilling it: a repeated 3.6K-token prompt goes from 18.6 s to 0.03 s, with
  byte-identical output.
- **A quality gate, not a vibe.** Every lossy switch is measured against the exact fp16 path with a KL / top-1 harness
  (`bonsai kl-eval`), and switches that fail it are documented as failed instead of shipped.

## Quick start

```bash
# 1. Build (Xcode command line tools, macOS 15+; Metal 4 / MPP needs a recent SDK)
make            # builds build/bonsai, build/bonsai.metallib and the dev tools

# 2. Get the model: a single GGUF (~6.4 GB), e.g.
#    hf download prism-ml/Bonsai-2-27B-gguf bonsai-2-27b-mtp-ptq1_0.gguf --local-dir models/

# 3. Run
build/bonsai run --model models/bonsai-2-27b-mtp-ptq1_0.gguf \
                 --prompt "Explain why a delta-rule recurrence is cheaper than attention at inference." \
                 --max-tokens 256

# 4. Interactive chat (streaming, thinking levels, prefix cache on by default)
build/bonsai chat --model models/bonsai-2-27b-mtp-ptq1_0.gguf
```

## Usage

```
bonsai run    --model M.gguf --prompt P [--system S] [--raw] [--no-think]
              [--effort xhigh|medium|low] [--temp T] [--top-k K] [--top-p P]
              [--max-tokens N] [--ctx N] [--no-spec] [--no-cache] [--exact]
bonsai chat   --model M.gguf [same options]
bonsai kl-eval --model M.gguf --file corpus.txt [--tokens N] [--save-ref R.bin | --ref R.bin]
bonsai verify-check --model M.gguf ...
```

Quality / performance switches:

| flag | effect | quality (top-1 / mean KL) |
|---|---|---|
| *(default)* | exact fp16 prefill, MTP speculative decoding on | reference |
| `--fast-ffn` | int8 FFN gate/up/down during prefill: **+10%** prefill | 99.87% / 7.5e-5 ✅ |
| `--fast` | int8 for every prefill matrix: +11% prefill | 98.48% / 0.0026 ❌ |
| `--kv int8` | int8 KV cache: half the KV memory, +14% decode at long context | 99.51% / coarse scales ⚠️ |
| `--exact` | force every lossy path off | reference |
| `--no-spec` | disable MTP speculative decoding | identical output |

## Performance

Apple M6, 16 GB, `--temp 0`, single run per row (measurements drift ±5% with thermal state).

| Workload | tok/s |
|---|---|
| Prefill, 1K tokens (exact / `--fast-ffn` / `--fast`) | 249 / 280 / 262 |
| Prefill, 7.3K prompt (`--fast-ffn`) | 227 |
| Decode, prose | 25.4 |
| Decode, reasoning text (MTP accepting 85–90% of drafts) | 28–29 |
| Decode, structured/greedy code | 31.6 |
| Decode, 29K context | 18.4 |

Decode is bound by streaming 2.4 GB of weights per token: at 25–31 tok/s the engine sustains roughly 70 GB/s of
effective bandwidth, which is where the roofline for this model sits on a laptop memory system. Prefill is bound by the
ALU cost of decoding 2-bit weights, not by bandwidth (`--fast-ffn` wins by shrinking the ALU work, not the bytes).

## Quality gates

Lossy switches are only accepted if they stay above **99.7% top-1** and below **KL 1e-3** against the exact fp16 path
over 3072 positions:

```bash
# exact vs int8 prefill (single process, both paths)
build/bonsai kl-eval --model $M --file corpus.txt --tokens 3072 --keep 256 --chunk 512

# a configuration that changes buffer layouts (e.g. int8 KV) needs two processes
BONSAI_KV=fp16 build/bonsai kl-eval --model $M --file corpus.txt --save-ref ref.bin
BONSAI_KV=int8 build/bonsai kl-eval --model $M --file corpus.txt --ref ref.bin
```

Current results: `--fast-ffn` 99.87% / 7.5e-5 ✅ · `--fast` 98.48% / 2.6e-3 ❌ · `--kv int8` **99.51% top-1** — one
scale per (position, KV head) covers 256 elements, so an outlier costs 1-2 effective bits. Asymmetric (mean + scale)
quantisation was implemented and measured *worse* (99.37%), because a head's values are already near-symmetric; opt-in,
not a default. (Its
mean-KL looks alarming only because a few positions have reference logits around 1e6, where a 0.5% relative error is a
5000-nat shift and the KL is meaningless; `BONSAI_KL_DBG=1` prints those positions.)
Speculative decoding and the prefix cache are exact by construction and verified byte-identical.

## How it works

```
x = embed(token)
for il in 0..63:
    h = rmsnorm(x, attn_norm[il])
    a = gated_delta_net(h)        if il % 4 != 3
      = gqa_attention(h)          if il % 4 == 3     # 16 layers, per-head QK-norm, partial RoPE, sigmoid gate
    x = x + a
    h = rmsnorm(x, post_attention_norm[il])
    x = x + ffn_down(silu(ffn_gate(h)) * ffn_up(h))
logits = lm_head(rmsnorm(x, output_norm))
```

Every ternary matmul rotates its input with a fast Walsh–Hadamard transform first (the quantisation basis), which is
fused into the RMSNorm kernel. One command buffer per layer keeps the ~50 dispatches of a decode token in flight without
a CPU round trip; the MTP draft, target forward and verification share that buffer.

Repository layout:

```
src/        engine (model, layers, kernels dispatch), session, sampler, tokenizer, chat template, CLI
shaders/    Metal kernels: ternary GEMV/GEMM, GDN, attention, norms, MPP matmul, quantisation
tools/      reference logit exporters, kernel microbenchmarks, quality harnesses
docs/       model spec, per-milestone reports, checkpoints
```

## What did not work (and why it is in the docs)

This engine has a paper trail of negative results, because they are the expensive part:

- **Turbo-fish 0.5-bit and 2:4 sparsity**: both need a different weight layout; measured slower than the 1.58-bit
  layout for this model.
- **Deeper MTP chains** (2–3 drafts): acceptance collapses in the chain path and, even at the reference acceptance
  rate, the wider verification costs 1.57× per round while producing 1.68× the tokens — a wash at best.
- **A DFlash2-style 7-token drafter**: the draft network's first-token agreement is only ~0.4 in this port, so the
  wide window does not pay off yet.
- **Wide verification on tensor cores**: the `matmul2d` path at M=16 runs at 9.4 TFLOPS (54% of peak on this chip),
  which puts the achievable speedup of an 8-row verification at 1.8×, not the 1.6× that would make deep speculation
  profitable. Two-row-group pipelining was implemented, measured 1.6× *slower*, and removed.

## Documentation

`docs/MODEL_SPEC.md` is the ground truth for the architecture (verified against the reference implementation and
per-layer logit dumps). `docs/CHECKPOINT-00{1..4}.md` are milestone summaries; `docs/reports/M*.md` are the individual
investigations, each with the measurement that justified the next step — including the ones that failed.

## Requirements

macOS on Apple Silicon with Metal 4 (MPP `matmul2d`), 16 GB of unified memory, Xcode command line tools. The engine
targets one model (Bonsai 2 27B, PTQ1_0 + MTP head) and one machine class; the kernels are specialised for the model's
shapes (hidden 5120, FFN 17408, 24 Q / 4 KV heads, vocab 248320) rather than general.

## References

This engine is a clean-room implementation, but it would not exist without these sources:

**The model and its quantisation format**
- Bonsai 2 27B (Prism) — model weights, config and the PTQ1_0 ternary quantisation format, plus the published
  whitepaper describing the architecture (`docs/reference/bonsai-2-27b-whitepaper.pdf`).
- `docs/MODEL_SPEC.md` — the architecture as verified against the reference implementation and per-layer logit dumps.

**Reference implementation**
- [llama.cpp](https://github.com/ggml-org/llama.cpp) (ggml) — the GGUF container format, the general graph structure,
  and the reference behaviour for sampling, chat templates and tokenizer handling.
- The Prism llama.cpp fork — `src/llama-graph.cpp` / `src/llama-model.cpp` for the Qwen3.5-style graph, the PTQ1_0
  dequantisation (`prism_ptq1_0_quant_dequant.c`), the Metal gated-DeltaNet kernel
  (`prism_metal_gated_delta_net.metal`) and the T=4 verification graph (`prism_qwen35_graph.cpp`). These were read as
  the specification for numerics; the kernels here are independently written.

**Apple / Metal**
- *Metal Shading Language Specification* (Metal 4) — the authoritative reference for the language, `simdgroup_matrix`,
  `threadgroup` memory and barrier semantics.
- *Metal Performance Primitives* (`MetalPerformancePrimitives/MetalPerformancePrimitives.h`) — `matmul2d_descriptor`,
  `matmul2d::run`, `cooperative_tensor`, `get_row_reduction_destination_cooperative_tensor`, which the prefill GEMM and
  the attention kernels are built on.
- *Metal Feature Set Tables* — for what is available at each `MTLGPUFamily` / Apple-GPU generation.

**Papers the algorithms come from**
- Yang, Kautz, Hatamizadeh, *Gated Delta Networks: Improving Mamba2 with Delta Rule* (2024) — the gated DeltaNet layer.
- Schlag, Irie, Schmidhuber, *Linear Transformers Are Secretly Fast Weight Programmers* (2021) — the delta rule.
- Ainslie et al., *GQA: Training Generalized Multi-Query Transformer Models* (2023) — the 24/4 grouped-query attention.
- Dao et al., *FlashAttention* (2022) — the tiled, online-softmax attention structure used in the prefill kernel.
- Leviathan, Kalman, Matias, *Fast Inference from Transformers via Speculative Decoding* (2023) — the draft/verify
  loop and its acceptance rule.
- Hu et al. / the MTP head design in DeepSeek-V3 (2024) — multi-token-prediction heads used as a self-draft.
- Ashkboos et al., *QuaRot* (2024) and Tseng et al., *QuIP#* (2024) — the randomised Hadamard transform used as the
  input basis for the ternary matmuls, and the incoherence argument behind it.

## License

MIT — see [LICENSE](LICENSE). Model weights are distributed by their authors under their own licence; nothing here
redistributes them.
