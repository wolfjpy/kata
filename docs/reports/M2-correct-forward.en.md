# Milestones M1/M2: a correct full forward (all 64 layers, every path except MTP)

**English** · [简体中文](M2-correct-forward.md)

Date: 2026-10-24

## Goal

Implement the whole of Qwen3.8-27B (PTQ1_0 + Hadamard) as a single-token forward, written the most direct way, and check
it numerically against Prism llama.cpp layer by layer — proving every formula and format detail first, and only then
starting performance work.

## Implementation

- `src/gguf.{h,cpp}`: a GGUF v3 reader that mmaps the whole file and wraps it as one `MTLBuffer(bytesNoCopy)`, binding
  tensors by file offset and reading the Hadamard sign vectors from the metadata.
- `src/gpu.{h,cpp}`: the Metal context (metal-cpp, with the offline-compiled `kata.metallib`).
- `config/kata_m6.h`: compile-time constants, validated against the GGUF metadata at load.
- `shaders/`:
  - `ops.metal`: embedding (dequantise + `s⊙(Hz)`), `k_rms_fwht` (four modes: RMSNorm / SwiGLU / head permutation, plus
    the signs and the 1024-point FWHT), `k_rmsnorm`, add, argmax.
  - `gemv.metal`: reference-quality PTQ1_0 and bf16 matrix-vector products, one simdgroup per row.
  - `gdn.metal`: GDN's causal convolution and recurrence step, one threadgroup per V head with each thread holding one row
    of the state.
  - `attn.metal`: attention's q/k RMSNorm + RoPE + KV write, and the streaming-softmax decode.
- `src/engine.{h,cpp}`: one command buffer per token, encoded layer by layer, with optional recording of each layer's
  residual and block output for the comparison.
- `src/main.cpp`: `kata --model M --ids 1,2,3 [--ref-dir D] [--ref-logits F] [--gen N]`.
- The Makefile compiles shaders with `-fno-fast-math` so the numbers stay comparable with the reference.

## Results

Comparison script: `tools/ref_logits` exports the reference, `build/kata` compares layer by layer.

| prompt | positions | inter-layer residual relative error | top-1 agreement | max logit difference |
|---|---|---|---|---|
| "The capital of France is" (5 tokens) | 5 | 1e-7 to 7e-5 | **5/5** | 0.0005 |
| a 72-token English passage | 72 | 7e-4 to 1.8e-3 | **72/72** | 0.0131 |
