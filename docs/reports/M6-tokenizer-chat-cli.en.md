# Milestone M6: tokenizer, chat template, sampling and the command line

**English** · [简体中文](M6-tokenizer-chat-cli.md)

Date: 2026-10-24

## Implementation

- `src/tokenizer.{h,cpp}`: byte-level BPE. The vocabulary and merges are read straight from the GGUF; GPT-2's
  byte<->unicode mapping; O(n log n) BPE merges with a priority queue over a doubly linked list; 33 added tokens (types
  3/4) matched as literals first; incremental UTF-8 detokenisation, so streaming output never shows a truncated character.
- `src/pretok.mm`: NFC normalisation and Qwen's pre-tokenizer regex (an ICU regex with `\p{L}\p{M}\p{N}`, called through
  Foundation's `NSRegularExpression`). This is the only Objective-C++ file in the project.
- `src/chat.{h,cpp}`: the parts of the Qwen3.8 chat template the engine needs (system/user/assistant, `enable_thinking`,
  `reasoning_effort` at xhigh/medium/low, the `<think>` generation hint) plus the sampler (llama.cpp's default chain:
  top-k -> top-p -> min-p -> temperature -> draw; temperature <= 0 means greedy).
- `src/cli.cpp`: `kata run` (single turn), `kata chat` (multi-turn, reusing the KV cache and GDN state and prefilling only
  the new suffix each turn) and `kata tokenize`.
- Short suffixes (<= 24 tokens) use per-token decode instead of a full prefill, because a prefill pays ~0.4-0.6 s of fixed
  weight-expansion cost every time.

## Verification

- **The tokenizer is id-for-id identical to `llama-tokenize` on 24 inputs**: 14 hand-written cases (prose, code, CJK, emoji
  and ZWJ, numbers with full-width/superscript/circled forms, whitespace and CRLF, abbreviations, JSON, special tokens
  and `<think>`/`<tool_call>`) and 10 real files (a 40k-token mixed Chinese/English document, Python, C++, Metal sources,
  a Makefile). Note that `llama-tokenize` needs `--no-escape` for the comparison, since it expands escapes like `\n` by
  default.
- End to end: `"What is the capital of France?"` -> `The capital of France is Paris.`; thinking mode produces a reasoning
  chain and the right answer (17 x 24 = 408); in a two-turn conversation the second turn correctly recalls the first
  turn's information, which proves the KV and GDN state reuse is correct.

## Known limits

- The first call compiles MPP's GEMM pipeline for about 2 s (hence the slow first `prefill`); a binary archive or creating
  the pipelines in parallel at load time would remove that.
- The default path is exact fp16; `--fast` enables the int8 path, which will not become the default until it clears the
  quality gate on a larger sample.
- Tool / function-calling template rendering is not supported; for agent scenarios pass an already-rendered prompt with
  `--raw`.
- Sampling happens on the CPU, which reads back 1 MB of logits per token (~0.3-0.5 ms); moving it to the GPU is planned.
