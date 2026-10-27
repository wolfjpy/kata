> 本文以中文写成;**英文全译:[M6-tokenizer-chat-cli.en.md](M6-tokenizer-chat-cli.en.md)**,下方 `## Summary (EN)` 为摘要,双语文档索引见 [docs/README.md](../README.md)。
> *This document is in Chinese. An English summary follows; the bilingual index is at [docs/README.md](../README.md).*

## Summary (EN)

**Goal.** Tokenizer, chat template, sampling and the command line.

**Result.** Byte-level BPE decoded straight from the GGUF (word table + merges, priority queue over a doubly linked list, 33 added tokens as literals); NFC normalisation and the Qwen pre-tokenizer regex through ICU via `NSRegularExpression` — the only Objective-C++ file in the project; the parts of the Qwen3.8 chat template the engine needs (system/user/assistant, `enable_thinking`, `reasoning_effort` xhigh/medium/low) plus sampling; and `kata run` / `kata chat` / `kata tokenize`. Short suffixes (<=24 tokens) use per-token decode instead of a full prefill, because a prefill pays a 0.4-0.6 s fixed cost for expanding the weights.

**Verification.** Tokeniser output is id-identical to `llama-tokenize` on 24 inputs (14 hand-written: prose, code, CJK, emoji and ZWJ, numbers with full-width/superscript/circled forms, whitespace and CRLF, abbreviations, JSON, special tokens and `<think>`/`<tool_call>`; plus 10 corpus samples). End to end: correct factual answers, correct chain-of-thought with thinking on, and a two-turn conversation that recalls the first turn's information, which proves the KV and GDN state reuse is correct.

**Known limits.** The first MPP GEMM pipeline compiles for ~2 s (hence the slow first prefill); the default path is the exact fp16 one.

---

# 里程碑 M6:分词器、chat template、采样与命令行

日期:2026-10-24

## 实现

- `src/tokenizer.{h,cpp}`:byte-level BPE。词表和 merges 直接从 GGUF 读取;GPT-2 的 byte↔unicode 映射;用优先队列加双向链表做 O(n log n) 的 BPE 合并;33 个 added token(类型 3/4)作为字面量先匹配;增量 UTF-8 反分词(流式输出不会出现被截断的字符)。
- `src/pretok.mm`:NFC 归一化和 Qwen 的 pre-tokenizer 正则(带 `\p{L}\p{M}\p{N}` 的 ICU 正则,通过 Foundation 的 `NSRegularExpression` 调用)。这是整个工程里唯一的 Objective-C++ 文件。
- `src/chat.{h,cpp}`:Qwen3.8 chat template 中引擎需要的部分(system/user/assistant、`enable_thinking`、`reasoning_effort` 的 xhigh/medium/low、`<think>` 生成提示),以及采样器(llama.cpp 的默认链:top-k → top-p → min-p → temperature → 抽样;temperature≤0 为贪心)。
- `src/cli.cpp`:`kata run`(单轮)、`kata chat`(多轮,复用 KV 缓存和 GDN 状态,每轮只 prefill 新增后缀)、`kata tokenize`。
- 短后缀(≤24 个 token)用逐 token 的 decode 而不是整套 prefill(prefill 有每次约 0.4~0.6 秒的固定权重展开开销)。

## 验证

- **分词器与 `llama-tokenize` 在 24 个输入上逐 id 完全一致**:14 个手写用例(散文、代码、中日韩、emoji 与 ZWJ、数字与全角/上标/带圈数字、空白与 CRLF、缩写、JSON、特殊 token 与 `<think>`/`<tool_call>`)和 10 个真实文件(4 万 token 的中英混合文档、Python、C++、Metal 源码、Makefile)。注意比较时要给 `llama-tokenize` 加 `--no-escape`(它默认展开 `\n` 等转义序列)。
- 端到端:`"What is the capital of France?"` → `The capital of France is Paris.`;thinking 模式给出推理链和正确答案(17×24=408);两轮对话中第二轮能正确回忆第一轮的信息,证明 KV 与 GDN 状态的复用是正确的。

## 已知的限制

- 第一次调用时 MPP 的 GEMM pipeline 要编译约 2 秒(`prefill` 的第一次耗时因此偏高);之后用 binary archive 或在加载时并行预创建可以消除。
- 默认走精确 fp16 路径;`--fast` 打开 int8 路径,在更大样本上通过质量门槛之前不设为默认。
- 不支持 tools/function calling 的模板渲染;agent 场景请用 `--raw` 传入已经渲染好的提示。
- 采样在 CPU 上做,每个 token 要把 1MB 的 logits 读回(约 0.3~0.5 ms),之后会搬到 GPU 上。
