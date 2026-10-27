> 本文以中文写成;下方 `## Summary (EN)` 是英文摘要,双语文档索引见 [docs/README.md](../README.md)。
> *This document is in Chinese. An English summary follows; the bilingual index is at [docs/README.md](../README.md).*

## Summary (EN)

**Goal.** Pin every version, verify the model, build the reference runtime, and lay down the skeleton the rest of the work would measure against.

**What was pinned** (see `VERSIONS`): macOS 27.0.1, Xcode 27.1, Metal compiler 32023.921 (language version 4.1), metal-cpp `8cdde68`, Prism llama.cpp `b3ebc47`; the model `mannix/Bonsai-2-27B:mtp-ptq1_0`, 6,397,969,920 bytes, SHA-256 `174c9c65…` matching the Ollama blob. The reference runtime and the model both live outside the repository.

**What was built.** `tools/ref_logits`: links Prism's libllama and dumps per-position logits, per-layer residual outputs and per-layer norm/attention/FFN outputs for a given token list, which is what every later numerical claim was checked against. The skeleton: plain `make`, `config/kata_m6.h` shared by C++ and MSL, vendored `third_party/metal-cpp`, offline Metal toolchain verified. `docs/MODEL_SPEC.md` was written from the Prism sources in this milestone.

**Baseline on this machine** (`llama-bench -ngl 99 -r 3`): **pp512 208.08 ± 7.12 tok/s, tg128 15.64 ± 0.06 tok/s**.

---

# 里程碑 M0:基线、参考工具与项目骨架

日期:2026-10-24

## 做了什么

1. **固定并核对最新版本**(见根目录 `VERSIONS`):macOS 27.0.1、Xcode 27.1、Metal 编译器 32023.921(语言版本上限 4.1)、metal-cpp(`apple/metal-cpp` main,macOS 27.0.1)、Prism llama.cpp `b3ebc47`(2026-10-24 的最新提交,比上次的 `73bfe6cc` 新)。
2. **下载并校验模型**:`mannix/Bonsai-2-27B:mtp-ptq1_0`,6,397,969,920 字节,SHA-256 与 Ollama 的 blob 名一致(`174c9c65…`)。存放在仓库外 `../../models/`。
3. **编译参考运行时**:Prism llama.cpp(Metal,Release),存放在仓库外 `../../refs/llama.cpp-prism`。
4. **参考工具** `tools/ref_logits.cpp`:链接 Prism 的 libllama,对给定 token id 或文本导出每个位置的 logits、每层残差输出(`l_out-<il>`)、各层 norm/attention/FFN 输出,并支持贪心续写。构建:`make ref_logits`。
5. **项目骨架**:`Makefile`(纯 make)、`config/kata_m6.h`(C++ 与 MSL 共用的编译期常量)、`third_party/metal-cpp`(固定版本)、`shaders/`、`src/`。已验证离线 Metal 工具链(`metal -std=metal4.1` → `metallib`)和共享头在 MSL 里能编译、kernel 在 GPU 上能运行。
6. **模型规格** `docs/MODEL_SPEC.md`:从 Prism 源码核实的精确公式(GDN 递推、attention、Hadamard 的顺序与置换、embedding 反向变换)。

## 基线(本机实测)

`llama-bench -ngl 99 -r 3`,Prism llama.cpp `b3ebc47`,模型 `Ternary-Bonsai-2-27B-PTQ1_0-MTP`:

| 测试 | 吞吐 |
|---|---|
| **pp512** | **208.08 ± 7.12 tok/s** |
| **tg128** | **15.64 ± 0.06 tok/s** |

- decode 的有效带宽约 15.64 × 5.66GB = 88.5GB/s,只有持续带宽(142GB/s)的 62%。
- 参考输出示例:`The capital of France is` → ` Paris.\nThe capital of Germany is Berlin.\nThe`,token id `760 6511 314 9338 369`,最后位置 top-1 是 `11751`(logit 14.7327)。

## 目标(相对基线)

- decode:纯自回归约 25 tok/s(+60%),加 MTP 约 36~40 tok/s(+130%~+155%)。
- prefill:int8 路径约 440~590 tok/s(2.1~2.8 倍),fp16 精确路径约 160~215 tok/s(与基线持平,所以精确路径只在质量需要时用)。

## 下一步

M1:GGUF 加载器与权重绑定、嵌入查表、三值 GEMV 的正确性内核,逐步对照 `ref_logits` 的逐层输出。
