# Documentation index · 文档索引

**[English](#english) · [简体中文](#简体中文)**

The engine's code and the README are in English; the engineering log — the milestone checkpoints and the 24 experiment
reports — is written in Chinese, in the order it happened. This page gives both audiences a map: an English one-liner
for every document, and the Chinese original right below it.

**Every document in `docs/` now opens with a `## Summary (EN)` block** — goal, the measured numbers, the verdict and the
next step — so you can read the conclusion of any report without Chinese, and only dive into the Chinese body when you
want the details.

> 代码与 README 为英文;工程日志(阶段检查点 + 24 篇实验报告)按其发生顺序以中文写成。本页给两种读者一张地图:
> 每篇文档都有一句英文摘要,下面紧跟中文原文标题。

---

## English

### Start here
| Document | What it is | Language |
|---|---|---|
| [`../README.md`](../README.md) | Project overview, quick start, performance vs llama.cpp, quality gates, references | English |
| [`../README.zh-CN.md`](../README.zh-CN.md) | The same README in Chinese | 中文 |
| [`MODEL_SPEC.md`](MODEL_SPEC.md) · **[`.en.md`](MODEL_SPEC.en.md)** | **Ground truth for the architecture**: every layer, shape, formula and kernel, verified against the reference implementation and per-layer logit dumps | 中文 + **English (full)** |
| [`reference/README.md`](reference/README.md) | Provenance and licence for the upstream source copies kept in `docs/reference/` | English |

### The engineering log (newest thinking last, in the order it happened)

**Phase 1 — getting it correct**
| Document | One line |
|---|---|
| [`reports/M0-baseline-and-skeleton.md`](reports/M0-baseline-and-skeleton.md) · **[.en](reports/M0-baseline-and-skeleton.en.md)** | M0: baseline measurement, the reference tooling, and the project skeleton. |
| [`reports/M2-correct-forward.md`](reports/M2-correct-forward.md) · **[.en](reports/M2-correct-forward.en.md)** | M1/M2: a numerically correct full forward — all 64 layers — checked against the reference logits. |
| [`CHECKPOINT-001.md`](CHECKPOINT-001.md) | Checkpoint 1: the reference sources are read end to end, the architecture is pinned down, the first kernels are written. The longest document here (~25k characters) and the most useful if you want the model's internals. |

**Phase 2 — make it fast**
| Document | One line |
|---|---|
| [`reports/M3-decode-optimization.md`](reports/M3-decode-optimization.md) · **[.en](reports/M3-decode-optimization.en.md)** | M3: decode optimisation — the single-token path becomes bandwidth-bound instead of ALU-bound. |
| [`reports/M5-prefill.md`](reports/M5-prefill.md) · **[.en](reports/M5-prefill.en.md)** | M5: batched prefill, both the exact fp16 path and the first int8 attempt. |
| [`reports/M5b-prefill-optimization.md`](reports/M5b-prefill-optimization.md) | M5b: prefill optimisation — flash attention, GEMM ordering, unrolling, and the first int8 quality evaluation. |
| [`reports/M12-weight-layout.md`](reports/M12-weight-layout.md) · **[.en](reports/M12-weight-layout.en.md)** | M12: 16-byte-wide weight reads. Gate not met, not implemented. |
| [`reports/M13-fast-ffn.md`](reports/M13-fast-ffn.md) · **[.en](reports/M13-fast-ffn.en.md)** | M13: int8 prefill for the FFN gate+up (`--fast-ffn`) — the one lossy switch that passed the gate. |
| [`reports/M11-int8-kv.md`](reports/M11-int8-kv.md) · **[.en](reports/M11-int8-kv.en.md)** | M11: int8 KV cache. Includes the later correction: the original quality claim did not reproduce, the real number is 99.51% top-1, and asymmetric quantisation measured worse. |
| [`reports/M17-small-items.md`](reports/M17-small-items.md) · **[.en](reports/M17-small-items.en.md)** | M17: a batch of small ideas, each measured; none cleared the 1% keep line. |
| [`reports/M26-verification-side.md`](reports/M26-verification-side.md) · **[.en](reports/M26-verification-side.en.md)** | M26: arbitrary-width verification, state snapshots and commit. |
| [`reports/M27a-small-grid-fixes.md`](reports/M27a-small-grid-fixes.md) · **[.en](reports/M27a-small-grid-fixes.en.md)** | M27a: small grid fixes in the verification window — 20% off per round, ceiling raised to +38%. |

**Phase 3 — speculation, and what did not work**
| Document | One line |
|---|---|
| [`reports/M4-mtp-speculative-decoding.md`](reports/M4-mtp-speculative-decoding.md) | M4: MTP speculative decoding at depth 1 with a 2-wide verification. |
| [`reports/M7-speculation-options.md`](reports/M7-speculation-options.md) · **[.en](reports/M7-speculation-options.en.md)** | M7: the other speculation options (DFlash2, the 2B drafter, deeper MTP chains) and the cost of wide verification. |
| [`reports/M10-sampling-aware-draft.md`](reports/M10-sampling-aware-draft.md) · **[.en](reports/M10-sampling-aware-draft.en.md)** | M10: sampling-aware drafts — estimated offline, gate not met, implementation deferred. |
| [`reports/M14-wide-verify-gate.md`](reports/M14-wide-verify-gate.md) · **[.en](reports/M14-wide-verify-gate.en.md)** | M14: the wide-verification microbenchmark gate; it did not pass, so DFlash2 (M15/M16) was closed. |
| [`reports/M25-wide-verify-kernel.md`](reports/M25-wide-verify-kernel.md) · **[.en](reports/M25-wide-verify-kernel.en.md)** | M25: the wide verification kernel that paved the way for DFlash2 — and the ablation showing the MMA is the floor. |
| [`reports/M28a-draft-loading-and-quant.md`](reports/M28a-draft-loading-and-quant.md) · **[.en](reports/M28a-draft-loading-and-quant.en.md)** | M28a: loading the DFlash2 drafter and decoding its Q4_K / Q6_K tensors. |
| [`reports/M28b-draft-module.md`](reports/M28b-draft-module.md) · **[.en](reports/M28b-draft-module.en.md)** | M28b: the draft module — loading, ops, geometry — and the exact to-do list for M28c. |
| [`reports/M28c-draft-forward.md`](reports/M28c-draft-forward.md) · **[.en](reports/M28c-draft-forward.en.md)** | M28c: the draft forward assembled and running end to end; acceptance still 0 (three bugs located, two fixed). Also records the MTP-chain bookkeeping audit. |
| [`reports/R2-dflash2-revisit.md`](reports/R2-dflash2-revisit.md) | R2: DFlash2 revisited — the earlier conclusion was wrong, reopened. |

**Infrastructure**
| Document | One line |
|---|---|
| [`reports/M6-tokenizer-chat-cli.md`](reports/M6-tokenizer-chat-cli.md) · **[.en](reports/M6-tokenizer-chat-cli.en.md)** | M6: tokenizer, chat template, sampling and the command line. |
| [`reports/M9-prefix-cache.md`](reports/M9-prefix-cache.md) · **[.en](reports/M9-prefix-cache.en.md)** | M9: the persistent prefix cache (18.6 s → 0.03 s on a repeated 3.6K-token prompt). |
| [`reports/REVIEW-01-repo-assessment.md`](reports/REVIEW-01-repo-assessment.md) | REVIEW-01: a read-through assessment of the repository, the open issues and the next steps. |
| [`CHECKPOINT-002.md`](CHECKPOINT-002.md) · **[.en](CHECKPOINT-002.en.md)** | Checkpoint 2: state of play after the decode and prefill work. |
| [`CHECKPOINT-003.md`](CHECKPOINT-003.md) · **[.en](CHECKPOINT-003.en.md)** | Checkpoint 3: speculative decoding, the prefix cache and the CLI are in place. |
| [`CHECKPOINT-004.md`](CHECKPOINT-004.md) · **[.en](CHECKPOINT-004.en.md)** | Checkpoint 4: the draft-model line of work, and where it stands. |
| [`../VERSIONS`](../VERSIONS) | The exact toolchain, model hash and baseline commit every number in this repository was measured with. |

### Reading the reports without Chinese
Three layers, cheapest first: the one-liners in the tables above; the `## Summary (EN)` block at the top of each document
(goal, numbers, verdict, next step); and finally the Chinese body, whose tables are numbers and English identifiers
anyway. A three-word vocabulary covers the verdicts: 接受 = accept, 否决 = reject, 闸门未达到 = gate not met.

---

## 简体中文

### 从这里开始
| 文档 | 内容 | 语言 |
|---|---|---|
| [`../README.md`](../README.md) | 项目总览、快速开始、与 llama.cpp 的性能对比、质量门限、参考来源 | English |
| [`../README.zh-CN.md`](../README.zh-CN.md) | 同一份 README 的中文版 | 中文 |
| [`MODEL_SPEC.md`](MODEL_SPEC.md) · **[`.en.md`](MODEL_SPEC.en.md)** | **架构的事实基准**:每一层、每个形状、公式与内核,已对照参考实现与逐层 logit 导出验证 | 中文 + **英文全译** |
| [`reference/README.md`](reference/README.md) | `docs/reference/` 中上游源码副本的来源与许可 | English |

### 工程日志(按发生顺序,最新的思考在最后)

**第一阶段 —— 先做对**
| 文档 | 一句话 |
|---|---|
| [`reports/M0-baseline-and-skeleton.md`](reports/M0-baseline-and-skeleton.md) · **[.en](reports/M0-baseline-and-skeleton.en.md)** | M0:基线测量、参考工具链与项目骨架。 |
| [`reports/M2-correct-forward.md`](reports/M2-correct-forward.md) · **[.en](reports/M2-correct-forward.en.md)** | M1/M2:数值正确的完整前向(全部 64 层),与参考 logits 对齐。 |
| [`CHECKPOINT-001.md`](CHECKPOINT-001.md) | 检查点 1:通读参考源码、钉死架构、写出第一批内核。全仓库最长的一篇(约 2.5 万字),想了解模型内部结构看它。 |

**第二阶段 —— 做快**
| 文档 | 一句话 |
|---|---|
| [`reports/M3-decode-optimization.md`](reports/M3-decode-optimization.md) · **[.en](reports/M3-decode-optimization.en.md)** | M3:解码优化 —— 单 token 路径从 ALU 受限转为带宽受限。 |
| [`reports/M5-prefill.md`](reports/M5-prefill.md) · **[.en](reports/M5-prefill.en.md)** | M5:批量预填充,精确 fp16 路径与第一次 int8 尝试。 |
| [`reports/M5b-prefill-optimization.md`](reports/M5b-prefill-optimization.md) | M5b:预填充优化 —— flash attention、GEMM 顺序、展开,以及第一次 int8 质量评估。 |
| [`reports/M12-weight-layout.md`](reports/M12-weight-layout.md) · **[.en](reports/M12-weight-layout.en.md)** | M12:16 字节宽权重读取。闸门未达到,不实现。 |
| [`reports/M13-fast-ffn.md`](reports/M13-fast-ffn.md) · **[.en](reports/M13-fast-ffn.en.md)** | M13:FFN gate+up 的 int8 预填充(`--fast-ffn`)—— 唯一过了门限的有损开关。 |
| [`reports/M11-int8-kv.md`](reports/M11-int8-kv.md) · **[.en](reports/M11-int8-kv.en.md)** | M11:int8 KV 缓存。含后期更正:最初的画质结论无法复现,真实值是 top-1 99.51%,而非对称量化实测更差。 |
| [`reports/M17-small-items.md`](reports/M17-small-items.md) · **[.en](reports/M17-small-items.en.md)** | M17:一批小想法,逐个测量,没有一个达到 1% 的保留线。 |
| [`reports/M26-verification-side.md`](reports/M26-verification-side.md) · **[.en](reports/M26-verification-side.en.md)** | M26:任意宽度验证、状态快照与提交。 |
| [`reports/M27a-small-grid-fixes.md`](reports/M27a-small-grid-fixes.md) · **[.en](reports/M27a-small-grid-fixes.en.md)** | M27a:验证窗口的小网格修复 —— 每轮省 20%,上限抬到 +38%。 |

**第三阶段 —— 投机,以及失败的部分**
| 文档 | 一句话 |
|---|---|
| [`reports/M4-mtp-speculative-decoding.md`](reports/M4-mtp-speculative-decoding.md) | M4:MTP 投机解码,深度 1、验证宽度 2。 |
| [`reports/M7-speculation-options.md`](reports/M7-speculation-options.md) · **[.en](reports/M7-speculation-options.en.md)** | M7:其它投机方案(DFlash2、2B 草稿器、更深的 MTP 链)与宽验证的代价。 |
| [`reports/M10-sampling-aware-draft.md`](reports/M10-sampling-aware-draft.md) · **[.en](reports/M10-sampling-aware-draft.en.md)** | M10:采样感知草稿 —— 离线估计,闸门未达到,实现推迟。 |
| [`reports/M14-wide-verify-gate.md`](reports/M14-wide-verify-gate.md) · **[.en](reports/M14-wide-verify-gate.en.md)** | M14:宽验证微基准闸门未通过,DFlash2(M15/M16)结案。 |
| [`reports/M25-wide-verify-kernel.md`](reports/M25-wide-verify-kernel.md) · **[.en](reports/M25-wide-verify-kernel.en.md)** | M25:为 DFlash2 铺路的宽验证内核 —— 以及证明 MMA 是硬底的消融。 |
| [`reports/M28a-draft-loading-and-quant.md`](reports/M28a-draft-loading-and-quant.md) · **[.en](reports/M28a-draft-loading-and-quant.en.md)** | M28a:DFlash2 草稿的加载与 Q4_K / Q6_K 张量解码。 |
| [`reports/M28b-draft-module.md`](reports/M28b-draft-module.md) · **[.en](reports/M28b-draft-module.en.md)** | M28b:草稿模块(加载 + 算子 + 几何)与 M28c 的精确待办。 |
| [`reports/M28c-draft-forward.md`](reports/M28c-draft-forward.md) · **[.en](reports/M28c-draft-forward.en.md)** | M28c:草稿前向装配完成、端到端跑通;接受率仍为 0(定位 3 个 bug、修好 2 个)。同时记录了 MTP 链式的记账审计。 |
| [`reports/R2-dflash2-revisit.md`](reports/R2-dflash2-revisit.md) | R2:重新研究 DFlash2 —— 上一轮结论错了,重开。 |

**基础设施**
| 文档 | 一句话 |
|---|---|
| [`reports/M6-tokenizer-chat-cli.md`](reports/M6-tokenizer-chat-cli.md) · **[.en](reports/M6-tokenizer-chat-cli.en.md)** | M6:分词器、chat template、采样与命令行。 |
| [`reports/M9-prefix-cache.md`](reports/M9-prefix-cache.md) · **[.en](reports/M9-prefix-cache.en.md)** | M9:持久化前缀缓存(重复的 3.6K 提示从 18.6 s 降到 0.03 s)。 |
| [`reports/REVIEW-01-repo-assessment.md`](reports/REVIEW-01-repo-assessment.md) | REVIEW-01:仓库通读评估、遗留问题与下一步方案。 |
| [`CHECKPOINT-002.md`](CHECKPOINT-002.md) · **[.en](CHECKPOINT-002.en.md)** | 检查点 2:解码与预填充工作之后的进展。 |
| [`CHECKPOINT-003.md`](CHECKPOINT-003.md) · **[.en](CHECKPOINT-003.en.md)** | 检查点 3:投机解码、前缀缓存与 CLI 就位。 |
| [`CHECKPOINT-004.md`](CHECKPOINT-004.md) · **[.en](CHECKPOINT-004.en.md)** | 检查点 4:草稿模型这条线的现状。 |
| [`../VERSIONS`](../VERSIONS) | 本仓库每个数字的测量环境:工具链版本、模型哈希、基线 commit。 |
