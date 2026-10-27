> 本文以中文写成;**英文全译:[CHECKPOINT-004.en.md](CHECKPOINT-004.en.md)**,下方 `## Summary (EN)` 为摘要,双语文档索引见 [docs/README.md](README.md)。
> *This document is in Chinese. An English summary follows; the bilingual index is at [docs/README.md](README.md).*

## Summary (EN)

**What this is.** The fourth checkpoint (2026-10-25), written after DFlash2 was reopened: the progress of M25-M26 and what
follows. Self-contained, like the others.

**The four decisions it records** (from `R2` §7): (1) accept `r <= 1.6` as the condition to continue with DFlash2; (2) go
straight to the Q4_K_M drafter, no Q8 intermediate; (3) ignore the memory budget — 16 GB of unified memory and 256 GB of
SSD exist, use them; (4) accept `r ~ 2` as the lower-bound line rather than treating it as a failure.

**Working rules it fixes for the rest of the project:** documents and reports in Chinese, commit messages in English, one
commit per milestone with its report attached, only ever commit self-generated files (never the PDFs, PNGs or notes the
owner keeps in the repository root), lossy switches must clear the quality gate and stay off by default, and timing must
be interleaved A/B because the machine throttles by ~10% after a minute of sustained load.

**Contents.** The recovery guide, then a table of what this round completed (the wide verification kernel at r = 2.47, the
verification side with snapshots and commit, and the small-grid fixes that took a round from 177 ms to 141 ms), and the
follow-on plan.

---

# CHECKPOINT-004:DFlash2 重开后的进展(M25–M26)与后续

日期:2026-10-25 · 自包含的恢复文档:**重启后只读本文件(以及它指向的两份文档)就能继续**。
前序:`CHECKPOINT-003.md`(M8–M17 的结果)、`reports/R2-dflash2-revisit.md`(推翻 M14 判断、重开计划)、`reports/M25-wide-verify-kernel.md`、`reports/M26-verification-side.md`。

## 0. 恢复指南

- 项目:`<repo root>`;模型 `../../models/bonsai-2-27b-mtp-ptq1_0.gguf`;构建 `make`(约 6 s);参考仓库 `../../refs/{Strata,ninfer,llama.cpp-prism}`;草稿权重 `../../models/drafters/{dflash2-q4km.gguf,bonsai2-2b-ptq1.gguf}`。
- 用户本轮的四个决定(R2 §7):① 接受 r ≤ 1.6 作为继续条件;② 直接上 **Q4_K_M** 草稿,不走 Q8 中间态;③ **不限内存**,能跑多快就多快(16 GB UMA / 256 GB SSD);④ 接受 r≈2 的下界线。
- 工作规则:回复中文;文档/报告中文;**提交信息只用英文**;每个里程碑独立提交并附报告;只提交自己生成的文件(根目录用户的 PDF/PNG/笔记绝不提交);有损开关必须过质量门限且默认关闭;测速必须交错 A/B(连续满载一分钟后会降频 ~10%)。

## 1. 本轮完成什么

| 里程碑 | 状态 | 关键数字 |
|---|---|---|
| **M25** 宽验证内核 | ✅ 提交 `b1f984b` | ffn_gate:420 µs(T=16)/170 µs(T=2)= **r 2.47**(目标 1.6);数值误差 3e-6 |
| **M26** 验证侧 | ✅ 提交 `fc95985` | `--vwidth 8` **逐 token 复现普通贪心**;每轮 156 ms(T=2)→ 167 ms(T=8),**几乎与 T 无关** |

新增代码:`shaders/wide.metal`(参数化 `k_wide`,双缓冲/消融开关,`k_wide_hyb`)、`shaders/prefill.metal` 的 `k_gdn_rec_snap`+`k_gdn_conv_snap`、`Engine::verifyWide/commitWide/dWide`、`Engine::kMaxVerify=8`、测试入口 `--vwidth T`、`tools/bench_wide.cpp`。

## 2. M25:宽验证内核的结论(为什么是 2.47 而不是 1.2)

- 每个 simdgroup 独占 N=16 行、无 threadgroup 屏障,把权重经 16 字节三值 LUT 解码进私有 fp16 tile(列主序、跨度 KC+8、scale 已乘、精确),再 `matmul2d(M=16,N=16,K=128)`。
- **试过并否掉**:K=256/512(threadgroup 内存翻倍 → 占用率掉到 50%,更慢)、双缓冲(639 µs,更慢)、ALU+矩阵单元混合(473 µs,无重叠收益)、形状扫描(N=8/24/32/64 都更差)。
- 硬限制:**M=16 时矩阵单元的吃入率只有峰值的 53%**(MMA 单独 306 µs = 291 G 权重/s);而生产预填充 GEMM(M=32/64、一次调用跑完整个 K)能到 14.7–16.6 TFLOPS。差距在"小 M + 每次调用只覆盖 K=128"。
- 外部对照:ninfer 在 5090 上 T=8 是单步的 1.55×(≈ 我们的 r 1.35),我们比它慢 1.8 倍。
- 唯一没试的方向:**B 操作数从寄存器喂 `matmul2d`**(`get_right_input_cooperative_tensor`),需要按 lane 的元素映射重排权重文件。

## 3. M26:验证侧的结论

- 每轮成本(72 token 上下文,24 轮平均):T=2 156 ms、T=4 159、T=6 161、T=8 **167 ms**。成本平,说明"权重只读一遍"的设计成立。
- T=8 分解:宽 GEMV 126 ms + 注意力 10.3 + alpha/beta 6.0 + 其余 ~25(每层 2 次 norm、3 次 FWHT、conv/qk/gates、argmax)。
- 后两项是**小网格延迟**:72 token 时注意力只有 4 个 threadgroup,alpha/beta 的 GEMM 只有 2 个。预计可省 ~20 ms(→147 ms)。

## 4. 修正后的 DFlash2 收益(用实测数字)

DFlash2 K=7 的实测接受长度(Prism harness,Kata 目标模型):代码/推理 **5.20** token/轮,散文 **2.07**。

| 配置 | 一轮 | 代码/推理 | 散文 |
|---|---|---|---|
| 现在(K=7,167+9 ms) | 176 ms | **29.5 tok/s**(+18%) | 11.8(−53%) |
| 修掉小网格开销 | 156 ms | 33.3(+33%) | 13.3 |
| 宽内核若到 r=1.6 | 115 ms | 49(+96%) | 19.5 |

结论:DFlash2 目前的价值是**代码/推理 +18~33%**,散文必须靠 M29 宽度自适应回退。这与 R2 的乐观估计(48–59 tok/s)有差距,原因是宽内核落在下界线 + 每轮固定开销实测 41 ms(估的 16 ms)。

## 5. 下一步(按顺序,都带验收标准)

| 里程碑 | 内容 | 验收 |
|---|---|---|
| **M27a** ✅ 完成(`reports/M27a-small-grid-fixes.md`) | 三处修复:alpha/beta 改 T 行 GEMV、注意力改 key 分段、状态快照改记录+重放(省 1.2 GB) | **T=8 每轮 142 ms(原 177,−20%)**;逐 token 复现贪心;DFlash2 收益上限升到 **+38%** |
| **M27b** | MTP 链草稿(K=3 → T=4,零新模型):用 MTP 块自己的输出 hidden 作为下一行的输入,MTP 块内串行 3 次 | 推理文本 ≥ 3.1 token/轮(Prism 实测值);端到端 ≥ 28 tok/s,`verifyWide` 路径不变 |
| **M28a** ✅ 完成(`reports/M28a-draft-loading-and-quant.md`) | 草稿加载 + Q4_K/Q6_K 解码(与 ggml **逐位一致**,258 048 元素差 0.0)+ 草稿矩阵乘(Q4_K 78–118 GB/s) | 草稿一轮约 21 ms(Q6_K 向量化后 15 ms);DFlash2 收益修正为 **+28~32%** |
| **M28b** ✅ 完成(`reports/M28b-draft-module.md`) | Q6_K 向量化(17→73.5 GB/s)+ 草稿全部算子(conv/qk/attn/ctx_kv/矩阵乘)+ 加载器与环形缓冲(80 MB) | 草稿一轮 **14 ms**;缺口只剩"装配 5 层前向 + 选择器 + spec 集成" |
| **M28c** 🔶 进行中(`reports/M28c-draft-forward.md`,已修 **5** 个 bug) | 前向装配完成、端到端跑通(tap 采集 / commit / materialise / 5 层 / 共享头),修掉 2 个装配 bug(norm 绑错 buffer、Q6_K dtype),NaN 已消除;接受率仍 0/7 | 草稿换成 Kata 专用:`kata2-df2-ft5-Q2_K`(常规基,Q2_K/Q3_K)与 `Kata2-ternary`(2-bit prism + 自带 Hadamard 符号,未开始);`k_draft_mm` 支持四种量化。**已修 9 个 bug**(delta 索引、RoPE 约定 = MROMA sections 和 = n_rot/2 → 全 128 维旋转);**每个算子已单独验证通过**(矩阵乘 vs ggml、卷积/注意力/fc vs CPU 独立实现、row 0 自洽);接受率 ~5%,问题定位到**结构假设**而非算子 |:5 层草稿(plain RMSNorm、QKV、逐头 qk norm、scalar RoPE、非因果滑窗 2048 环形 KV、双抽头动态分组卷积、SwiGLU)+ 目标第 5/19/33/47/61 层残差作条件 + 上下文物化器 + top-16/rank-256 选择器 + **Q4_K_M 解量化**;先只支持 K=7 | 接受长度与 llama.cpp `draft-dflash` 一致(±0.3);代码/推理 ≥ 29 tok/s(现在)、≥ 33(修完 M27a);散文不回归(需要 M29) |
| **M29** | 宽度自适应:在线 EMA 学每个宽度的 token/ms,散文自动退回 MTP-1 | 散文 ≥ 24 tok/s;代码/推理不回退 |
| **M30** | (可选)宽内核的寄存器 B 操作数 + 权重重排 | r ≤ 2.0 → 代码 +40% |

**决策点(需要用户)**:M28 是 3–5 天的移植,换来代码/推理 +18~33%(散文先靠 M29 兜底)。建议先做 M27a(半天,把上限抬到 +33%)再决定是否进 M28;也可以直接进 M28。

## 6. 已知的坑(这一轮踩过,记下来)

1. **`snapStride` 单位**:内核里指针是 `float*`,步长必须传**元素数**;传字节会 4 倍越界写,表现为 SIGSEGV(shared 内存会写坏宿主堆)。
2. **`k_gdn_ab_b` 的 alpha/beta 是两个独立缓冲区**(步长 48),不是 pAB 布局;verify 路径要用 prefill 的 `dGemm(wabAll→pAB)`。
3. **`dWide` 的网格**:`k_wide` 每个 threadgroup 是 4 simdgroup × 16 行 = **64 行**,不是 16 行。
4. **`prefill` 不重置状态**:在同一个进程里做多组实验前必须 `resetState()`,否则上一组的状态会污染下一组(这是我第一次测 `--vwidth` 出错的原因)。
5. **stdout 缓冲**:崩溃时缓冲的输出会丢失,调 GPU 代码时用 `setvbuf(_IONBF)` 或每行 fflush(main 已经设了无缓冲)。
6. 微基准在依赖链上不可靠(同形状两次差 3.8 倍、超过理论峰值的读数);以真实内核 + 真实张量为准。

## 7. 当前性能基线(不要忘)

| 指标 | 数值 |
|---|---|
| 解码,普通 | 19.8–20.0 tok/s(50.1 ms/token) |
| 解码,MTP 深度 1 | 25.0(贪心)/ 22.7(采样)tok/s |
| 解码,29K 上下文 | 13.7(fp16 KV)/ 15.6(int8 KV)tok/s |
| 预填充,精确 | 255–263 tok/s(1024 chunk)、207–213(7.3K) |
| 预填充,`--fast-ffn` | 274–280、226.5 tok/s |
| 前缀缓存 | 2.9K 系统提示:15 s → 0.03 s 恢复 + 21 token |
| 宽验证一轮 | T=2 156 ms / T=8 167 ms |

## 8. 环境与版本(2026-10-25 核对)

macOS 27.0.1、Xcode 27.1、Metal 编译器 32023.921(metal4.1 上限)、metal-cpp `8cdde68`、Prism llama.cpp `b3ebc47`(prism 分支 head `dc21b2c`,新提交全是 CUDA/x86,`ggml-metal` 无变化)、Strata `4b9e05a`、ninfer `64d10fb`。详见 `VERSIONS`。
