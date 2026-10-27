# Kata

[English](README.md) · **简体中文**

![Kata — Bonsai-2-27B 的 Metal 4 推理引擎](docs/assets/social-preview.png)

从零手写的 **Metal 推理引擎**,服务于 [Bonsai-2-27B](https://huggingface.co/prism-ml) —— 一个 27B 混合注意力模型,权重以
**1.58 bit** 存储(三值 ±1 + fp16 分组 scale)—— 完整跑在一台 **16 GB** 的 Apple 笔记本上。

不用 llama.cpp、不用 MLX、不用 MPS graph:每个内核都是手写的 Metal Shading Language,包括在途的 2-bit 权重解码器、门控
DeltaNet 递推、注意力路径、张量核(`matmul2d`)预填充 GEMM,以及 MTP 投机解码头。

**[Release v0.1.0](https://github.com/wolfjpy/kata/releases/tag/v0.1.0)** · **[工程日志](docs/CHECKPOINT-001.md)**([1](docs/CHECKPOINT-001.md) · [2](docs/CHECKPOINT-002.md) · [3](docs/CHECKPOINT-003.md) · [4](docs/CHECKPOINT-004.md),外加 [`docs/reports/`](docs/reports/) 下的 29 篇里程碑报告)—— 每一个优化决策都附有支撑它的实测数据,包括失败的那些。

| | |
|---|---|
| **权重(磁盘 / 内存)** | 6.4 GB 单个 GGUF(27B 参数:三值权重 + fp16 分组 scale,外加一个 Q8_0 的 MTP 头) |
| **预填充** | 240 tok/s(`pp512`,精确路径)· 227 tok/s(7.3K 提示,`--fast-ffn`) |
| **解码** | 18.9 tok/s 纯自回归 · 25.6 tok/s 开 MTP 投机 · 31.6 tok/s 结构化代码 |
| **上下文** | 32K(fp16 KV ≈ 2.1 GB,int8 KV ≈ 1.1 GB,需显式开启) |
| **硬件** | Apple M6,16 GB 统一内存,Metal 4 |

---

## 亮点

- **三值权重,在内核里解码。** PTQ1_0 把 5 个权重以三进制数字的形式压进 2 字节。GEMV 内核用一张 256 项的"8 宽三值模式"查表,
  一次 dispatch 完成解码 —— 所以内存里**从不存**反量化后的权重副本。
- **两种注意力都在。** 64 层里有 48 层是门控 DeltaNet(卷积状态 + delta 规则递推,每 token 一个内核,状态留在寄存器里);
  另外 16 层是 GQA 注意力,带逐头 QK-norm、部分 RoPE 和 sigmoid 输出门。两条路径在每层的同一个 command buffer 里端到端融合。
- **用模型自带的 MTP 头做投机解码**(blk.64,Q8_0):每步草稿一个 token,与目标前向在**同一个 command buffer** 里验证,
  所以草稿从不被重复付费。
- **前缀缓存。** 见过相同 token 前缀的提示,直接从 `~/.cache/kata` 恢复完整状态(KV、卷积状态、递推状态)而不重新预填充:
  重复的 3.6K token 提示从 18.6 s 降到 0.03 s,且输出逐字节一致。
- **质量门限,不是感觉。** 每个有损开关都用 KL / top-1 工具(`kata kl-eval`)对精确 fp16 路径测量,过不了门限的就**记录为失败**
  而不是发布出去。

## 快速开始

```bash
# 1. 构建(需要 Xcode command line tools;需要 macOS 26+ 且 SDK 含 Metal 4 / MPP)
make            # 产出 build/kata、build/kata.metallib 以及开发工具

# 2. 取模型:单个 GGUF(约 6.4 GB),例如
#    hf download prism-ml/Bonsai-2-27B-gguf bonsai-2-27b-mtp-ptq1_0.gguf --local-dir models/

# 3. 运行
build/kata run --model models/bonsai-2-27b-mtp-ptq1_0.gguf \
                 --prompt "Explain why a delta-rule recurrence is cheaper than attention at inference." \
                 --max-tokens 256

# 4. 交互式对话(流式输出、思考档位、默认启用前缀缓存)
build/kata chat --model models/bonsai-2-27b-mtp-ptq1_0.gguf
```

## 用法

```
kata run    --model M.gguf --prompt P [--system S] [--raw] [--no-think]
              [--effort xhigh|medium|low] [--temp T] [--top-k K] [--top-p P]
              [--max-tokens N] [--ctx N] [--no-spec] [--no-cache] [--exact]
kata chat   --model M.gguf [same options]
kata kl-eval --model M.gguf --file corpus.txt [--tokens N] [--save-ref R.bin | --ref R.bin]
kata verify-check --model M.gguf --ids "$(cat prompt.ids)"   # 用精确单 token 解码验证 T=2 路径
```

质量 / 性能开关:

| 开关 | 作用 | 质量(top-1 / 平均 KL) |
|---|---|---|
| *(默认)* | 精确 fp16 预填充,MTP 投机解码开启 | 基准 |
| `--fast-ffn` | 预填充阶段 FFN 的 gate/up/down 走 int8:**+10%** 预填充 | 99.87% / 7.5e-5 ✅ |
| `--fast` | 预填充所有矩阵走 int8:+11% 预填充 | 98.48% / 0.0026 ❌ |
| `--kv int8` | int8 KV 缓存:KV 内存减半,长上下文解码 +14% | 99.51% / scale 粒度粗 ⚠️ |
| `--exact` | 强制关闭所有有损路径 | 基准 |
| `--no-spec` | 关闭 MTP 投机解码 | 输出完全一致 |

## 性能

### 测试机器

| | |
|---|---|
| 机器 | Apple M6,16 GB 统一内存(无风扇笔记本) |
| 系统 / SDK | macOS 27.0.1,Xcode 27.1,SDK MacOSX27.0,Metal 编译器 32023.921(`metal4.1`) |
| 模型 | `Bonsai-2-27B` PTQ1_0 + MTP 头,GGUF 6,397,969,920 字节,sha256 `174c9c6584e118766d5511e00a84fd8a31d146248915a726cfac8e6d140c6204` |
| Kata 构建 | 本仓库,`build/kata` + `build/kata.metallib` |
| 基线构建 | llama.cpp,Prism 分支,`prism` 分支,commit `b3ebc47c286e58ead5fc6abb3655974d3bcc1355`,用同树的 `llama-bench` |

### Kata vs llama.cpp 参考实现(同机、同 GGUF)

| 测量项 | llama.cpp `b3ebc47` | Kata | 倍数 |
|---|---|---|---|
| 提示处理 512 token(`pp512`) | 211.4 ± 4.8 tok/s | 240.1 tok/s(精确)· 243.4(`--i8 ffn`) | **1.14×** |
| 生成 128 token,纯自回归(`tg128`) | 17.2 ± 0.1 tok/s | 18.9 tok/s | **1.10×** |
| 生成 128 token,Kata 默认(开 MTP 投机) | —(`llama-bench` 无 MTP) | 25.6 tok/s | 1.49× |
| 提示处理,7.3K 提示 | —(未测) | 227 tok/s(`--fast-ffn`) | — |

基线与 Kata 跑的是**同一个** 6.4 GB GGUF、同一台机器、背靠背。`tg128` 那行才是同口径比较(`llama-bench` 是纯自回归,
`build/kata --ids … --gen 128` 也是)。第三行是 Kata 的默认配置,在纯自回归之上叠加了 MTP 投机:
**相对自己的纯自回归路径 1.35×,相对基线 1.49×**。无风扇机器上有 ±5% 的热漂移,建议每项跑三次取中位数。

### Kata 明细(均为 `--temp 0`)

| 工况 | tok/s |
|---|---|
| 预填充 1K token(精确 / `--fast-ffn` / `--fast`) | 249 / 280 / 262 |
| 预填充 7.3K 提示(`--fast-ffn`) | 227 |
| 解码,散文 | 25.4 |
| 解码,推理文本(MTP 草稿接受率 85–90%) | 28–29 |
| 解码,结构化 / 贪心代码 | 31.6 |
| 解码,29K 上下文 | 18.4 |

### 如何复现

```bash
M=models/bonsai-2-27b-mtp-ptq1_0.gguf

# 基线:PP512 与 TG128,重复 2 次(llama-bench 打印的是均值 ± 标准差)
llama-bench -m $M -p 512 -n 128 -r 2
# -> pp512 211.42 ± 4.79 t/s | tg128 17.23 ± 0.11 t/s   (M6, build b3ebc47)

# Kata:同一条 512 token 提示的预填充,以及 128 token 纯自回归
build/kata --model $M --ids "$(cat prompt512.ids)" --prefill --repeat 2
# -> prefill (fp16 exact path) of 512 tokens: 2.132 s = 240.1 tok/s
build/kata --model $M --ids "$(cat prompt512.ids)" --gen 128
# -> 18.94 tok/s (52.8 ms/token)

# Kata 开投机(CLI 的默认配置)
build/kata run --model $M --raw --file prompt512.txt --temp 0 --max-tokens 128 --no-cache
# -> [prefill ... | decode 128 tokens in 5.00 s = 25.6 tok/s]
```

解码受限于**每 token 要流式读取 6.4 GB 权重**:在 25–31 tok/s 时,引擎维持大约 160–200 GB/s 的有效权重带宽,这基本就是
这个模型在笔记本内存系统上的 roofline。预填充受限于**解码 2-bit 权重的 ALU 成本**,而不是带宽(`--fast-ffn` 之所以有效,
是因为它削减了 ALU 工作量,而不是字节数)。

## 质量门限

有损开关只有在 3072 个位置上,相对精确 fp16 路径**top-1 ≥ 99.7%** 且 **KL ≤ 1e-3** 才被接受:

```bash
# 精确 vs int8 预填充(单进程内跑两条路径)
build/kata kl-eval --model $M --file corpus.txt --tokens 3072 --keep 256 --chunk 512

# 会改变缓冲区布局的配置(例如 int8 KV)必须分两个进程
KATA_KV=fp16 build/kata kl-eval --model $M --file corpus.txt --save-ref ref.bin
KATA_KV=int8 build/kata kl-eval --model $M --file corpus.txt --ref ref.bin
```

当前结果:`--fast-ffn` 99.87% / 7.5e-5 ✅ · `--fast` 98.48% / 2.6e-3 ❌ · `--kv int8` **99.51% top-1** ——
一个 (position, KV head) 只有 1 个 scale 覆盖 256 个元素,一个离群值就吃掉 1–2 个有效 bit。非对称(mean + scale)量化
实现后被实测**更差**(99.37%),因为单头内的值本来就近似零对称;所以它保持 opt-in、不做默认。(它的平均 KL 看起来很吓人,
但那纯粹是因为少数位置的参考 logits 在 1e6 量级:在那种尺度上 0.5% 的相对误差就是 5000 nat 的偏移,KL 已失去意义;
`KATA_KL_DBG=1` 会打印这些位置。)
投机解码与前缀缓存**按构造即精确**,并已验证输出逐字节一致。

## 实现概要

```
x = embed(token)
for il in 0..63:
    h = rmsnorm(x, attn_norm[il])
    a = gated_delta_net(h)        if il % 4 != 3
      = gqa_attention(h)          if il % 4 == 3     # 16 层,逐头 QK-norm、部分 RoPE、sigmoid 门
    x = x + a
    h = rmsnorm(x, post_attention_norm[il])
    x = x + ffn_down(silu(ffn_gate(h)) * ffn_up(h))
logits = lm_head(rmsnorm(x, output_norm))
```

每个三值矩阵乘的输入都先做一次快速 Walsh–Hadamard 变换(量化基),并融合进 RMSNorm 内核。每层一个 command buffer,
让解码 token 的约 50 次 dispatch 在途而不产生 CPU 往返;MTP 草稿、目标前向与验证共用这个 buffer。

仓库结构:

```
src/        引擎(模型、层、内核调度)、会话、采样器、分词器、chat 模板、CLI
shaders/    Metal 内核:三值 GEMV/GEMM、GDN、注意力、norm、MPP matmul、量化
tools/      参考 logits 导出、内核微基准、质量工具
docs/       模型规格、逐里程碑报告、检查点
```

## 哪些做法失败了(以及为什么写进文档)

这个引擎有一份**负结果的记录**,因为那才是最贵的部分:

- **Turbo-fish 0.5-bit 与 2:4 稀疏**:两者都需要不同的权重布局;在这个模型上实测比 1.58-bit 布局更慢。
- **更深的 MTP 链**(2–3 个草稿):链式路径上接受率塌陷;即便达到参考接受率,更宽的验证每轮贵 1.57×,而产出只多 1.68 倍的
  token —— 最多只能打平。
- **DFlash2 风格的 7-token 草稿器**:这个移植版里草稿网络的 first-token 一致率只有约 0.4,所以宽窗口还不划算。
- **张量核上的宽验证**:`matmul2d` 路径在 M=16 时只有 9.4 TFLOPS(本芯片峰值的 54%),这意味着 8 行验证的加速上限是 1.8×,
  而不是让深投机划算所需的 1.6×。双行组流水实现过、实测**慢了 1.6×**,已移除。

## 文档

- `docs/README.md` —— 双语文档索引(每篇一句英文 / 中文摘要)
- `docs/MODEL_SPEC.md` —— 架构的**事实基准**(已对照参考实现与逐层 logit 导出验证)
- `docs/CHECKPOINT-00{1..4}.md` —— 阶段总结
- `docs/reports/M*.md` —— 逐项实验报告,每篇都写明了支撑下一步决策的实测数据,包括失败的

## 环境要求

Apple Silicon 上的 macOS,需要 Metal 4(MPP `matmul2d`)、16 GB 统一内存、Xcode command line tools。引擎针对**单一模型**
(Bonsai-2-27B,PTQ1_0 + MTP 头)与**单一机型**做了特化:内核是为该模型的形状(hidden 5120、FFN 17408、24 Q / 4 KV 头、
词表 248320)硬编码的,不是通用实现。

## 参考

本引擎是干净实现,但如果没有下面这些来源,它不可能存在:

**模型与量化格式**
- Bonsai-2-27B(Prism)—— 模型权重、配置与 PTQ1_0 三值量化格式,以及描述该架构的公开白皮书
  (`docs/reference/bonsai-2-27b-whitepaper.pdf`)。
- `docs/MODEL_SPEC.md` —— 对照参考实现与逐层 logit 导出验证过的架构说明。

**参考实现**
- [llama.cpp](https://github.com/ggml-org/llama.cpp)(ggml)—— GGUF 容器格式、整体图结构,以及采样、chat 模板、分词处理的
  参考行为。
- Prism 的 llama.cpp 分支 —— `src/llama-graph.cpp` / `src/llama-model.cpp` 的 Qwen3.5 风格图、PTQ1_0 反量化
  (`prism_ptq1_0_quant_dequant.c`)、Metal 门控 DeltaNet 内核(`prism_metal_gated_delta_net.metal`)以及 T=4 验证图
  (`prism_qwen35_graph.cpp`)。这些被当作**数值规范**阅读,本仓库的内核是独立实现的。
- `third_party/metal-cpp` —— Apple 的 metal-cpp 头文件,MIT,许可文件随目录提供。
- [baseRT](https://github.com/basecompute/baseRT) —— Metal 4 / Metal Performance Primitives 计算路径的参考阅读
  (tensor ops、cooperative tensor、command buffer 结构)。
- [ninfer](https://github.com/Neroued/ninfer) —— 另一个从零手写的 Metal 推理引擎;作为内核结构与"手写 Metal 解码器
  在实践中的代价"的独立参照点。
- [Strata](https://github.com/Niko1221/Strata) —— `shaders/` 所遵循的分层 Metal 内核组织与 dispatch 模式的参考。
- [Qwen3.8-27B-DFlash2-GGUF](https://huggingface.co/incoai/Qwen3.8-27B-DFlash2-GGUF) —— 宽窗口投机实验所用的
  DFlash2 块扩散草稿权重(`docs/reports/R2-dflash2-revisit.md`、`docs/reports/M28c-draft-forward.md`),以及该草稿器
  GGUF 布局的参考。

**Apple / Metal**
- *Metal Shading Language Specification*(Metal 4)—— 语言、`simdgroup_matrix`、`threadgroup` 内存与 barrier 语义的
  权威参考。
- *Metal Performance Primitives*(`MetalPerformancePrimitives/MetalPerformancePrimitives.h`)—— `matmul2d_descriptor`、
  `matmul2d::run`、`cooperative_tensor`、`get_row_reduction_destination_cooperative_tensor`,预填充 GEMM 与注意力内核
  都建立在它们之上。
- *Metal Feature Set Tables* —— 用于确认各 `MTLGPUFamily` / Apple GPU 世代可用的能力。

**算法出处的论文**
- Yang, Kautz, Hatamizadeh,*Gated Delta Networks: Improving Mamba2 with Delta Rule*(2024)—— 门控 DeltaNet 层。
- Schlag, Irie, Schmidhuber,*Linear Transformers Are Secretly Fast Weight Programmers*(2021)—— delta 规则。
- Ainslie 等,*GQA: Training Generalized Multi-Query Transformer Models*(2023)—— 24/4 分组查询注意力。
- Dao 等,*FlashAttention*(2022)—— 预填充内核采用的 tiled + online softmax 注意力结构。
- Leviathan, Kalman, Matias,*Fast Inference from Transformers via Speculative Decoding*(2023)—— 草稿/验证循环及其
  接受判据。
- Hu 等 / DeepSeek-V3(2024)的 MTP 头设计 —— 用作自草稿的多 token 预测头。
- Ashkboos 等,*QuaRot*(2024)、Tseng 等,*QuIP#*(2024)—— 作为三值矩阵乘输入基的随机 Hadamard 变换,以及其背后的
  incoherence 论证。

## 许可

MIT —— 见 [LICENSE](LICENSE)。模型权重由其作者按各自的许可分发,本仓库不再分发它们。
