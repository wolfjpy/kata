> 本文以中文写成;**英文全译:[M28a-draft-loading-and-quant.en.md](M28a-draft-loading-and-quant.en.md)**,下方 `## Summary (EN)` 为摘要,双语文档索引见 [docs/README.md](../README.md)。
> *This document is in Chinese. An English summary follows; the bilingual index is at [docs/README.md](../README.md).*

## Summary (EN)

**Goal.** Step one of the DFlash2 port: load the drafter and decode its Q4_K / Q6_K tensors, in preparation for the 5-layer forward and the selector. Weights are the existing `incoai/Qwen3.8-27B-DFlash2-GGUF` `Q4_K_M` file.

**How much work is the port, really?** In `refs/llama.cpp-prism` the DFlash2 draft model is `src/models/dflash.cpp` and there is **no DFlash-specific Metal kernel** — it is an ordinary 5-layer model running through the normal ggml ops. So the port is loading plus a handful of conventional ops, not a new architecture.

**Geometry, read from the GGUF metadata:**

```
dflash.block_count = 5            dflash.block_size = 8            (W=8, K=7)
dflash.embedding_length = 5120    dflash.feed_forward_length = 17408
dflash.attention.head_count = 32  head_count_kv = 8  key_length = 128  causal = False
dflash.attention.sliding_window = 2048, pattern = [T,T,T,T,T]
dflash.conv_kernel_size = 2, conv_group_size = 16
dflash.selector_rank = 256, selector_top_k = 16
```

**Scope.** Q4_K and Q6_K dequantisation kernels, the matvec path for the draft model, and the tensor-name mapping from GGUF — no speculative machinery yet.

---

# M28a:DFlash2 草稿的加载与 Q4_K / Q6_K 解码

日期:2026-10-25 · M28 的第一步(加载 + 量化解码 + 草稿矩阵乘),为后续 5 层前向和选择器铺路。权重用现成文件:`incoai/Qwen3.8-27B-DFlash2-GGUF` 的 `Q4_K_M`(本机 `../../models/drafters/dflash2-q4km.gguf`)。

## 1. 先弄清"移植"到底是多少工作

`refs/llama.cpp-prism` 里 DFlash2 草稿是 `src/models/dflash.cpp`,**没有 DFlash 专用 Metal 内核**——它就是一个普通的 5 层小模型,走常规 ggml 算子。所以需要的是加载 + 几个常规算子,不是逆向一个新运行时。

GGUF 元数据(实测):

```
dflash.block_count = 5            dflash.block_size = 8            (W=8, K=7)
dflash.embedding_length = 5120    dflash.feed_forward_length = 17408
dflash.attention.head_count = 32  head_count_kv = 8  key_length = 128  causal = False
dflash.attention.sliding_window = 2048, pattern = [T,T,T,T,T]
dflash.conv_kernel_size = 2, conv_group_size = 16
dflash.selector_rank = 256, selector_top_k = 16
dflash.target_layers = [6, 20, 34, 48, 62]        tokenizer.ggml.mask_token_id = 248070
```

81 个张量:**45 Q4_K + 4 Q6_K + 32 F32**,共 1132 MB(Q4_K 976 + Q6_K 155 + F32 1.1)。Q6_K 只出现在 `blk.2/blk.4` 的 `attn_v` 与 `ffn_down`。每层 = attn_conv_base/proj、q/k/v/output、q/k norm、attn_norm、ffn_conv_base/proj、gate/up/down、ffn_norm;外加 `fc.weight`(25600→5120,把 5 个目标残差拼起来投影)、两个 `output_norm`、`selector_*`。

## 2. 完成的部分(都有数值对照)

**(1)GGUF 加载**:现有加载器增加 Q4_K(144 B/256)与 Q6_K(210 B/256)的字节数,张量类型/维度/偏移本来就有。

**(2)Q4_K / Q6_K 解码**(`shaders/draft.metal`):与 **ggml 自己的 `dequantize_row_q4_K/q6_K` 逐位一致**——`tools/dq_probe` 在 8 个真实张量 × 3 行 × 全部 K 块上比较,**258 048 个元素,最大差 0.0**。过程中修掉三个我自己引入的错误:每 64 组只消耗 32 字节(我写成 32 组)、一个 64 组内低/高半字节是**两个不同**的 (scale, min)、以及最高字节的低半字节需要 `& 0x0F`。

**(3)草稿矩阵乘 `k_draft_mm`**:复用 `k_wide` 的"解码进私有 fp16 tile → 单 simdgroup `matmul2d`"结构,按 MT=8 个 block 列、N=16 行/ simdgroup、KC=128、4 simdgroup 实例化;Q4_K 每次处理 16 字节(= 16 低半字节 + 16 高半字节的 32 个权重),Q6_K 先按元素解码(正确但慢)。

真实张量上的正确性与速度(`tools/bench_draft`,行 0 与 CPU 反量化的点积对照):

| 张量 | 类型 | 大小 | 时间 | 带宽 |
|---|---|---|---|---|
| ffn_gate 5120→17408 | Q4_K | 50 MB | 0.64 ms | **78.5 GB/s** |
| ffn_down 17408→5120 | Q4_K | 50 MB | 0.61 ms | 82.7 |
| attn_q 5120→4096 | Q4_K | 11.5 MB | 0.10 ms | 117.7 |
| conv_proj 5120→1280 | Q4_K | 3.6 MB | 0.06 ms | 58.2 |
| ffn_down 17408→5120 | Q6_K | 73 MB | 4.28 ms | **17.1**(待向量化) |

## 3. 修正后的草稿成本与 DFlash2 收益

- 草稿一轮读 1132 MB。按现在的速度:Q4_K 976 MB @ ~80 GB/s = 12.2 ms,Q6_K 155 MB @ 17 = 9.1 ms → **约 21 ms**。把 Q6_K 向量化到 ~60 GB/s 后 → **约 15 ms**。
- 加上 M27a 的验证成本(142 ms):

| 配置 | 一轮 | 代码/推理(K=7,5.20 token) | 散文(2.07) |
|---|---|---|---|
| M26 + 9 ms 草稿(旧估算) | 176 ms | 29.5 tok/s | 11.8 |
| **M27a + 21 ms 草稿(实测)** | **163 ms** | **31.9 tok/s(+28%)** | 12.7 |
| M27a + 15 ms 草稿(Q6_K 向量化后) | 157 ms | 33.1(+32%) | 13.2 |

所以 DFlash2 的预期收益修正在 **+28~32%(仅代码/推理)**;散文仍需 M29 宽度自适应回退。

## 4. 剩下的 M28 工作(按顺序)

1. **Q6_K 向量化**(0.5 天,省 6 ms):照 Q4_K 的 16 字节 job 思路,`ql`/`qh` 各取 16 字节 → 32 个 6-bit 值 + 组内 2 个 int8 scale。
2. **上下文物化**(0.3 天):主干的 5 个残差 tap(第 6/20/34/48/62 层块输出)→ `fc`(25600→5120)→ 每层 `attn_conv_proj`... 不,是每层的 `W_ck/W_cv` 投影 → 环形 KV(2048 窗口 × 5 层 × 8 kv 头 × 128 × 2 × 2 B = 40 MB)。
   注:GGUF 里每层只有 `attn_k/attn_v`,没有单独的 context K/V 投影 —— 按 ninfer 的文档"Source K/V parameters are used for both query-block and context projections"(官方 recipe 共享同一份物理数据),即上下文 K/V 复用 `attn_k/attn_v` 权重,需按 `dflash.md` 的 `k_ctx = rope(plain_head_rmsnorm(W_ck c_t))` 走。
3. **5 层前向**(0.5 天):plain RMSNorm、QKV、逐头 qk norm、标量 RoPE(sections [64,0,0,0] = 只旋转前 64 维、配对 (i,i+32))、**非因果滑窗注意力**(|p_key − p_query| < 2048,包含块内 8 列)、双抽头动态分组卷积、SwiGLU。
4. **选择器**(0.3 天):每个 mask 行的 top-16 来自**目标词表头**(用我们已有的 `k_wide` 一次算 8 行)、从 `selector_predecessor/successor` 各 gather 16 行(每行 256 个 Q4_K 值)、`g_i = W_hidden h_i`(5120→256)、按 `E_i[p,c] = u_i[c] + Σ_r W_pred[pred,r] g_i[r] W_succ[C,c,r]` 做条件路径游走。
5. **spec 循环 + 集成**(0.3 天):草稿 → `verifyWide(T=8)` → 接受前缀(采样时按 `min(1,p/q)` + 残差重采样)→ `commitWide`。
6. **验收**:与 Prism 的 `llama-speculative-simple --spec-type draft-dflash` 在同一台机器上的接受长度对齐(±0.3);端到端代码/推理 ≥ 31 tok/s。

## 5. 复现

```
make build/dq_probe build/bench_draft
build/dq_probe  ../../models/drafters/dflash2-q4km.gguf    # 解码 vs ggml,逐位
build/bench_draft ../../models/drafters/dflash2-q4km.gguf  # 草稿矩阵乘的正确性与带宽
```
