> 本文以中文写成。英文摘要见双语文档索引 [docs/README.md](../README.md)。
> *This document is in Chinese; see the bilingual index at [docs/README.md](../README.md) for an English summary.*

# M27a:验证窗口的小网格修复(每轮 −20%,上限抬到 +38%)

日期:2026-10-25 · 对应 `CHECKPOINT-004` §5 的 M27a。目标是把"每轮固定开销"从 41 ms 压下来,让 DFlash2 的收益上限提高。

## 1. 结果

| 配置 | T=2 | T=4 | T=8 |
|---|---|---|---|
| M26(修复前) | 156 ms | 159 | 167 |
| **M27a(现在)** | **141 ms** | **140** | **142 ms** |

同一时段内的普通解码基线 50.7 ms/token(降频状态),修复前的同一状态测得 177 ms/轮 → **−20%**(比 CHECKPOINT-004 定的 147 ms 目标更好)。7.2K 上下文下 T=8 为 176 ms(分段注意力)对 173 ms(flash),两者等价。

对 DFlash2(K=7,实测 5.20 token/轮,草稿 Q4_K_M 约 9 ms)的影响:

| 配置 | 一轮 | 代码/推理 | 散文 |
|---|---|---|---|
| M26 | 176 ms | 29.5 tok/s(+18%) | 11.8 |
| **M27a** | **151 ms** | **34.4 tok/s(+38%)** | 13.7 |
| 修完 + 草稿用 Q8_0(16 ms) | 158 ms | 32.9 | 13.1 |

## 2. 三处修复

**(1) alpha/beta:2 个 threadgroup → 96×T 个。**
原来复用 prefill 的 `dGemm(wabAll → pAB)`,Mpad=32 时只有 2 个 threadgroup,每层都是纯延迟(实测 6–12 ms/轮)。新内核 `k_gdn_ab_t`:一个 simdgroup 负责一个(输出行, token),grid (96, T) = 768 个 threadgroup,直接从 bf16 权重与 fp16 的归一化隐状态算点积,写进 `ab[t][0..47]=alpha` / `[48..95]=beta`。

**(2) 注意力:flash → key 分段(分段按上下文长度取 1–32)。**
`k_flash_prefill` 每个 KV 头一个 threadgroup → T=8 短上下文时整个 GPU 只有 4 个 threadgroup(实测 17 ms/轮,纯延迟)。改用 `k_attn_split_b` + `k_attn_combine_b`:grid (S, 4, T) = 短上下文 32 个、7K 上下文 1024 个 threadgroup。7.2K 上下文两者等价(176 vs 173 ms),所以短上下文用分段、**int8 KV 仍走 flash**(分段内核只读 fp16 缓存)。两条路径都验证过:T=8 只接受第 0 行时逐 token 复现贪心(550, 220, 16, 13)。

**(3) GDN 状态:9 份快照 → 记录 + 重放(省 1.2 GB 内存 + 5 ms)。**
M26 的 `k_gdn_rec_snap` 每轮写 8 份 147 MB 状态 = 1.2 GB(8.4 ms)+ 1.36 GB 显存。现在验证过程**不动状态**(`k_gdn_rec_snap` 的 `snapStride = 0` 模式),把每层的 conv/qk/gate 记录存到 `vRec`(48 层 × 8 行 × 14332 float = 22 MB),等接受长度确定后由 `k_gdn_rec_b2` 从已提交状态**一次读完、一次写完**地重放接受的前缀(状态始终在寄存器里跨 F 个 token)。重放代价 = 2 × 151 MB + 记录读取 ≈ 3 ms,比写 8 份快照便宜,还少了 1.36 GB 显存。

## 3. 现在的每轮分解(T=8,约 142 ms)

| 项 | 时间 | 备注 |
|---|---|---|
| 宽 GEMV(FFN 92.5 / GDN 26 / 注意力 13 / lm_head 6) | ~126 | M25 的硬限制(46 GB/s) |
| GDN 递推 + 重放 | ~4 | 记录+重放 |
| 注意力(分段) | ~3 | 原 17 |
| alpha/beta | ~1 | 原 6–12 |
| 其余(norm/FWHT/conv/qk/gates/gnorm/argmax) | ~8 | |

**剩下 89% 是宽 GEMV**,也就是 M25 的矩阵单元吃入率问题。要再快只能动那个(r 从 2.47 → 1.6 会让每轮降到 ~107 ms、代码/推理到 48 tok/s)。

## 4. 复现

```
build/kata --model $M --ids "$(cat /tmp/ids72.txt)" --vwidth 8 --gen 16   # 每轮成本 + 逐 token 复现贪心
KATA_KV=int8 build/kata ... --vwidth 8                                   # int8 KV 走 flash
KATA_SKIP=v_ffn|v_gdn|v_ab|v_attn_only                                     # 分项消融
```
