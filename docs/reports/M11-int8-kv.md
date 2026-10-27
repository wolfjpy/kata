> 本文以中文写成;**英文全译:[M11-int8-kv.en.md](M11-int8-kv.en.md)**,下方 `## Summary (EN)` 为摘要,双语文档索引见 [docs/README.md](../README.md)。
> *This document is in Chinese. An English summary follows; the bilingual index is at [docs/README.md](../README.md).*

## Summary (EN)

**Goal.** An int8 KV cache (`--kv int8`), after ninfer's default KV formats (FP8 row-256, INT8 group-64).

**Result.**

| item | fp16 KV (default) | int8 KV (`--kv int8`) |
|---|---|---|
| KV memory | 64 KB/token | **33 KB/token** (K and V, 16 layers x 1024 bytes, plus one fp16 scale per head-row) |
| KV at 32K / 64K context | 2.1 / 4.2 GB | 1.1 / 2.2 GB |
| decode, 7.3K context (plain / MTP) | 17.5-18.0 / 21.4-21.5 tok/s | 18.1 / 21.3-21.5 tok/s (no real gain) |
| **decode, 29K context (plain)** | 13.7 tok/s | **15.6 tok/s (+14%)** |
| **prefill, 29K prompt** | 162 tok/s | **178 tok/s (+10%)** |
| quality: 3072 positions of a 24.5K corpus | — | **mean KL 1.28e-4, top-1 99.71%** (3063/3072), max KL 0.18 |
| needle in a haystack (11.4K tokens, depth 10/50/90%) | 3/3 | 3/3 |
| `verify2` against exact decode | KL 2.5e-7 | KL 9.2e-7, top-1 66/66 |

**Correction (added later, see the end of this report).** The 99.71% figure does not reproduce: the in-process comparison is invalid because the KV mode decides the buffer allocation at load time. Measured cross-process it is **99.51% top-1**, and the mean KL is dominated by a handful of degenerate positions whose reference logits are around 1e6. Asymmetric (mean + scale) quantisation was implemented to fix it and measured *worse* (99.37%) because a head's values are already near-symmetric. Hence: opt-in, never a default.

---

# M11:int8 KV 缓存(`--kv int8`,有损,默认关闭)

日期:2026-10-25 · 对应 CHECKPOINT-003 路线图 M11。依据:ninfer 的 KV 默认格式("FP8 row-256 / INT8 group-64")。

## 1. 结果

| 项目 | fp16 KV(默认) | int8 KV(`--kv int8`) |
|---|---|---|
| KV 内存 | 64 KB/token | **33 KB/token**(K、V 各 16 层 × 1024 字节 + 每 head-row 一个 fp16 scale) |
| 32K / 64K 上下文的 KV | 2.1 / 4.2 GB | 1.1 / 2.2 GB |
| 解码,7.3K 上下文(普通 / MTP) | 17.5–18.0 / 21.4–21.5 tok/s | 18.1 / 21.3–21.5 tok/s(**无明显收益**) |
| **解码,29K 上下文(普通)** | 13.7 tok/s | **15.6 tok/s(+14%)** |
| **预填充,29K 提示** | 162 tok/s | **178 tok/s(+10%)** |
| 质量:24.5K token 语料,3072 个位置,对比 fp16 KV | — | **平均 KL 1.28e-4,top-1 99.71%**(3063/3072),最大 KL 0.18 |
| 大海捞针(11.4K token,深度 10/50/90%) | 3/3 | 3/3 |
| `verify2` 对精确解码(72 token) | KL 2.5e-7 | KL 9.2e-7,top-1 66/66 |

结论:质量刚好过门限(KL ≤ 1e-3、top-1 ≥ 99.7%),**内存减半,长上下文(≳ 16K)才有速度收益**;短上下文没有速度收益。所以保持**默认关闭**,需要长上下文或省内存时用 `--kv int8`。1536 个位置时 top-1 是 99.67%、3072 个位置时是 99.71%,说明真实值就在 99.7% 附近,边缘通过。

## 2. 设计

- **格式**:K、V 各按"位置 × KV head"一行 256 个值,`scale = fp16(amax / 127)`,值 `round(x / scale)` 限制在 ±127;K 存的是 RoPE 之后的值。scale 存在 `[层][位置][4 个 head]` 的 fp16 平面里。
- **写入**:现有的 `k_attn_prep`/`k_attn_prep_b` 加了一个 `cacheOff` 参数,int8 模式下把新行写到一个很小的 fp16 暂存区(`[kMaxChunk][1024]`),再由 `k_kv_quant` 量化进 int8 缓存和 scale 平面。解码每层多一次小 dispatch,预填充每层一次。
- **读取**:`k_flash_prefill` 和 `k_attn_dec_split` 改成对 K/V 元素类型做模板(`half` / `int8_t`);`matmul2d` 直接吃 `half × int8_t`(规范里的原生类型,吞吐约等于 fp16)。`S = Q Kᵀ` 的每一列(= 每个 key)乘它的 K scale;`P` 的每一列乘 V scale 之后再去乘 V(行和 `l` 用未缩放的 `p`)。
- MTP 头的那 1 层 KV 在所有模式下都保持 fp16(占主 KV 的 ~6%)。
- 前缀缓存:检查点按"平面"读写(fp16 是 K、V 两个平面,int8 是四个),数值标签带 `+kv8`,格式版本升到 2(旧检查点自动失效)。

## 3. 敏感度:是 K 还是 V?(fp16 缓存里做量化-反量化的模拟,同一份 3072/1536 个位置的评测)

模拟 `kv256`(K+V,每行一个 scale)的结果与真实内核吻合(1.53e-4 / 99.67% vs 1.77e-4 / 99.67%,同样的 1536 个位置),所以模拟可信:

| 量化范围 | 平均 KL | top-1(1536 个位置) |
|---|---|---|
| K+V,每行 | 1.5e-4 | 99.67% |
| 只量化 K,每行 | 8.4e-4(一个 KL = 1.2 的离群位置贡献了 7.9e-4) | 99.61% |
| 只量化 V,每行 | **3.3e-5** | 99.80% |
| K+V,每 64 维一组 | 1.2e-4 | 99.80% |

要点:**误差主要来自 K**(softmax 之前的分数被指数放大);V 几乎无损。每 64 维一组的 scale 能把 top-1 拉到 99.80%,但差别在测量噪声内(±2 个位置),且 K 的分组 scale 要把 `Q Kᵀ` 拆成 4 个 K=64 的 matmul 再分别缩放,代价不小,所以没做。若以后需要更高的质量余量,更便宜的选项是**只把 V 做成 int8(K 保持 fp16)**:KL 3.3e-5,但只省 25% 的 KV 字节。

## 4. 已知限制

- 7K 上下文以内没有速度收益(注意力那点时间被延迟而不是字节主导);只有内存收益。
- 旧的标量注意力内核(`KATA_SCALAR_*`)只支持 fp16 KV,int8 模式下被忽略。
- 评测语料是论文/文档/规范的混合文本;需要更长(32K+)的真实对话/代码语料才能得出更窄的置信区间。

## Correction (M29e, re-measured properly)

The "KL 1.3e-4 / top-1 99.71%" figure above does not reproduce. The paired `kl-eval` mode is invalid for int8 KV
because `kvInt8` decides the buffer allocation at load time, so running both paths in one process only measures the
layout change. Cross-process measurement instead:

```
KATA_KV=fp16  kata kl-eval ... --save-ref ref.bin
KATA_KV=int8  kata kl-eval ... --ref ref.bin
-> 2048 positions | mean KL 3.1e+01 | max KL 4.7e+04 | max |logit diff| 7.9e+06 | top-1 2038/2048 (99.51%)
```

**The huge KL numbers are a metric artifact, not a broken kernel.** Running the per-position diagnostic
(`KATA_KL_DBG=1`) shows every position with KL > 1 also has enormous *reference* logits:

| position | KL | reference top logit | int8 top logit |
|---|---|---|---|
| 172 | 8.2e+03 | 1.342e+06 | 1.334e+06 |
| 939 | 8.3e+03 | 4.295e+05 | 4.193e+05 |
| 1711 | 4.7e+04 | 7.261e+05 | 7.117e+05 |
| 387 | 2.1e+00 | 226.2 | 225.6 |

A 0.5% relative error at a position whose logits are 1e6 is a ~5000 nat shift, so the two softmax distributions become
effectively disjoint and the KL explodes; the *mean* KL over the corpus is then dominated by a handful of such
positions (`(8243 + 8269 + 47090 + ...) / 2048 = 31`). At ordinary positions the error is small. The honest metric is
top-1, and it is **99.51%**, below the 99.7% gate.

**Root cause: quantisation granularity, not a kernel bug.** One scale covers all 256 elements of a (position, KV head)
vector, so a single outlier costs 1-2 effective bits and the whole head pays. That is the *design* constraint of this
scheme: the MMA path can only apply a scale as a per-key scalar (`cS *= kS[key]`, `P *= vS[key]`), so finer groups
cannot simply be added — a group scale would have to be folded into the dot product itself.

Two fixes are viable, in increasing effort:

1. **Asymmetric (scale + zero-point) quantisation**, which is exact to fold into the MMA path:
   `k = sc * q + mean`, so `S = sc * (Q K_q8) + mean * sum_d(q_d)` — the correction is one scalar per (query row, KV
   head) because `mean` is constant across the head's 256 dims, and `sum_d q_d` is computed once per row. For V the same
   decomposition gives `O += sum_k(p_k * mean_k)`, one scalar per (query row, KV head). This removes the outlier bias
   and typically recovers ~1 bit.
2. **Two scales per head** by splitting the 256 dims into two 128-wide MMAs (`S = sc_lo (Q_lo K_lo) + sc_hi (Q_hi K_hi)`),
   which is also exact and cheap (same FLOPs, two calls).

Until one of those lands, int8 KV stays opt-in and is not a default.

## Asymmetric quantisation: implemented, measured, rejected (M29g)

Fix 1 above (asymmetric `k = sc*q + mean`, folded back into the attention as `+ mean * sum_d(q_d)`) was implemented for
the key side in all three places (quantiser, prefill kernel, decode kernel; `kmean` buffer, per-row `qSum`, A/B switch
`KATA_KV_ASYM`). It measures **worse**, reproducibly:

| quantiser for K | top-1 (2048 positions) |
|---|---|
| symmetric `amax/127` (default) | **99.51%** |
| asymmetric `(min,max)` + mean correction | 99.37% |

The reason is arithmetic rather than a coding error: when the values of one head are roughly symmetric about zero,
`(MX - MN)/254 = 2*amax/254 = amax/127`, i.e. the asymmetric scale is the *same* as the symmetric one, so there is no
extra resolution to gain; the mean term then only adds one more fp16 rounding (`mean` is stored as fp16 and multiplied
by `sum_d(q_d)`, which is O(1) after the 1/sqrt(256) prescale). Fix 2 (two scales per head, 128-dim split MMAs) is
exact but needs a second 16x256 accumulator for V (128 registers/lane), which does not fit.

So the honest status of the int8 KV cache: **99.51% top-1, opt-in, not a default**; the coarse per-(position, head)
scale is a property of what the `matmul2d` path can fold, and passing the 99.7% gate needs either fp8 storage or
per-group scales with one MMA per group.

## M31:长上下文逐位置 KL、外部对照,以及 K/V 拆分的实现

由一位读者提出的两点质疑驱动:(1) top-1 会掩盖只在长上下文深处才出现的漂移,应该记录**逐位置 KL**;(2) 如果 K 和 V 都做 int8
有害,K 通常比 V 敏感,应该做 **K fp16 / V int8** 的廉价 A/B。两条都成立,下面是数据。

### 1. 工具:逐位置 KL 与深度分桶

`kl-eval` 增加 `KATA_KL_TRACE=1`:逐位置打印 `pos KL 参考logitmax 候选logitmax top1`,并给出**按深度分桶**的汇总
(0-1K / 1-4K / 4K-8K / 8K-16K / 16K-32K / ≥32K)。另加 `KATA_KV_DUMP=文件`:导出第 0 层 K/V 平面的前 64 KB,用于两种模式逐字节对比。

**控制实验**(同一份参考对自身,fp16 KV,16384 token):`mean KL 0.000e+00 | max 0.000e+00 | 4096/4096` —— harness 确定性,
所以下面的深度趋势是信号而不是噪声。

### 2. 深度曲线:int8 KV 对 fp16 KV,16384 token,4096 个位置

排除退化位置(参考 top logit ≥ 1e3,约占 20%)后的分位数:

| 深度 | n | p50 | p90 | p99 | max | KL>1e-3 | top-1 错 |
|---|---|---|---|---|---|---|---|
| 0-4K | 1021 | 2.45e-05 | 2.32e-04 | 5.88e-03 | 1.01e-01 | 2.6% | 0.8% |
| 4-8K | 1020 | 2.16e-05 | 2.09e-04 | 9.44e-03 | 4.36e-02 | 3.3% | 0.6% |
| 8-12K | 1020 | 1.94e-05 | 2.11e-04 | 1.07e-02 | 4.16e+00 | 3.1% | 1.2% |
| 12-16K | 1020 | 7.36e-06 | 3.13e-04 | 2.21e-02 | 1.01e+00 | 3.9% | 0.8% |

**中位数随深度不涨**(对 log10(KL) 回归位置:斜率 **−0.022 dex / 1K token**,基本平坦),但**尾部在变重**:
p99 从 5.9e-3 涨到 2.2e-2(3.7 倍),过门限(KL>1e-3)的比例 2.6% → 3.9%。所以读者的判断对:**top-1(0.8%~1.2%)掩盖了尾部的恶化**;
但"慢性漂移"不是发生在每一处,而是**少数位置在深处变得更差**。

被排除的 ~20% 退化位置(参考自身 logit 在 1e3~1e10)与之前一致:那里 KL 无量纲意义(1e6 尺度上 0.5% 的相对误差就是数千 nat)。

### 3. 外部对照:llama.cpp 的 `--kl-divergence`

同一模型、同一段文本(512 token),llama.cpp 自己的 fp16 KV 作为基线,再把 KV 换成 `-ctk q8_0 -ctv q8_0`:

```
llama-perplexity ... --save-all-logits /tmp/ll_base.bin          # fp16 KV 基线
llama-perplexity ... -ctk q8_0 -ctv q8_0 --kl-divergence --kl-divergence-base /tmp/ll_base.bin
  -> KLD: -0.000017 / -0.000031 / -0.000041 (逐千分位), Mean Δp: -0.000 ± 0.000 %
```

即 **|KL| ≈ 4e-5**,而我们的 int8 KV 在同一量级上是 **≈1e-4**(中位数 2e-5、p90 2e-4)。两者同数量级 → **本仓库的 KL harness 得到
独立实现印证**;而 2-3 倍的差距正好来自粒度:llama.cpp 的 q8_0 是 **32 元素一块**(每个 256 维头 8 个 scale),我们是
**一个 256 维头一个 scale**。这与 §3 的模拟结论一致(把 scale 细分到 64 维能把 KL 拉低一个档),也解释了为什么 M29g 的非对称
量化救不了它:问题在**分组**,不在零点。

### 4. K/V 拆分(K fp16 / V int8):已经实现,但**当前不可用**

- 新增两个内核 `k_flash_prefill_k16v8` 与 `k_attn_dec_split_k16v8`(K 走 fp16 精确路径、V 走 int8 + 每 (位置, KV head) 一个 scale),
  引擎侧把 KV 模式的 K/V 两个方向彻底拆开(`kvInt8K` / `kvInt8V`、各自的层偏移、各自的 scale 平面、前缀缓存 3 平面),
  并新增模式 `KATA_KV=int8v`。
- **验证到的事实**:V 的 int8 数据平面在 `int8` 与 `int8v` 两种模式下**逐字节相同**(100%),说明 V 侧的量化与寻址正确;
  K 侧在混合模式下**完全没有被写入**(K 平面前 64 KB 全零,而 fp16 模式 65421/65536 字节非零)。
- 定位到的两个真 bug(已修):`prep` 内核把"写进缓存还是写进暂存"的偏移与"哪一侧量化"的标志混用(`cOff` 应为 `kvInt8K`),
  以及混合模式下 K/V 需要**两个不同的行偏移**(K 按层偏移绑定 → 0;V 写暂存 → pos0),为此给 prep 内核加了 `cacheOffV`。
  但修完之后 K 仍然未被写入,原因未定位。
- 因此 **`KATA_KV=int8v` 现在会直接拒绝运行并打印原因**,而不是给出错误结果。fp16 与 int8 两条路径的数值与改动前**完全一致**
  (2048 位置:fp16 KL 0/2048-2048,`int8` KL 3.1e+01 / 99.51%,与 M29f 相同),生成正常。

**下一步(接手的入口)**:在混合模式下 dump `kvScrK` 与 `kcache` 的第 0 层并对比 prep 的实际写地址;或者临时把 prep 的
buffer 7 在混合模式下改绑到暂存区 + 把暂存内容拷进缓存,二分出是"写错位置"还是"没写"。
