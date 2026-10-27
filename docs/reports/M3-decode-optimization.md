> 本文以中文写成;**英文全译:[M3-decode-optimization.en.md](M3-decode-optimization.en.md)**,下方 `## Summary (EN)` 为摘要,双语文档索引见 [docs/README.md](../README.md)。
> *This document is in Chinese. An English summary follows; the bilingual index is at [docs/README.md](../README.md).*

## Summary (EN)

**Goal.** Make single-token decode fast. The M2 reference kernels decoded the ternaries element by element and cost ~500 ms/token, i.e. ~2 tok/s.

**Result.**

| stage | ms/token | tok/s | note |
|---|---|---|---|
| M2 reference kernels | ~500 | ~2 | per-lane element-wise decode |
| + fast GEMV (`k_gemv_v1`) | 52.4 | 19.1 | operating directly on the GGUF layout |
| + key-major GDN state, fused alpha/beta, fused residual-add + norm, split attention | **49.6** | **20.1** | numerics unchanged: 5/5 top-1, max logit diff 0.0005 |
| baseline (Prism llama.cpp `tg128`) | 63.9 | 15.64 | |

That is **+29% over the baseline** at the time.

**The idea that mattered.** Each 128-weight block is decoded cooperatively by 8 lanes with a fixed byte-to-element mapping (`lane i` owns `qs[2i]`, `qs[2i+1]`, `qs[16+i]` and one trit of `qh[i&1]`), and the base-3 digits are recovered in pure float: with `u = b/256`, the n-th trit is `floor(3^(n+1) u) - 3 floor(3^n u)`. No integer division, no lookup table.

---

# 里程碑 M3:decode 优化

日期:2026-10-24

## 结果

| 阶段 | 每 token | tok/s | 备注 |
|---|---|---|---|
| M2 参考内核 | 约 500 ms | 约 2 | 每个 lane 逐元素解码 |
| + 快速 GEMV(`k_gemv_v1`) | 52.4 ms | 19.1 | 直接作用在 GGUF 原始布局上 |
| + key-major GDN 状态、融合 alpha/beta、融合残差加法与 norm、分段 attention | **49.6 ms** | **20.1** | 数值不变:5 token 提示 top-1 5/5,最大 logit 差 0.0005 |
| 基线(Prism llama.cpp tg128) | 63.9 ms | 15.64 | |

相对基线 **+29%**。

## 快速 GEMV 的关键点

- 每个 128 权重块由 8 个 lane 协作:lane `it` 拥有字节 `qs[2it]`、`qs[2it+1]`(元素 `n*16+2it+{0,1}`)、`qs[16+it]`(元素 `80+n*8+it`)和 `qh[it&1]` 的一个 trit。
- 纯 float 管线解码:字节是 base-3 的 256 分数 `u=b/256`,`t_n = floor(3^(n+1) u) - 3 floor(3^n u)`;因此
  `sum_n t_n y_n = sum_{k=1..4} g_k (y_{k-1} - 3 y_k) + g_5 y_4`,`g_k = floor(3^k u)`。系数只依赖激活,每个字节 5 次 `floor` 加 5 次 `fma`。(思路来自 Prism 内核的注释;实现是我们自己的。)
- 每个 simdgroup 负责 4 个相邻输出行,激活系数在寄存器里被 4 行复用。
- 微基准(`tools/bench_gemv`,真实模型张量,8 个不同层轮换避免缓存):

| 形状 | 参考内核 | `v1_nr4_sg4` | 占持续带宽 142GB/s |
|---|---|---|---|
| ffn gate/up 5120→17408 | 12.6 GB/s | **131.6 GB/s** | 93% |
| ffn down 17408→5120 | 16.3 GB/s | **127.0 GB/s** | 89% |
| gdn qkv 5120→10240 | 12.5 GB/s | **129.1 GB/s** | 91% |
| lm_head 5120→248320 | 12.8 GB/s | **134.1 GB/s** | 94% |

- 结论:**GEMV 不受 ALU 限制,不需要重排布局或 LUT**。

## 逐项消融(每 token,基线 49.6 ms)

| 项 | 代价 |
|---|---|
| 全部 GEMV(含 lm_head) | 45.0 ms(5.657GB ÷ 45ms = 126GB/s) |
| ffn gate / up / down | 9.1 / 9.7 / 10.0 ms |
| gdn qkv / z / ssm_out | 4.3 / 1.9 / 2.6 ms |
| attention q / k+v / o | 1.0 / 0.3 / 0.8 ms |
| lm_head | 1.3 ms |
| GDN 递推步 | 2.6 ms(48 层 × 54µs;状态读写下限 48µs) |
| 融合残差 + norm | 0.5 ms |
| conv、alpha/beta、FWHT ×3、attention(短上下文) | 各 <0.1 ms |
| 所有非 GEMV 的合计 | 5.4 ms |
| 空转(embed + 提交) | 0.4 ms |

## 剩余空间与下一步判断

- 纯 GEMV 在上下文里是 126GB/s(孤立基准 130~134),kernel 之间的尾部/启动损失约 3 ms。
- GDN 状态读写下限约 2.3 ms(fp32);fp16 状态能省约 1.2 ms,但要过质量门槛。
- 这条路线继续抠还能得到约 10%(到约 22 tok/s)。**更大的杠杆是 MTP 投机解码(约 1.6~1.8 倍)和 prefill(目前等于 decode 速度,约 20 tok/s,而基线 208)。**

## 其它变化

- 新增 `--skip name,...`(按 kernel 名称做消融)、`--gen N` 打印 decode 速度。
- 提示 token 只在最后一个位置计算 lm_head。
- attention 改为 flash-decoding:`k_attn_split`(4 个 KV 头 × S 段,GQA 组内 6 个 Q 头共享 K/V 读取)+ `k_attn_combine`。
