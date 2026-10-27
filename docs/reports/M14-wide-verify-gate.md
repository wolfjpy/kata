> 本文以中文写成。英文摘要见双语文档索引 [docs/README.md](../README.md)。
> *This document is in Chinese; see the bilingual index at [docs/README.md](../README.md) for an English summary.*

# M14:宽验证微基准闸门——未通过,DFlash2(M15/M16)结案

> **结论已被推翻(2026-10-25)。** 本文的测量数据仍然有效,但"结案"的判断错了:它拿 T=2 当基准而不是带宽下限,并且原型内核对 K=128/M=8/N=16/串行这几项都是有缺陷的。重新研究见 **`R2-dflash2-revisit.md`**:DFlash2 已重开(M25–M29)。

日期:2026-10-25 · 对应 CHECKPOINT-003 路线图 M14。用户的决定:DFlash2 作为多天高风险项目,先用宽验证微基准做闸门。**闸门标准:T=8 每矩阵 ≤ 1.5 × T=2(ffn_gate 约 255 µs)。** 结果:**未通过**,DFlash2 不做。

## 1. 设计(吸取 Strata 的结构,用 M6 的矩阵硬件)

`shaders/wide.metal`、`tools/bench_wide.cpp`。每个 simdgroup 独立负责 R 个输出行(没有 TG 级屏障,只有 `simdgroup_barrier`):每个 128 宽的 K 块,把自己的 R×128 权重经 16 字节 LUT 解码进**私有**的 TG 内存 tile(行跨度填充 8 个元素避免 bank 冲突),然后单 simdgroup 的 `matmul2d`(M = token 行,N = R,K = 128)把 fp16 激活(直接从设备内存取)和这个 tile 相乘,累加在协作张量里。权重只解码一次、所有 token 共用。两个版本:
- `k_wide_sg`:tile 是 fp16(已乘 scale,精确);
- `k_wide_q8`:tile 是 int8 三值(精确,不乘 scale),`half × int8 → float`,每个 128 块的行 scale 作用在 MT×R 的部分结果上再累加(每 lane 4 个元素)。

昨天原型(`k_wide_gemm`,831 µs/16 token)的两个问题都被解决了:4 个 simdgroup 共用 20 KB 块 → 每个 simdgroup 私有 4–8 KB;TG 级屏障 → simdgroup 级。**数值正确**(与 fp32 参考的最大误差 1e-6 到 1e-5,比 half 累加的 LUT 内核精确 2 个数量级);**M=8 的 `matmul2d` 可用**,且 M=8 与 M=16 耗时相同(token 维度基本免费)。

## 2. 结果(每矩阵,µs;同一次运行内比较)

| 形状 | T=2 LUT(现用) | T=4 FMA-LUT | T=8 FMA-LUT | **T=8 wide fp16** | **T=8 wide int8** | wide int8 / T=2 |
|---|---|---|---|---|---|---|
| ffn_gate/up 5120→17408 | 170 | 297 | 589 | 416 | **367** | **2.16×** |
| ffn_down 17408→5120 | 174 | 307 | 615 | 441 | **388** | **2.23×** |
| gdn_qkv 5120→10240 | 101.5 | 180 | 351 | 256 | **233** | **2.30×** |

闸门 1.5× 在三个形状上都没过(2.2–2.3×)。宽内核比 T=8 的 FMA 版快 1.4–1.6×,但在 T=4 以下比 FMA 版慢,只在 T ≥ 6 才有意义。

## 3. 为什么卡在这里

消融(M=8,R=16,fp16 版):整体 416 µs;跳过解码 281 µs → **`matmul2d` 本身要 ~280 µs**,解码贡献 ~135 µs。把 K 块从 128 加大到 256/512,耗时按比例增长(620/908 µs),说明不是每次调用的固定开销,而是**矩阵单元从 TG 内存里的 B 操作数每秒能吃进的权重量**:约 318 G 权重/s(ffn_gate 的 89M 个权重 / 280 µs),只有带宽下限(640 G 权重/s)的一半。无论解码怎么优化,T=8 每矩阵也不可能低于 ~280 µs。int8 版把 TG tile 字节减半,只买到 12%。

## 4. 经济账(为什么结案是对的)

按 367 µs 估算:T=8 验证的 GEMV 部分 ≈ 51 ms × 367/170 ≈ 110 ms,加上注意力/GDN 在 T=8 下的额外开销约 15 ms,一轮验证 ≈ 125 ms;DFlash2 的草稿约 9 ms。

| 文本 | DFlash2 每轮 token | 轮耗时 | tok/s | 现在的 MTP 深度 1 |
|---|---|---|---|---|
| 代码 / 推理 | 5.1 | ~134 ms | **~38** | ~29(1.8 token / 64 ms) |
| 散文 | 2.2 | ~134 ms | ~16 | ~25 |

即使在最有利的文本上也只有 ~+30%,散文反而慢 35%(需要自适应回退),而且代价是多天、高风险的移植(块扩散 Transformer、选择器、动态卷积、上下文物化、ReplaySSM 重放、Q4_K_M 解量化)。**不值得。**

## 5. 什么会改变这个结论

- 把 B 操作数从寄存器而不是 TG 内存喂给 `matmul2d`(规范里有 `get_right_input_cooperative_tensor`,可以让每个 lane 直接用算术解码自己拥有的那些元素,省掉 TG tile 和屏障)。**这是唯一没有试过的变体**;若矩阵单元的 B 流入速率是限制,它有可能翻倍,但每个 lane 拥有的元素位置分散,解码要逐元素做算术(每元素 ~3–4 条指令),收益不确定。
- 更新的 OS/硬件(M6 Pro/Max 有更多矩阵单元)改变 `matmul2d` 的小 tile 吞吐。
- 如果以后有 ≥ 5 的稳定接受长度并且验证 T=8 ≤ 255 µs/矩阵,重新评估 DFlash2。

## 6. 随带产物

`shaders/wide.metal`(两种宽内核,实验性,引擎不使用)、`tools/bench_wide.cpp`(闸门基准,`make build/bench_wide`;命令:`build/bench_wide $M build/kata.metallib k_gemv_lut2_nr4_sg4 k_wide_q8_m8_r16_g4 ...`)。R=32/64 的 int8 变体速度相近但结果不对(没有追查原因),报告中不使用。
