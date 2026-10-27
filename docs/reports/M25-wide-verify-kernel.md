> 本文以中文写成;下方 `## Summary (EN)` 是英文摘要,双语文档索引见 [docs/README.md](../README.md)。
> *This document is in Chinese. An English summary follows; the bilingual index is at [docs/README.md](../README.md).*

## Summary (EN)

**Goal.** Build the wide verification kernel that reopens DFlash2 (M14's verdict had been overturned by `R2`). Target: `r = T=16 / T=2 <= 1.6`.

**Result.**

| item | target | achieved |
|---|---|---|
| r on `ffn_gate` | <= 1.6 | **2.47** (420 us vs 170 us) |
| numerics vs fp32 reference | <= 1e-5 | **2.98e-6** (two orders better than the 5.6e-4 of the half2 path) |
| verdict | — | missed 1.6, landed on the lower-bound line of R2's table (r ~ 2) |

**Projection at r = 2.47:** one DFlash2 round (K=7) is `51 x 2.47 + 16 ~ 142 ms` for verification plus 9 ms for the Q4_K_M draft = 151 ms for 5.20 tokens, i.e. **34 tok/s** on code/reasoning (+38% over today's 25), and 2.07 tokens per round on prose.

**What was built.** `k_wide` in `shaders/wide.metal`: one simdgroup owns N output rows with no threadgroup-level barrier; each 128-weight group is decoded through a 16-byte ternary LUT into a **private** fp16 tile (column-major, row stride `KC+8` to break bank conflicts); followed by a shape sweep on the real tensors, comparing variants within a single cooled-down run.

---

# M25:宽验证内核(为 DFlash2 铺路)

日期:2026-10-25 · 对应 `R2-dflash2-revisit.md` 的 M25。前置:R2 推翻了 M14 的"结案"判断,决定重开。

## 1. 目标与结果

| 项目 | 目标 | 结果 |
|---|---|---|
| ffn_gate 上的 r = T=16 / T=2 | ≤ 1.6 | **2.47**(420 µs vs 170 µs) |
| 数值 | 与 fp32 参考 ≤ 1e-5 | **2.98e-6**(比 half2 路径的 5.6e-4 精确两个数量级) |
| 结论 | — | 没达到 1.6,落在 R2 §5 表格的"下界线"(r≈2) |

按 r=2.47 推算一轮 DFlash2(K=7):验证 ≈ 51×2.47 + 16 ≈ 142 ms,加 Q4_K_M 草稿 9 ms → **151 ms / 5.20 token = 34 tok/s**(代码/推理,今天 25,+38%);散文 2.07 token = 14 tok/s(−45%,需要 M29 的宽度自适应回退)。

## 2. 做了什么(以及每一步为什么)

内核 `k_wide`(`shaders/wide.metal`):每个 simdgroup 独占 N 个输出行、无 threadgroup 级屏障;每个 128 权重组经 16 字节三值 LUT 解码进**私有** fp16 tile(列主序、行跨度 KC+8 破坏 bank 冲突、scale 已乘、精确),再由单 simdgroup 的 `matmul2d(M=16, N, K=KC)` 与 MT 个 token 行相乘。

形状扫描(真实张量,冷却后单次运行内相对比较):

| 形状 | 时间 | 说明 |
|---|---|---|
| `k_gemv_lut2`(现用,T=2) | 170 µs | 基线 |
| M16 N16 K128 G4 | **420 µs** | 最好;4 sg/核 = 满占用,17.4 KB |
| M16 N24 K128 G4 | 569 | 26 KB,占用率没变但 N=24 效率差 |
| M16 N32 K128 G2 | 471 | 17.4 KB 但只有 2 sg/核 |
| M16 N32 K128 G1 | 556 | 1 sg/核 |
| M16 N16 K256 G2 | 657 | 16.9 KB,2 sg/核 |
| M16 N8 K256 G4 | 686 | 每 lane 两个 job |
| M16 N64 K128 G1 | 425(仅 MMA) | N 大但只 1 sg/核 |

## 3. 五个试过、无效或有害的方向(附数据,避免重复)

1. **K 加大到 256/512**:微基准里 K 越大越好(每次 `matmul2d` 有约 30 ns 固定开销),但**真实内核里更慢**(657 vs 420):threadgroup 内存翻倍后每核只剩 2 个 simdgroup,占用率从 100% 掉到 50%。占用率比调用开销更重要。
2. **双缓冲(解码下一块与 MMA 当前块并行)**:639 µs,比单缓冲还慢 52%。threadgroup 内存翻倍 + 屏障结构改变,抵消了理论上的重叠。
3. **纯 MMA 消融**:306 µs(291 G 权重/s、9.4 TFLOPS),与形状无关(N=8/16/32/64、K=128/256 都在 306–430)。**M=16 时矩阵单元的吃入率就只有峰值的 53%**;而生产预填充 GEMM(M=32/64、一次调用跑完整个 K=5120)能到 14.7–16.6 TFLOPS —— 差别在于每次调用覆盖的 K 和 M 的大小,token 维度太小时无法摊薄。
4. **ALU + 矩阵单元混合**(按吞吐 2:1 把输出行分给两条路,各自读自己的行):473 µs,介于 MMA-only(420)和 ALU-only(590)之间,**没有重叠收益**。这台机器上 ALU 与矩阵单元跨 warp 时互相抢发射槽,不是互补。
5. **用设备内存喂 B**(避免 threadgroup 中转):必须先展开权重,每轮 12.8 GB 写入 = 90 ms 下限,是带宽下限(40 ms)的两倍多,不可行。

## 4. 与外部实现的对照

ninfer 在 RTX 5090 上,DFlash2 一轮(T=8)是单步解码的 **1.55×**。换算到我们的 T=2 基准大约是 r≈1.35。我们做到 2.47,即**比一个已知实现慢 1.8 倍**。差距集中在"矩阵单元在 M=16 时的吃入率",而这在我们能控制的层面(tile 形状、占用率、屏障)已经扫完了;再往上需要 MPP 内部的信息(比如 B 操作数在寄存器里的那条路),风险高、不确定。

## 5. 决定

- 采用 `k_wide_m16_n16_k128_g4`(420 µs)作为验证内核,进入 M26。
- 收益预期按 r=2.47 计算:**代码/推理 +38%,散文靠 M29 的宽度自适应回退到今天的水平**。
- 如果以后想再压:唯一没试的是"B 操作数从寄存器喂给 matmul2d"(`get_right_input_cooperative_tensor`),它要求权重按 lane 的元素映射重排,与 M12 的结论(重排对 T=1 无用)不冲突但成本高;记为可选实验。

## 6. 产物

`shaders/wide.metal`(`k_wide` 参数化 M/N/KC/NSG,双缓冲与消融开关,`k_wide_hyb` 混合路径)、`tools/bench_wide.cpp`(真实张量上的形状扫描 + 逐 token 正确性对照)。两者都是生产路径的候选/证据,不是一次性脚本。
