> 本文以中文写成。英文摘要见双语文档索引 [docs/README.md](../README.md)。
> *This document is in Chinese; see the bilingual index at [docs/README.md](../README.md) for an English summary.*

# 里程碑 M5:批量 prefill(精确 fp16 路径 + int8 路径)

日期:2026-10-24

## 设计

- 每个 chunk(最多 1024 个 token)逐层处理。每个三值矩阵:先在 GPU 上把 PTQ1_0 展开成 tile-major 的矩阵布局,再用 MPP `matmul2d` 做 GEMM。展开和 GEMM 在同一个 command buffer 里串行,权重只在 GPU 上以 1.75bpw 常驻。
- **路径 A(精确)**:展开成 fp16(`w = scale * trit`,fp16 乘 ±1/0 是精确的),`half × half → float`,tile 32×64,1 个 simdgroup,N 方向最快。
- **路径 B(int8)**:展开成 int8 trit 加每 (行, 128 组) 的 fp32 scale;激活每 token 一个 scale 量化成 int8;`int8 × int8 → int32`,**每个 K=128 组做一次精确的 scale epilogue**(int32 累加器在 cooperative tensor 里,按元素乘以该组权重 scale 后累加到 fp32 累加器);tile 64×64,4 个 simdgroup。
- 其余算子都做了批量版:embedding、融合残差加法 + RMSNorm + 符号 + FWHT(每 token 一个 threadgroup)、alpha/beta 并成一个 128 列的 GEMM、conv(滑动窗口)、q/k L2 归一化、门控、GDN 递推、gated RMSNorm、attention(批量 RoPE + KV 写入 + 分段 flash-decoding 式 attention)。
- prefill 之后的 GDN 状态、conv 状态、KV 缓存与逐 token decode 完全兼容,可以直接接着 decode。

## cooperative tensor API 的发现(见 `docs/probes/int8_epilogue_api_debug.cpp`)

- `op.get_destination_cooperative_tensor<A, B, int>()` 和 `<..., float>()` 的**布局相同**,可以逐元素混用:`cAcc[i] += float(cInt[i]) * scale`。
- `get_multidimensional_index(i)` 返回 `[列 n, 行 m]`。
- **`cS.load(broadcast tensor)` 的广播加载给出错误结果**(实测),所以改用"按元素索引取 scale",列号和行号在循环外一次算好。
- tile 64×128 和 32×64 因为寄存器溢出只有 4~5 TOPS;**64×64 / 4 simdgroup 达到 26 TOPS**(含精确 epilogue,对比 fp16 持续吞吐 12.6,即 2.1 倍)。

## 质量(72 个位置,以 fp32 逐 token 为金标准)

| 路径 | 平均 KL | 最大 KL | top-1 一致 |
|---|---|---|---|
| llama.cpp 参考实现 | 7.1e-7 | 3.5e-6 | 72/72 |
| A:fp16 精确 | 1.1e-7 | 5.6e-7 | 72/72 |
| B:int8 | 1.9e-4 | 7.7e-4 | 72/72 |

- A 与 fp32 几乎等价(比 llama.cpp 更接近)。
- B 的平均 KL 1.9e-4 低于提议的门槛 1e-3;作为参照,这相当于对权重做 Q8 的量级,而模型本身已经是 1.58 bit。
- **样本只有 72 个位置**,需要在更大、更多样的语料上确认后才能把 B 设为默认(见"待办")。

## 速度(1024 token 的 chunk,热状态,本机)

| 路径 | 耗时 | tok/s | 对比基线(llama.cpp pp512 208) |
|---|---|---|---|
| A:fp16 精确 | 5.60 s | **183** | 0.88× |
| B:int8 | 3.68 s | **278** | **1.34×** |

int8 路径的分解(1024 token,优化前的 4.28 s):GEMM 2.29 s、权重展开 0.60 s、量化 0.04 s、其余(GDN、attention、conv、FWHT)1.48 s。

## 过程中的发现

- **第一次计时包含了 pipeline 编译**(MPP 的 GEMM kernel 编译约 2 秒),必须先预热(`--repeat 2`)。
- GDN 递推原版每线程持有 128 个状态寄存器,**溢出到内存**,耗时 0.83 s;改成"4 个 lane 分担一个状态列,每线程 32 个",用 `simd_shuffle_xor` 归约后降到约 0.14 s。
- 把 alpha/beta 从逐 token 的 GEMV(每 token 重复读 47GB 权重)改成一个 128 列的 GEMM,耗时从 0.34 s 降到接近 0;norm 从每 token 5 个 threadgroup 重复读改成每 token 1 个 threadgroup。

## 待办(prefill 继续优化)

1. int8 权重展开 0.60 s → 理想 0.2 s(目前 threadgroup 转置有 bank 冲突、解码用的是通用慢函数)。
2. GEMM 2.29 s → 微基准是 26 TOPS(1.9 s):按每个 shape 自动调 tile 和网格顺序。
3. attention 目前是逐 token 的分段 decode,O(N²) 且没有 K/V 复用;换成用矩阵单元的 flash attention。
4. 层优先(layer-major)prefill,让权重展开每个 prompt 只做一次,并支持更长的 prompt。
5. 质量:在 ≥10 万 token 的多样语料上验证 int8 路径,过门槛后再作为默认。
6. 逐组激活 scale(而不是逐 token)可以进一步降低 int8 的误差,代价是 epilogue 多一次访存。
