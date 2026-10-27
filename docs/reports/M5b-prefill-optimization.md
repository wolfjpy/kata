> 本文以中文写成。英文摘要见双语文档索引 [docs/README.md](../README.md)。
> *This document is in Chinese; see the bilingual index at [docs/README.md](../README.md) for an English summary.*

# M5b:预填充优化(flash attention、GEMM 顺序、展开、int8 质量评估)

日期:2026-10-24 · 提交 `00264f7`(注意力)、`96e048d`(展开)、`a3581ee`(int8)、`53b04a8`(GEMM 顺序)

## 1. 结果

| 场景 | M5 之前 | 现在 | 基线(Prism llama.cpp) |
|---|---|---|---|
| 精确 fp16,512 token | — | **245 tok/s** | 208 tok/s(pp512) |
| 精确 fp16,1024 token chunk | 177 tok/s | **257 tok/s**(+45%) | |
| 精确 fp16,7.3K token 提示 | 164 tok/s(已有 flash attention,旧 GEMM 顺序) | **213 tok/s**(+30%) | |
| int8(`--fast`,opt-in),1024 chunk | 278 tok/s | 270–283 tok/s(已无明显优势) | |
| int8,7.3K 提示 | 176 tok/s(标量注意力) | 232 tok/s | |

(GPU 持续负载下会降频,同一配置重复测量可差 10–15%。上表取降频后的稳定区间;冷机时 int8 1024 chunk 可到 330 tok/s。)

精确路径的 GEMM 现在是 3.0 s / 1024 token = **16.6 TFLOPS,已贴近 fp16 矩阵单元的峰值(17.3)**。

## 2. 改动与测量

### 2.1 矩阵单元 flash attention(`shaders/flash.metal`)

旧的注意力是标量内核:每个 query token 单独扫描整段 K/V,复杂度 O(n²) 且无法利用矩阵单元。7.3K token 提示的 41 s 里注意力占 ~17 s,32K 上下文会是灾难。

新内核 `k_flash_prefill`:
- 一个 simdgroup = 16 个 query 行 × 1 个头;同一 GQA 组的 6 个头组成一个 threadgroup(192 线程),读同一份 K/V。
- `S = Q Kᵀ`:`matmul2d_descriptor(16, 64, 256, transpose_right=true)`,K 直接以转置视图从 fp16 缓存读取(张量 stride 1024)。
- 掩码 + 在线 softmax:`reduce_rows(max / sum)` 把每行的结果存到 TG 内存。
- `O += P V`:P 经 TG 内存(half)作为左操作数,`multiply_accumulate` 累加进协作张量(16×256 fp32,每线程 128 个寄存器)。
- 尾部 key 块使用真实的张量 extent(`dextents(256, nk)`),MPP 对界外自动补零:既不读缓存之外,也避免 `NaN × 0 = NaN`(毒药测试:缓存有效区外填 NaN,结果有限且正确)。
- 整个 chunk 一次 dispatch(旧实现按 256 token 切分,再 split + combine)。

微基准(`tools/probe_flash.cpp`):与 CPU fp64 参考的最大误差 6.6e-5(参考值量级 0.28);3.1–3.3 有效 TFLOPS。每层、每 1024 token chunk:位置 0 时 4.0 ms,位置 7168 时 62 ms。
整个 8K 预填充的注意力总计约 4 s(旧版 ~17 s);32K 约 67 s(仍是线性部分 118 s 之外的可观开销,见 §4)。
真实模型上精确预填充 KL 不变(均值 1.39e-7,旧内核 1.13e-7,top-1 72/72)。

### 2.2 GEMM 网格顺序(对精确路径影响最大)

意外发现:把 `dispatchThreadgroups` 从二维 `(N/64, M/32)` 改成**一维线性 id + 自己的映射**后,N 优先(GM=0)慢到 12.8 s——原先二维 dispatch 被硬件按局部性更好的顺序调度,所以之前的"N 优先最好"结论其实是在这个隐含顺序下得到的。

在一维线性顺序下扫描"分组顺序"(每组 GM 个相邻的 M 块一起遍历全部 N 块,同一条 B 条带被 GM 个块复用,而这 GM 个块的 A 行留在缓存里):

| GM | 精确路径,1024 token |
|---|---|
| 0(N 优先) | 12.8 s |
| 4 | 8.2 s |
| 8 | 5.6 s |
| 16 | 4.4 s |
| 24 | 4.9 s |
| **32(= 1024 行全在一组,即 M 优先)** | **3.98 s** |

chunk 取 2048 时 GM=32 只有 190 tok/s(8K),GM=64 只有 157:A 矩阵(M×K)太大装不进缓存。结论:**chunk 1024、GM 32**。int8 路径对顺序不敏感(受 ALU/epilogue 而不是 B 带宽限制)。
实现:`tile_order(id, Mt, Nt, gm)`,环境变量 `KATA_GM` 可覆盖(默认 32)。

### 2.3 权重展开内核

旧内核每权重约 30 条指令(逐元素 `ptq1_elem` 解码、经 TG 内存转置、逐字节写出)。新内核 `k_expand_*_v2`:线程 = (4 行 × 8 个连续 k);一个 8 元素切片总是共享同一个 trit 位置 `n`(s<10:`n=s/2`、字节 `(s&1)*8..`;s=10..14:`n=s-10`、字节 16..23;s=15:`qh`),所以解码是统一的 `floor(3^(n+1)u) - 3 floor(3^n u)`,四行的 trit 打包成一个 32 位字直接写出(无 TG 内存、无 barrier)。与旧内核**逐位一致**(fp16 与 int8 路径的 KL 完全相同)。int8 展开 0.64 → 0.41 s。

### 2.4 int8 路径:per-group 激活 scale + 向量化 epilogue

- 激活量化粒度从"每 token 一个 scale"改为"每 (token, 128 元素组) 一个 scale",epilogue 在逐组缩放时多乘一个行因子。72 个位置:平均 KL 2.3e-4 → 8.2e-5,最大 KL 3.2e-3 → 3.1e-4。
- 协作张量元素布局(`tools/layout_probe`):每个 lane 的 32 个元素 = 4 行(r, r+8, r+32, r+40)× 2 个列块(c0..c0+3, c0+32..c0+35)。epilogue 因此只需要每组 2 次 `float4` 读 weight scale + 4 次读 activation scale(运行时校验布局,不符则回退通用路径)。同条件下比通用路径快 10–13%,结果逐位一致。

## 3. int8 质量评估(新工具 `kata kl-eval`)

在 24.5K token 的混合语料(论文、中英文文档、MSL 规范片段)上,精确路径与 int8 路径各自完整预填充(各自维护状态),比较每个 chunk 末尾 64 个位置(共 1536 个)的全词表分布:

| 版本 | 平均 KL | 最大 KL | top-1 一致 |
|---|---|---|---|
| per-token scale | 1.10e-3 | 1.04 | 99.35% |
| **per-group scale(现在)** | **4.4e-4** | 0.34 | **99.54%**(1529/1536) |
| 提议的门限 | ≤ 1e-3 | — | ≥ 99.7% |

→ **int8 未通过 top-1 门限**,因此保持 opt-in(`--fast`),默认走精确路径。
72 个位置的小测试(KL 8e-5)掩盖了误差随上下文累积的问题,必须用长语料评估。

前 8K token 上的逐矩阵敏感度(`KATA_I8_EXCLUDE`,被排除的矩阵走 fp16):

| 排除 | 平均 KL | top-1 |
|---|---|---|
| 无 | 1.04e-4 | 511/512 |
| down | 7.8e-5 | 511/512 |
| down, ssm_out, wo | 6.2e-5 | 510/512 |
| gate, up, down(整个 FFN) | 5.4e-5 | 511/512 |

没有哪一类矩阵占主导(排除整个 FFN 也只改善 48%):误差来自 7 位激活量化本身,均匀分布。两项 int8(高低位分解)的计算量与 fp16 持平,没有意义。

现在精确路径已贴近矩阵单元峰值,int8 相对精确路径只剩 ~10% 优势,所以不建议使用。

## 4. 已知限制与下一步

- **冷启动**:每个进程第一次预填充要多花 ~2.3 s(缺页:6.4 GB 映射第一次被 GPU 访问;GPU 端首次 dispatch 的开销)。测试过:启动时用 CPU 把所有页触摸一遍要 1.7 s,首次预填充只降到 2.1 s,净收益为负;Metal 二进制归档也没有可测收益(管线创建 <20 ms)。两项实验均已撤销。
- **展开仍占 0.56 s/chunk(14%)**。试过**重叠**:并发 dispatch + 双缓冲暂存区,把"下一个矩阵的展开"和"当前 GEMM"放在同一个屏障区间(结果逐位一致,没有竞争)。结果**更慢**:1024 token 从 3.9 s 变成 5.6–5.8 s。并发编码器加全屏障(不重叠)和串行一样快(3.9–4.0 s),所以慢来自重叠本身:展开每个矩阵写 178 MB,冲掉 M 优先顺序下 GEMM 依赖的缓存复用。已撤销。剩下的办法是融合反量化进 GEMM(每个 M 块重复解码,需要大幅重写)。
- **长上下文注意力**:flash 内核只有 3.1–3.3 TFLOPS(M=16 的小块)。32K 提示的注意力约 67 s,值得继续调(每 simdgroup 32 行、让 P 留在寄存器里作为左输入)。
- 短提示的预填充(≤ 64 token)仍然要整套展开(~0.9 s),比逐 token 解码还慢一些情况存在(逐步路径阈值为 24 token)。
- 精确路径用 fp16 激活;`verify-check` 显示它与 fp32 解码的差异在 1e-6 量级。
