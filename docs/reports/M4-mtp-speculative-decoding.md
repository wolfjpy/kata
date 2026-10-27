> 本文以中文写成;下方 `## Summary (EN)` 是英文摘要,双语文档索引见 [docs/README.md](../README.md)。
> *This document is in Chinese. An English summary follows; the bilingual index is at [docs/README.md](../README.md).*

## Summary (EN)

**Goal.** Speculative decoding with the model's own MTP head, at depth 1 with a 2-wide verification.

**Result.**

| | plain decode | MTP speculative |
|---|---|---|
| greedy (temp 0), 200 tokens | 20.0 tok/s | **25.0 tok/s (+25%)** |
| sampled (temp 1, top-k 20, top-p 0.95, min-p 0.05) | 19.8 tok/s | **22.7 tok/s (+15%)** |
| predictable text (multi-turn, factual Q&A) | — | 29-31 tok/s (82-90% acceptance) |
| baseline (Prism llama.cpp `tg128`) | 15.6 tok/s | — |

A cycle at 60% acceptance and 1.60 tokens per cycle costs 5.7 ms of drafting plus 58.2 ms of verification, about 64 ms.

**Quality.** `verify2` against exact single-token decoding over 830 positions: **mean KL 4.3e-6, max KL 8.4e-4, top-1 830/830**. The greedy text is character-identical to plain decoding for the first 690 characters, after which a near-tie flips between "rolled tight" and "rolled so tightly" — both coherent, and the two paths had identical logits to within 5e-4 there.

---

# M4:MTP 投机解码(深度 1,验证宽度 2)

日期:2026-10-24 · 提交 `c435af5`(MTP 头)、`f7f3796`(投机解码)

## 1. 结果

| 项目 | 普通解码 | MTP 投机解码 |
|---|---|---|
| 贪心(temp 0),200 token | 20.0 tok/s | **25.0 tok/s**(+25%) |
| 采样(temp 1, top-k 20, top-p 0.95, min-p 0.05) | 19.8 tok/s | **22.7 tok/s**(+15%) |
| 可预测文本(多轮对话、事实问答) | — | 29–31 tok/s(接受率 82–90%) |
| 基线(Prism llama.cpp tg128) | 15.6 tok/s | — |

每个周期(贪心,接受率 60%,1.60 token/周期):草稿 5.7 ms + 验证 58.2 ms ≈ 64 ms。

质量:`verify2` 对比精确的单 token 解码,830 个位置:**平均 KL 4.3e-6,最大 KL 8.4e-4,top-1 一致 830/830**。
贪心输出在前 690 个字符与普通解码逐字一致,之后在 "rolled tight" / "rolled so tightly" 处出现一个两者都通顺的近似平局翻转(见 §5)。

## 2. 算法

每个周期(状态:已喂入 `n` 个 token,`next` = 已采样但未喂入的位置 `n` 的 token,`hPrev` = 位置 `n-1` 的隐状态):

1. **草稿**:MTP 头处理 1 或 2 行(第二行是上一周期接受时"欠"的那一行),预测 `d` = 位置 `n+1` 的 token。
2. **验证**:主模型一次处理 `[next, d]`(位置 `n`, `n+1`),每个权重矩阵只读一遍。
3. **采样**:对第 0 行 logits 用正常采样器采样得到 `s0`。
   - `s0 == d`:接受两个,再对第 1 行采样得到 `s1` 作为新的 `next`。
   - 否则只提交 `next`,`s0` 成为新的 `next`。

这是**确定性草稿的精确投机采样**:采样出的 `s0` 服从目标分布;当它恰好等于 `d` 时接受,接受概率就是 `p(d)`,拒绝时得到的 `s0` 服从"去掉 `d` 后重新归一化的 `p`"。输出分布与不用投机完全相同,温度、top-k/p、min-p 都照常工作,不需要拒绝采样的修正项。

### 状态回滚:三缓冲,零拷贝

GDN 状态(48 层 × 3.1 MB = 151 MB)在验证时有两个可能的终点(处理完 token 0 / 处理完 token 1)。`k_gdn_rec_v2` 读入输入状态一次,把两个结果分别写入另外两个缓冲;接受/拒绝只是 `stCur` 指针的选择(`commit2`)。conv 状态同理。
代价:多写一份 151 MB(约 +1.1 ms/周期)。注意力 K/V 不需要回滚:拒绝时位置 `n+1` 的 K/V 是脏的,但下一个周期在读取之前就会先覆盖它。

### MTP 缓存的维护

- 预填充(≥25 个 token):主干之后用 `eh_proj`/`wk`/`wv` 三个批量 GEMM 一次写入这一块所有位置的 MTP K/V(`Engine::encMtpKv`,与预填充同一个 command buffer)。
- 逐步喂入(≤24 个 token):逐步收集隐状态,最后一次性写入 MTP K/V。
- 修复了一个旧问题:助手回合结束时的 `<|im_end|>`(以及因 `--max-tokens` 提前停止时尚未喂入的最后一个 token)现在会被喂进缓存,多轮对话的上下文与模板一致。

## 3. 关键工程发现:T≥2 的验证 GEMV 是 ALU 受限(T=1 仍是带宽受限,见 M3)

第一版 `verify2` 每周期 85 ms(预期 ~52 ms),MTP 反而比普通解码慢(17.9 vs 20 tok/s)。消融:

| 跳过 | verify2 耗时 |
|---|---|
| 无 | 85.3 ms |
| 全部 GEMV | 12.3 ms |
| GDN 递推 | 81.3 ms |
| alpha/beta | 78.9 ms |
| 注意力 | 84.7 ms |

→ GEMV 占 73 ms(单 token 版本只有 45 ms)。

`bench_gemv2` 在真实张量上的微基准(T = 同时处理的 token 数):

| kernel | ffn_gate 5120→17408 |
|---|---|
| fp32 g/c 形式,T=1 | 154 µs(128 GB/s) |
| 同上,T=2 | 245 µs(80 GB/s) |
| 同上,T=3 | 459 µs(42 GB/s) |

T=1→2 增加 59%,T=3 增加近 200%:**ALU 受限**。按操作数估算 T=2 约 6 条指令/权重,总量 ≈ 2.1 Tops/s,已经顶到 GPU 的标称 FP32 吞吐。

`alu_probe`(`tools/alu_probe.cpp`)测硬件指令吞吐:

| 指令 | 吞吐 |
|---|---|
| fp32 FMA | 1.22 T/s |
| fp16 标量 FMA | 1.66 T/s |
| **half2 FMA** | **2.35 T 元素/s(约 1.9× fp32)** |
| half4 FMA | 3.12 T 元素/s |
| fp32 mul-add + floor | 与单独的 FMA 同速(**floor 几乎免费**) |
| fp32 mul-add + fract | 慢约 2×(fract 不免费) |
| int32 mul-add | 与 fp32 FMA 同速 |

据此尝试"字节→5 个 trit 的 half 查表 + 两个 token 打包成 half2"。消融(`gemv_exp.metal`,位 0=去掉 LUT 读取,位 1=去掉激活读取,位 2=去掉权重读取):

| 变体 | 耗时 |
|---|---|
| 2 字节 LUT 读取(每行每块每 lane 16 次) | 267 µs |
| 去掉 LUT 读取 | **146 µs**(= 带宽上限,133 GB/s) |
| 去掉激活读取 | 245 µs |
| 去掉权重读取 | 193 µs |

→ 瓶颈是 TG 内存里 2 字节的随机读。改成 **16 字节条目、每个权重字节一次向量读**(5 个 half trit 打包成 3 个 32 位字,TG 内存 4 KB):

| kernel | 三个张量 |
|---|---|
| fp32 g/c 形式,T=2 | 245 / 251 / 147 µs |
| **LUT + half2,T=2** | **170 / 174 / 101 µs**(−33%,113 GB/s) |

每 5 项部分和 flush 到 fp32(`float2(a0)+float2(a1)+…`),比整块 half 累加精度好,速度相同。

另一个发现:alpha/beta 用 `matmul2d`(M 填充到 32、只有 2 个 TG)耗 6.4 ms,换成专用 2-token bf16 GEMV(`k_gdn_ab2`)后几乎为零。

最终 `verify2` = 57.8 ms:GEMV 51 ms(113 GB/s)+ 其余 6.8 ms(GDN 递推 3.8 ms)。

## 4. 数值与质量

`kata verify-check` 逐位置对比 `verify2` 的 logits 与精确单 token 路径:

| 路径 | 位置数 | 平均 KL | 最大 KL | 最大 logit 差 | top-1 |
|---|---|---|---|---|---|
| LUT + half2(生产) | 66(72 token 提示) | 2.6e-7 | 1.2e-6 | 0.008 | 66/66 |
| LUT + half2(生产) | 830 | 4.3e-6 | 8.4e-4 | 1.02 | 830/830 |
| fp32 数学 + fp16 激活(`KATA_VERIFY_V1H=1`) | 830 | 1.6e-6 | 2.4e-4 | 0.44 | 830/830 |

误差来源:激活用 fp16(与预填充路径相同)贡献大部分;half 累加在此之上约 ×2.7。离群点与上下文长度无关,取决于具体内容。
均在提议的质量门限(KL ≤ 1e-3、top-1 ≥ 99.7%)之内,但**这是一个有损选项,需要用户确认门限**。

## 5. 已知限制

- 验证路径的数值与普通解码路径有 ~1e-6 量级的 KL 差异,因此贪心输出偶尔会在近似平局处分叉(上面的例子)。这不是错误;若要求逐 token 一致需要 fp32 激活的 2-token 内核(73 ms,比不投机还慢),不值得。
- 草稿深度固定为 1。分析:深度 2 需要多一次 5.8 ms 的 MTP 加上宽度 3 的验证(`half2` 装不下 3 个 token),预计与深度 1 持平,所以没做。
- 采样器仍用 `partial_sort` 遍历 248K 词表,约 1 ms/周期(占 1.5%)。
- 草稿的 lm_head(278 MB)占草稿成本的 ~38%;缩小到高频词表子集可省 ~2 ms/周期(3%),未做。

## 6. 下一步(按收益排序)

1. **矩阵单元 flash attention(预填充)**:当前预填充随上下文严重退化(见下),长上下文不可用。
2. 融合反量化的预填充 GEMM(当前展开写出 24 GB 的 int8 权重,占 16%,并且 GEMM 在真实形状上只有 21.6 TOPS,峰值 36)。
3. 层优先(layer-major)预填充:长提示下权重只展开一次。
4. 解码:验证 GEMV 从 113 GB/s 推到 ~125 GB/s(−5 ms/周期)、草稿词表裁剪、fp16 GDN 状态。

附:预填充随上下文的退化(int8,`--fast`)

| 提示长度 | 速度 |
|---|---|
| 1024 | 278 tok/s |
| 3639 | 201 tok/s |
| 7279 | 176 tok/s |

## 7. 补充:长上下文解码(flash-decoding)

7.3K token 上下文,贪心 64 token:

| | 普通解码 | MTP |
|---|---|---|
| 短上下文 | 19.8 tok/s | 25.0 tok/s |
| 7.3K 上下文,旧标量注意力 | 17.8 tok/s | 20.3 tok/s |
| 7.3K 上下文,矩阵单元 flash-decoding(默认) | 17.6–17.9 tok/s | **21.4–21.7 tok/s** |

`k_attn_dec_split` / `k_attn_dec_combine`(`shaders/flash.metal`):同一 KV 头的 6 个头 × 1–2 个 token = 最多 12 行,凑成一个 16 行的 `matmul2d` 块;按 128 个 key 一段拆成多个 split(每段一个 simdgroup),最后合并并乘 sigmoid 门。验证 2 个 token 时只读一遍 K/V(旧内核读两遍),所以 MTP 路径有 ~5% 提升;普通解码持平。数值与旧内核等价(72 个位置 KL 7e-9)。

用 `KATA_SKIP=k_attn_split` 整个跳过注意力:7.3K 上下文只多出 ~5 ms/token(17.1 → 18.7 tok/s)。纯带宽探针(`tools/kv_read_probe.cpp`,同样的访问模式)显示:K/V 的读取在 7.3K 个 key 时本身就要 205 µs/层 × 16 层 = **3.3 ms**(每个 key 每层 K+V 共 4 KB,每 token 64 KB),并且跨步布局(`[pos][4 头][256]`)与按头连续布局没有区别(都是 ~145 GB/s)。所以两种注意力内核都已经在带宽的 ~65%,继续优化内核没有意义;32K 上下文的 KV 读取理论值是 13 ms/token(权重 40 ms 之外的 +30%)。缩小这部分只有一条路:压缩 KV(fp8/int8,有损,需要质量门限),见下一步。

新增环境变量:`KATA_SKIP=a,b`(CLI 运行时的算子消融)、`KATA_SCALAR_DEC=1`(旧的标量解码注意力,用于对照)、`KATA_DEC_CHUNK`(split 的 key 数,默认 128)。
