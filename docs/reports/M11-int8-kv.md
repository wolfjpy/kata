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
- 旧的标量注意力内核(`BONSAI_SCALAR_*`)只支持 fp16 KV,int8 模式下被忽略。
- 评测语料是论文/文档/规范的混合文本;需要更长(32K+)的真实对话/代码语料才能得出更窄的置信区间。

## Correction (M29d, re-measured)

The "KL 1.3e-4 / top-1 99.71%" figure above does not reproduce with a cross-process reference. The paired `kl-eval` mode
is invalid for int8 KV because `kvInt8` decides the buffer allocation at load time, so mixing both paths in one process
only proves that the layout change alone corrupts the result. Measured properly:

```
BONSAI_KV=fp16  bonsai kl-eval ... --save-ref ref.bin     # reference
BONSAI_KV=int8  bonsai kl-eval ... --ref ref.bin
-> 1536 positions | mean KL 4.374e+01 nat | max KL 4.709e+04 | max |logit diff| 1.68e+07 | top-1 1527/1536 (99.41%)
```

Large but sparse: top-1 survives almost everywhere while individual positions blow up, which points at the chunked
prefill attention path (`k_flash_prefill_q8`) rather than at the quantiser — the single-token decode path produces
coherent text. Until that kernel is fixed, int8 KV stays opt-in and is not a default.
