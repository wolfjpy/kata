# M13:FFN gate+up 的 int8 预填充开关(`--fast-ffn`,有损,默认关闭)

日期:2026-10-25 · 对应 CHECKPOINT-003 路线图 M13,也是用户确认的决定之一。

## 1. 结果

| | 精确 fp16(默认) | `--fast-ffn`(gate+up int8) | `--fast`(全部 int8) |
|---|---|---|---|
| 预填充 1024 token chunk | 255–263 tok/s | **274–280 tok/s(+7–8%)** | 281 tok/s(+9–10%) |
| 预填充 7.3K token 提示 | 207 tok/s | **226.5 tok/s(+9.5%)** | 229 tok/s(+10.7%) |
| 质量(24.5K token 语料,对精确路径) | — | **平均 KL 7.5e-5,top-1 99.87%**(3068/3072),最大 KL 0.039 | 平均 KL 4.4e-4,top-1 99.54% |
| 是否过质量门限(KL ≤ 1e-3、top-1 ≥ 99.7%) | — | **过,有余量** | 不过 |

FFN 的 gate 和 up 占每层 FLOPs 的 ~47%,只对它们做 int8 就拿到了全部 int8 收益的约 80%,误差却小了一个数量级(平均 KL 小 6 倍,最大 KL 小 9 倍)。依据是 CHECKPOINT-003 §3.5 的帕累托:误差主要来自喂给 GDN 循环状态的输入投影(qkv/z/wq),FFN 的 gate/up 输出还要经过 SwiGLU 和 down 投影,对激活量化噪声不敏感。

## 2. 实现

- `Engine::setI8(spec)`:`"all"`(每个矩阵)、`"ffn"`(gate+up)、或逗号列表(`qkv z ssmout wq wk wv wo gate up down`),空串/`none` 关闭。每个 `dExpandGemm` 调用按矩阵名字决定走 int8 还是精确 fp16;其余矩阵不受影响。
- **只对真正会走 int8 的 GEMM 做激活量化**:`fusedNorm`/`fwhtB` 现在带 `needQ`,只有当这份激活的消费者里有 int8 矩阵时才跑 `k_quant_rows`(`--fast-ffn` 时只有 post-attention 归一化之后那一次)。
- 数值标签带上矩阵集合(`int8:gate+up`),所以前缀缓存里不同模式的检查点互不混用。
- 命令行:`run`/`chat` 的 `--fast-ffn`、`--fast`(全部)、`--i8 a,b,c`(开发用)、`--exact`;`kl-eval` 的 `--fast-ffn`/`--i8`;开发入口 `--i8`。原来未文档化的 `KATA_I8_EXCLUDE` 环境变量已删除。

## 3. 注意

- 有损:预填充用 int8 激活(每 token、每 128 元素组一个 scale)乘三值权重;误差只进入 gate/up 的输出,经 SwiGLU/down 投影后进入残差流。KV 和 GDN 状态都是在这条路径下产生的,所以影响会带到后面的解码(这也是评测用 24.5K token 的长语料而不是几十个位置的原因)。
- 只影响预填充;解码路径不变。
- 在 7K token 以内的短提示上,收益被固定开销(展开 0.56 s/chunk)稀释。
