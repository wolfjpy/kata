> 本文以中文写成;下方 `## Summary (EN)` 是英文摘要,双语文档索引见 [docs/README.md](../README.md)。
> *This document is in Chinese. An English summary follows; the bilingual index is at [docs/README.md](../README.md).*

## Summary (EN)

**Goal.** Revisit DFlash2. This document replaces the verdict of `M14-wide-verify-gate` (whose *measurements* remain valid) and rewrites the "do not do it" line about DFlash2 in `REVIEW-01` §4.

**Three methodological errors, admitted up front.**

1. **I used T=2 as the reference instead of the bandwidth floor.** A T=8 verification must read the 5.657 GB of weights once, so at 142 GB/s **39.8 ms is an irreducible lower bound**. The compute demand of T=8 is 8 x 48.7 GFLOP = 390 GFLOP, which at this machine's 14.7 TFLOPS is 26.5 ms — *still below the memory floor*. Comparing against T=2 (170 us) rather than against that floor made the gate far stricter than the physics requires.
2. **I framed it as a pass/fail gate rather than a "how cheap can it be" question.** The right question is what the cost floor of one round is and how fast it must be to cover the drafting cost; framed that way, the bar is much lower (see §4 of the report).
3. **The kernel I drew the conclusion from had four fixable flaws** (§3). Two of them — one `matmul2d` call per K=128 chunk, and M=8 — account for a 3x gap against the production prefill GEMM, which runs the whole K=5120 in one call at 14.7-16.6 TFLOPS.

**Verdict.** DFlash2 is reopened; M25 (wide verification kernel) and M26 (verification side) follow.

---

# R2:重新研究 DFlash2 —— 上一轮的结论错了,现在重开

日期:2026-10-25 · 取代 `reports/M14-wide-verify-gate.md` 的结论(该文档的测量数据仍然有效),也改写 `REVIEW-01` §4 "不建议做"里关于 DFlash2 的那一行。

## 1. 我错在哪里(三条,都是方法错误)

**(1) 我拿 T=2 当基准,而不是带宽下限。**
T=8 验证至少要读一遍 5.657 GB 权重 → 142 GB/s 下 **39.8 ms 是不可压缩的下限**。而 T=8 的算力需求是 8 × 48.7 GFLOP = 390 GFLOP,在本机 14.7 TFLOPS 下是 26.5 ms —— **仍然小于 39.8 ms,所以 T=8 验证本质上仍是带宽受限**,理论上可以和 T=2 一样快。我却把"我的内核是 T=2 的 2.3 倍"当成了硬件极限,这是把实现能力当成了物理规律。

**(2) 我把它设成"过/不过"的闸门,而不是"能不能便宜"的问题。**
正确的问法是:验证一轮的**成本下限**是多少、需要多快才能覆盖草稿的成本。用这个框架,要求的门槛比我的闸门低得多(见 §4)。

**(3) 我用来下结论的那个内核,犯了四个可以修的错**(§3),其中"每次 `matmul2d` 只做 K=128"和"M=8"这两条,与生产预填充 GEMM(一次调用跑完整个 K=5120、14.7–16.6 TFLOPS)的差别就是 3 倍以上。我用一个下午的原型否掉了一条可能的 2–3 倍加速路线,这不是谨慎,是偷懒。

## 2. 重新算账

| 量 | 数值 |
|---|---|
| 一轮验证要读的权重 | 5.657 GB |
| 带宽下限 | 39.8 ms(142 GB/s) |
| T=8 的算力需求 | 390 GFLOP → 26.5 ms @14.7 TFLOPS(仍带宽受限 ✓) |
| T=4 的算力需求 | 195 GFLOP → 13.3 ms(远低于下限 ✓) |
| 状态:8 份 GDN 快照(代替三缓冲) | 8 × 147 MB 写 ≈ 8.4 ms;内存 1.2 GB(16 GB 机器上可接受,比 ReplaySSM 重放 5 步 × 2.6 ms 更便宜) |
| T=8 的注意力(7.3K 上下文) | 约 8–13 ms(8 个 query 行) |
| DFlash2 草稿一次前向 | Q8_0 约 16 ms / Q4_K_M 约 9 ms(1.42–2.0 GB 权重) |

所以一轮 T=8 的**理论**成本是 `40 + 8.4 + 12 + 3 + 草稿 9~16 ≈ 72–80 ms`。

## 3. 我的原型内核为什么只有 47 GB/s

`k_wide_sg`(M14):ffn_gate 415 µs,消融显示解码 134 µs + MMA 281 µs 串行。四个错:

| 错 | 证据 | 修法 |
|---|---|---|
| **每次 `matmul2d` 只覆盖 K=128** | 生产预填充 GEMM 一次调用覆盖整个 K=5120,在同样硬件上达到 14.7–16.6 TFLOPS;而每次调用有约 30 ns 的固定开销,K=128 时算力时间只有 ~5 ns → 90% 是开销 | 每次调用覆盖 K=256–512(受 32 KB threadgroup 预算限制),固定开销摊薄 2–4× |
| **M=8 用了 8 行 tile** | 形状扫描里 M=16 的每权重吃入率比 M=8 高约 1.2×,而 M=8 只有 8 个真实 token | 把 T=8 补到 **M=16**:多一倍 FLOPs 但仍是带宽受限,反而更快;T=4 也补到 M=8/16 |
| **N=R=16(每 simdgroup 只算 16 行)** | N=32 明显更好,N=64 反而变差(threadgroup 装不下,只能降低 K) | N=32 + K=256(20 KB)或 N=16 + K=512(17 KB),实测二选一 |
| **解码与 MMA 串行** | 解码 134 µs + MMA 281 µs = 415 µs(完全相加) | 双缓冲 threadgroup tile:解码第 k+1 块时让矩阵单元消化第 k 块,取 max 而不是 sum |

另外两个已经踩过的坑保留:threadgroup 行跨度必须填充(`ld=128 → 136` 是 1.6×,银行冲突);尾块要用真实 extent 避免 `NaN×0`。

诚实说明:今天在 `/tmp` 写的形状微基准**不稳定**(同一个形状两次测得 2.19 和 8.40 TFLOPS;>17.3 TFLOPS 说明依赖链被编译器重排了),所以我不拿它当数据,只用它做定性判断(填充、M、K 的方向),真正的数字必须来自**真实内核 + 真实张量**。

## 4. 今天新测的关键曲线:接受长度 vs 草稿宽度

用 Prism 的 `llama-speculative-simple`(它的 Metal 投机路径很慢,但接受统计有效),目标模型 Bonsai-2-27B,DFlash2 Q4_K_M 草稿,贪心:

| 草稿宽度 K | 验证宽度 T=K+1 | 代码:token/轮 | 散文:token/轮 |
|---|---|---|---|
| 1 | 2 | 1.92 | — |
| 3 | 4 | **3.24** | 1.97 |
| 5 | 6 | **4.68** | — |
| 7 | 8 | **5.20** | **2.07** |

(K=1 就是"验证宽度 2",恰好等于我们 MTP 深度 1 的形状;DFlash2 在同等宽度下比 MTP 略好。)

这说明:**收益随宽度单调增长,而验证成本在带宽受限时几乎与宽度无关** → 一旦验证便宜,越宽越好。K=7 比 K=3 多 60% 的 token,只多一次草稿的量(约 +3 ms)。

## 5. 修正后的收益表(代码/推理文本,草稿宽度 7)

`verify ≈ 51·r + 15`(51 ms 是今天 T=2 的 GEMV 部分,`r` 是 T=8 相对 T=2 的倍数;+15 是状态/注意力/杂项),草稿取 Q4_K_M 的 9 ms:

| 验证内核状态 | r | 一轮 | 代码/推理 tok/s | 散文 tok/s | 相对今天的 25 tok/s |
|---|---|---|---|---|---|
| 今天的 T=2 基线 | 1.0 | 75 ms | **69** | 28 | 2.8× / 1.1× |
| 修好四个错(目标) | 1.2–1.6 | 86–108 ms | **48–59** | 19–24 | 1.9–2.4× / 0.8–1.0× |
| M14 那个原型(未修) | 2.3 | 145 ms | 36 | 14 | 1.4× / 0.6× |

两条读数:
- **代码/推理方向即使按未修的原型也是净赚**(36 vs 25)。我之前说"不值得"是错的。
- **散文是唯一需要担心的**:r > 1.6 时散文会变慢。解决方式不是放弃 DFlash2,而是**宽度自适应**(按 Strata 的 DraftPolicy 思路:用在线 EMA 学每个宽度的每毫秒产出,散文自动退回宽度 2 的 MTP 或普通解码)。这也解释了 Strata 的 spec 深度在散文上停在 2。

## 6. 修正后的路线(取代 M14/M15/M16 的结案)

**M25:宽验证内核(2–3 天,唯一的技术风险集中在这里)**
1. 真实内核 `k_wide_v2`:M=16 补齐、N=32、每次调用 K=256(threadgroup 20 KB)、双缓冲、LUT 解码、权重 tile 精确 fp16(或 int8 三值 + 每 128 组行 scale)。
2. 验收:**ffn_gate/ffn_down/gdn_qkv 上 r ≤ 1.6**(即 170 → ≤ 272 µs 的等效,target 200 µs);数值与 fp32 参考误差 ≤ 1e-5。
3. 顺带:8 份 GDN/conv 快照(替代三缓冲)、T 行 KV 写入(已有)、T 行 MTP K/V。

**M26:验证侧完整化(1 天,不含草稿模型)**
- 目标侧接受任意宽度:`verify(T, toks)` 产出 T 行 logits/hidden,按贪心前缀接受,提交对应快照;用**哑草稿**(重复/随机 token)测出真实的"每轮成本"。
- 验收:T=4/6/8 的一轮成本 ≤ 下表(r=1.6 时 ≤ 108 ms);与逐 token 解码的 KL ≤ 1e-5。

**M27:MTP 链(半天,零新模型)**
- 用现有 MTP 块做链式草稿(把上一行的输出 hidden 当作下一行的输入),K=3 → T=4 验证。Prism 实测 3.13 token/轮(推理)。
- 预期:推理文本 25 → 38–45 tok/s。这是**不引入 DFlash2 也能拿到的中间收益**,并且验证 M25/M26 的正确性。

**M28:DFlash2 移植(3–5 天)**
- 草稿侧:5 层(plain RMSNorm、QKV、逐头 qk norm、scalar RoPE、非因果滑窗 2048 的环形 KV、双抽头动态分组卷积、SwiGLU),目标第 5/19/33/47/61 层残差作条件,上下文物化器,top-16 + rank-256 选择器;Q8_0 先行(我们已有 Q8_0 GEMV/展开),Q4_K_M 作为后续优化。
- 验收:与 llama.cpp `draft-dflash` 的接受长度一致(±0.3);端到端 代码/推理 ≥ 45 tok/s、散文 ≥ 22。
- 风险:选择器与动态卷积的细节多(ninfer 的 `docs/maintainer/dflash.md` 给了完整合同,`include/ninfer/ops/*.h` 有精确的数值约定);Q4_K_M 解量化要新写内核。

**M29:宽度自适应(半天)**
- 在线估计每个宽度的 token/ms,散文自动退回窄宽度;避免 r 不理想时的回归。

## 7. 需要的数据/决定

1. **M25 的内核目标**是否接受"r ≤ 1.6"作为继续的条件(对应代码 48–59 tok/s、散文由 M29 兜底)。
2. **Q4_K_M 还是 Q8_0 草稿**:Q8_0 我们有现成内核、省 1–2 天,但草稿多 ~7 ms/轮(Q8 16 ms vs Q4_K 9 ms)。建议先 Q8_0 打通,再按收益决定是否加 Q4_K。
3. 内存:8 份状态快照 +1.2 GB,DFlash2 草稿权重 1.42–2.0 GB。64K 上下文 + int8 KV 下总占用约 9–10 GB,仍在 12.1 GB 工作集内;是否需要限制最大宽度以省内存。
4. 如果 M25 拿不到 r ≤ 1.6:仍可用 r≈2 的版本 + M29 自适应(代码 +44%、散文不降),只是收益小一半。是否接受这个下界线。

## 8. 再次强调的两个教训(写给下一次的自己)

- **判断"某条路值不值得"时,先算出它的物理下限,再和自己的实现比较**:下限和实现差 3 倍时,结论应该是"我的实现差 3 倍",而不是"这条路不通"。
- **微基准在依赖链上不可靠**:同形状两次差 3.8 倍、超过理论峰值的读数,说明测量本身有问题;只有真实内核 + 真实数据 + 交错对照才能下结论。
