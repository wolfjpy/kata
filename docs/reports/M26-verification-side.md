> 本文以中文写成;**英文全译:[M26-verification-side.en.md](M26-verification-side.en.md)**,下方 `## Summary (EN)` 为摘要,双语文档索引见 [docs/README.md](../README.md)。
> *This document is in Chinese. An English summary follows; the bilingual index is at [docs/README.md](../README.md).*

## Summary (EN)

**Goal.** Complete the verification side: verification at arbitrary width, state snapshots, and commit.

**What was built.**
- `Engine::verifyWide(toks, T, pos, argmaxOut)`: T <= 8 tokens go through the trunk in one pass. Every PQ1_0 matrix goes through `k_wide`, including the shared `lm_head` (278 MB read once, producing 8 rows of logits).
- `Engine::commitWide(accept)`: pick the snapshot, copy it back into the live state, rebuild the conv state from the window's first `accept` rows with the new `k_gdn_conv_snap` kernel (so no extra snapshot is needed), and top up the MTP cache for `pos+1 .. pos+accept`.
- Test entry `--vwidth T`: verify T rows per round (dummy drafts here) and accept only row 0, so the output must match plain greedy decoding token for token.

**Correctness.**

| check | result |
|---|---|
| `--vwidth 8` (accept 1) against plain greedy | **token-identical** (550, 220, 16, 13, 561) |
| `--vwidth` against the `verify2` path | identical sequence |
| 9 snapshots + conv rebuild + MTP top-up across rounds | identical (the sequence above spans 5 rounds) |

---

# M26:验证侧完整化(任意宽度验证 + 状态快照 + 提交)

日期:2026-10-25 · 对应 `R2-dflash2-revisit.md` 的 M26。前置 M25(宽验证内核,r=2.47)。

## 1. 做了什么

- **`Engine::verifyWide(toks, T, pos, argmaxOut)`**:T ≤ 8 个 token 一次走完主干。每个 PQ1_0 矩阵(含共享的 lm_head,278 MB 一次读、产 8 行 logits)都由 `k_wide` 处理;GDN 递推用新内核 `k_gdn_rec_snap`,每个"已接受前缀长度"存一份状态快照(9 份,9 × 147 MB);注意力用矩阵单元的 flash 内核;其余批量算子复用 prefill 的版本(T 行)。
- **`Engine::commitWide(accept)`**:选快照 → 拷回活状态;用新内核 `k_gdn_conv_snap` 从窗口前 `accept` 行的 qkv 重建 conv 状态(不需要额外快照);把 MTP 缓存补到 `pos+1..pos+accept-1` 行并更新 `hPrev`。
- **测试入口 `--vwidth T`**:每轮验证 T 个草稿(这里用哑草稿)、只接受第 0 行 → 输出必须与普通贪心解码逐 token 一致。

## 2. 正确性

| 检查 | 结果 |
|---|---|
| `--vwidth 8`(接受 1)对普通贪心 | **逐 token 一致**(550, 220, 16, 13, 561) |
| `--vwidth` 对 `verify2` 路径 | 序列完全一致 |
| 9 份快照 + conv 重建 + MTP 补齐后继续下一轮 | 一致(上面的序列跨了 5 轮) |

## 3. 每轮成本(72 token 上下文,贪心,24 轮取平均)

| T | 每轮 | 相对普通解码(50.1 ms/token) |
|---|---|---|
| 2 | 156 ms | 3.1× |
| 4 | 159 ms | 3.2× |
| 6 | 161 ms | 3.2× |
| 8 | **167 ms** | **3.3×** |

**成本几乎与 T 无关**,与"权重只读一遍"的预期一致 ✓。

T=8 的分解(用 `--skip` 消融):

| 项 | 时间 |
|---|---|
| 宽 GEMV(全部矩阵,含 lm_head) | 126 ms |
| 注意力(T=8 行) | 10.3 ms |
| alpha/beta 的 GEMM(每层只有 2 个 threadgroup) | 6.0 ms |
| 其余(每层 2 次 norm、3 次 FWHT、conv/qk/gates/gnorm、argmax) | ~25 ms |

小网格延迟是后两项的主因:72 token 上下文下,注意力只有 4 个 threadgroup;alpha/beta 的 GEMM 只有 2 个。这些是**可以修的**(见 §5)。

## 4. 对 DFlash2 的意义(用实测接受长度)

DFlash2 K=7(R2 实测:代码/推理 5.20 token/轮,散文 2.07),草稿 Q4_K_M 约 9 ms:

| 配置 | 一轮 | 代码/推理 | 散文 |
|---|---|---|---|
| 现在(167 + 9) | 176 ms | **29.5 tok/s**(今天 25,+18%) | **11.8 tok/s**(−53%) |
| 修掉小网格开销(~20 ms) | 156 ms | 33.3(+33%) | 13.3 |
| 如果宽 GEMV 还能到 r=1.6 | 106 ms+9 | 49(+96%) | 19.5 |

这比 `R2` 的估计(48–59 tok/s)低,原因是宽内核落在 r=2.47(而不是 1.2–1.6),而且每轮的固定开销实测 41 ms(估的是 16 ms)。**所以 DFlash2 的收益在 +18%~+33%(仅代码/推理),散文必须靠 M29 的宽度自适应回退,否则 −53%。**

## 5. 下一步(按收益/成本排序)

1. **小网格延迟修复(~20 ms/轮,12%)**:alpha/beta 改成 T 行的 GEMV 内核(96×T 个 threadgroup,而不是 2 个);注意力改成按 key 分段(像解码路径的 flash-decoding)让 8 行也有足够并行度;norm/FWHT 把每层的两个 norm 合成一次 dispatch。
2. **宽 GEMV 再压**:M25 已扫完 tile/占用率/双缓冲/混合路径;剩余唯一未试的是"B 操作数从寄存器喂 matmul2d"(`get_right_input_cooperative_tensor`),需要按 lane 的元素映射重排权重,成本高。
3. **DFlash2 移植(M28)**:按 `refs/ninfer/docs/maintainer/dflash.md` 的合同做 5 层草稿 + 选择器 + Q4_K_M 反量化;先只支持 K=7。
4. **M29 宽度自适应**:散文自动退回 MTP-1(25 tok/s)。

## 6. 现状判断(诚实版)

- 验证侧的基础设施是**对的、可测的、可复现的**:任意宽度、状态快照、提交、MTP 补齐全部工作,并且用"必须复现普通贪心"这条硬标准验证过。
- 但**每轮 167 ms 里只有 126 ms 是不可避免的权重读取**,其余 41 ms 是小网格延迟;而且 126 ms 本身是带宽下限(40 ms)的 3 倍——瓶颈仍是 M25 的矩阵单元吃入率。
- 因此 DFlash2 的价值目前是"代码/推理 +18~33%",不是 R2 期待的 +100%。是否继续投入移植(M28,3–5 天),取决于你认为这 18~33% 值不值;如果不值,建议先把 §5.1/§5.2 做完再看(那两项能把上限抬到 +33~96%)。
