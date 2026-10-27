> 本文以中文写成;下方 `## Summary (EN)` 是英文摘要,双语文档索引见 [docs/README.md](../README.md)。
> *This document is in Chinese. An English summary follows; the bilingual index is at [docs/README.md](../README.md).*

## Summary (EN)

**Goal.** Step two of the DFlash2 port: the draft module (loading, ops, geometry), plus the exact to-do list that M28c then worked through.

**Authoritative graph**, taken from `refs/llama.cpp-prism/src/models/dflash.cpp` rather than inferred from prose (the drafter is a plain 5-layer model; there is no dedicated Metal kernel for it):

- **Context encoder:** `c_t = plain_rmsnorm(fc(concat(5 target residuals)), enc.output_norm)`, where `fc` is a Q4_K 25600 -> 5120 matrix and the norm is applied **after** `fc`.
- **Per decoder layer** (`il = 0..4`): `n = rmsnorm(inpL, attn_norm)`; a dynamic convolution (`dyn = attn_conv_proj @ n`, then `conv(n, dyn, attn_conv_base)`); `Q, K, V = wq/wk/wv @ n` — the same `K`/`V` weights are also used for the context injection; per-head RMSNorm on `Q` and `K`; RoPE on the first 64 dimensions only (`dimension_sections [64,0,0,0]`), and so on.

**Deliverable.** The module plus a written, itemised to-do list for M28c — which is what made the three bugs found there findable.

---

# M28b:草稿模块(加载 + 算子 + 几何),以及 M28c 的精确待办

日期:2026-10-25 · M28 的第二步。用户要求:先插 M27a 抬上限(已完成,`reports/M27a-small-grid-fixes.md`),然后直接上 **Q4_K_M**(已完成解码与前向算子,见 §2)。

## 1. 从 Prism 的源码拿到的权威图(不再靠散文推断)

`refs/llama.cpp-prism/src/models/dflash.cpp` 是这段实现的唯一权威(草稿就是一个普通 5 层模型,没有专用 Metal 内核):

**上下文(encoder)**:`c_t = plain_rmsnorm(fc(concat(5 个目标残差)), enc.output_norm)`;`fc` 是 25600→5120 的 Q4_K;`enc.output_norm` 是 **fc 之后**的 RMSNorm。

**每层(decoder,il = 0..4)**:
```
n = rmsnorm(inpL, attn_norm)
dyn = attn_conv_proj @ n ;  n = conv(n, dyn, attn_conv_base, side=0)
Q, K, V = wq/wk/wv @ n            # K/V 也用于上下文的注入(同一份权重)
Q = per_head_rmsnorm(Q, q_norm);  K = per_head_rmsnorm(K, k_norm)
Q = rope(Q, pos); K = rope(K, pos)          # 只旋转前 64 维,dimension_sections [64,0,0,0]
cur = attn(Q, K, V, non-causal, sliding 2048) @ wo
cur = conv(cur, dyn, attn_conv_base, side=1)          # 输出侧复用输入侧算出的 dyn
ffn_inp = cur + inpL
n2 = rmsnorm(ffn_inp, ffn_norm)
dyn2 = ffn_conv_proj @ n2 ; n2 = conv(n2, dyn2, ffn_conv_base, side=0)
cur = ffn_down(silu(ffn_gate(n2)) * ffn_up(n2))
cur = conv(cur, dyn2, ffn_conv_base, side=1)
inpL = cur + ffn_inp
out = rmsnorm(inpL, output_norm); logits = target_lm_head(out)
```
**conv 的精确语义**(`build_dflash2_conv`):`out[t][c] = Σ_tap (base[side][tap][c] + delta[t][g(c)][tap][side]) * X[t−tap][c]`,`X[−1]=0`;`delta` = conv_proj 的输出,每 token 1280 个值,顺序 `[group(320)][tap(2)][side(2)]`;`base` 的布局是 `[side][tap][c]`(c 最快)。与 ninfer 文档一致。

**输入输出**:draft 用目标模型的 `tok_embd`(**需要反 Hadamard 后的普通 embedding**,即我们 `k_embed_b` 的输出)与目标的 `lm_head`(PTQ1_0 + Hadamard,所以要做 sign+FWHT 再走宽 GEMV)。

## 2. 本轮完成

| 项 | 状态 |
|---|---|
| Q4_K / Q6_K 解码 | ✅ 与 ggml **逐位一致**(258,048 元素,差 0.0) |
| Q6_K 向量化 | ✅ 17 → **73.5 GB/s**(草稿一轮 21 → **14 ms**) |
| 草稿矩阵乘 `k_draft_mm` | ✅ Q4_K 76–118 GB/s,行 0 与 CPU 反量化一致 |
| 动态卷积 `k_draft_conv` | ✅ 按上面的精确语义实现(含 side 0/1、delta 布局、zero-pad) |
| draft 的 QKV prep `k_draft_qk` | ✅ 逐头 RMSNorm(32/8 头、128 维)+ 部分 RoPE(前 64、配对 (i,i+32))+ ring 写入 |
| 非因果滑窗注意力 `k_draft_attn` | ✅ 一 TG 一 (头,行),窗口 2048,在线 softmax,ring 4096 |
| 上下文 K/V `k_draft_ctx_kv` | ✅ k_norm + rope 后写入 ring |
| 加载器 `src/draft.{h,cpp}` | ✅ 81 个张量全部绑定、环形缓冲(5×4096×1024×2×2 B = 80 MB)、scratch;`build/draft_probe` 打印几何 |
| **前向装配 + 选择器 + spec 集成** | ⬜ M28c |

## 3. M28c 的待办(按顺序,每步都可验证)

1. **`Draft::materialise`**:`catH`(fp16 [n][25600])→ `k_draft_mm`(fc,Q4_K,MT=8)→ `k_rmsnorm_rows_h`(encNorm)→ `cH`;每层 `k_draft_mm`(wk/wv,Q4_K,5120→1024)→ `k_draft_ctx_kv`。验收:与 llama.cpp 的 `enc_norm_out` 不可能直接比,但可以用"整条 draft 的接受率"做端到端验收。
2. **`Draft::forwardBlock`**:按 §1 的图装配(4 个 `k_draft_mm` + `k_rmsnorm_rows_h` + `k_draft_qk` + `k_draft_attn` + `k_draft_conv` ×2 ×2 + `k_swiglu` + `k_add2`),末尾 `k_add_rms_fwht_t`(outNorm,flags=0)+ `eng.xWideGemv(outW)` 得 8×248320 的 logits。
3. **提案**:先用**每行 argmax**(DFlash1 风格,跳过选择器)把整条链路跑通,量接受率;之后再加选择器(two codebooks + rank-256 游走,约 +0.3–0.7 token/轮)。
4. **spec 集成**:`specGenerate` 加一个 draft 回调;接受前缀用既有的贪心/采样规则;`commitWide` 之后调用 `materialise` 补齐新提交位置的上下文。
5. **验收**:同一提示上接受长度与 Prism 的 `llama-speculative-simple --spec-type draft-dflash` 差 ≤ 0.3;端到端代码/推理 ≥ 31 tok/s(现在 25)。

## 4. 现在的账(M27a + 实测草稿成本)

| 配置 | 一轮 | 代码/推理 | 散文 |
|---|---|---|---|
| 今天(普通解码 / MTP-1) | 50.7 / 62 ms | 20 / 25 tok/s | 25 |
| M27a 验证 + 草稿 14 ms,argmax 提案 | 156 ms | **33 tok/s**(按 5.2 token/轮) | 13 |
| 同上 + 选择器(若多 0.5 token/轮) | 156 ms | 36 | 13.5 |
| 散文需要 M29 宽度自适应回退到 MTP-1 | — | — | 25 |

## 5. 复现

```
make build/dq_probe build/bench_draft build/draft_probe
build/dq_probe       ../../models/drafters/dflash2-q4km.gguf    # 解码 vs ggml(逐位)
build/bench_draft    ../../models/drafters/dflash2-q4km.gguf    # 草稿矩阵乘正确性 + 带宽
build/draft_probe ../../models/bonsai-2-27b-mtp-ptq1_0.gguf ../../models/drafters/dflash2-q4km.gguf   # 绑定与几何
```
