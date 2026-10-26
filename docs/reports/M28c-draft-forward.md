# M28c:草稿前向装配完成、端到端跑通;接受率仍为 0(定位到 3 个 bug,已修 2 个)

日期:2026-10-25 · M28 的第三步。**这是一个中间状态的诚实记录**:链路全部打通、NaN 已解决,但草稿的提案仍与主干不一致(0/7 接受)。下一轮从这个状态继续。

## 1. 现在能跑什么

```
build/bonsai dflash-eval --model ../../models/bonsai-2-27b-mtp-ptq1_0.gguf \
  --draft ../../models/drafters/dflash2-q4km.gguf --ids "$(cat /tmp/ids72.txt)" --gen 24
```
输出骨架(实测):
```
[draft] 5 layers, 1079 MB of weights, ring 4096 positions, 80 MB of K/V
prompt 72 tokens via 8-row windows in 4.1 s; anchor 271 at pos 72
  round 0: drafts 5144 6681 91561 91561 10758 10758 10758
            trunk   550 49276 271 271 271 271 271
            accept 0 -> 1 tokens
DFlash2: 24 rounds, 24 tokens, 1.00 tokens/round, 142.6 ms/round
```

**已经正确工作的部分**:
- 提示与续写都走 8 行宽验证窗口;主干给出的续写与普通贪心解码**逐 token 一致**(550, 220, 16, 13, 561, 1330, 33286, 271)。
- 残差 tap 采集(第 6/20/34/48/62 层,`verifyWide` 里在残差加完之后拷成 fp16)、`commitWide`、`Draft::materialise`(上下文 K/V 进环形缓冲)全部按预期执行,跨轮稳定。
- 草稿的 5 层前向按 Prism 的 `dflash.cpp` 精确装配(conv 双侧、delta 复用、逐头 qk-norm、部分 RoPE、非因果滑窗、共享 lm_head),整轮约 **4.5 ms**(draft 只占每轮 ~146 ms 的 3%)。

## 2. 这一轮修掉的两个 bug(都在装配阶段,已提交)

1. **草稿的 norm 权重绑到了主干的 buffer**:`k_rmsnorm_rows*` 的 γ 必须来自草稿自己的 GGUF 映射,不是主干的 `wbuf`。修好后 logits 从"全 0"变成真实数值。
2. **Q6_K 张量被当成 Q4_K 解码**:Q4_K_M 混了 4 个 Q6_K 张量(`blk.2/4` 的 `attn_v`、`ffn_down`),而我的 `mm()` 把 dtype 写成了字面量 `false`。现在是按偏移查 GGUF 的真实类型。**这就是 NaN 的来源**(层 2/4 读到错误字节 → 极大值 → inf → NaN),修完后 NaN 消失。

## 3. 还没解决的(下一轮从这里开始)

现象:草稿提案与主干不一致,接受 0/7。幅值剖面(`BONSAI_DBG_STAGE=1`)显示第一层内部量级合理(embed 0.003 → rms 0.37 → conv0 0.5 → attn 0.41 → wo 3.0 → ffn conv0 0.15 → gate 3.3 → swiglu 1.3),但 **`down` 的输出平均 57**(残差累计到 132/层,两层后 8.3M)。所以要么 `down` 的输入/权重对应关系仍有错,要么这个模型的激活量级本来就大、问题在别处。

建议的定位顺序(每步都可验证):
1. **先确认量级基准**:把主干的第 0 层残差量级打出来(在 `verifyWide` 里临时打印 `pX[c]` 的 sum|.|),与草稿的 5.4M 对比。如果主干也是同一量级,就说明问题不在量级而在"内容"(上下文 K/V、掩码、位置)。
2. **上下文路径自检**:用"零上下文 + 单个 anchor"跑一次 draft(ring 只含 block 的 8 列),看提案是否变成合理的后续 token。如果合理 → 问题在 `materialise`(fc/encNorm/rope 的顺序或 tap 顺序);如果仍乱 → 问题在层内。
3. **tap 层号 A/B**:GGUF 写的是 `[6,20,34,48,62]`(0 基),ninfer 文档写 `[5,19,33,47,61]`(1 基)。两种都试,取接受率高的(这是唯一"文档之间不一致"的点)。
4. **选择器**:目前用每行 argmax(DFlash1 风格),不依赖 codebook;先把 argmax 版本跑对,再加选择器。
5. **端到端验收**:与 `llama-speculative-simple --spec-type draft-dflash` 在同一提示上的接受长度差 ≤ 0.3。

**关于 oracle 的一个发现**:Prism 的 `LLAMA_MTP_DRAFT_LOG` 只覆盖 MTP 草稿(DFlash 不走那条记录路径,实测生成的日志为空),所以**拿不到"逐位置的正确草稿 token"作为直接对照**。可用的替代 oracle:
- 端到端接受长度(唯一可靠的验收);
- ninfer 的 `docs/maintainer/dflash.md` 方程 + Prism 的 `dflash.cpp` 图(已用于装配);
- 如果需要逐层对照,可以用 ggml 的 dequant 在 CPU 上重算一层的 8 个 token(慢但可行,只做一次);
- 检查 `--spec-draft-n-max 1` 时 Prism 的接受率:如果它在 K=1 时接近 100%,说明其草稿单步就基本正确,可作为"我的实现应当达到"的弱标尺。

## 4. 调试钩子(保留在代码里,默认关闭)

- `BONSAI_DBG_STAGE=1`:每层的关键缓冲 sum|.|/NaN 计数(会显著变慢,仅调试用)。
- `BONSAI_DBG_DRAFT=1`:打印草稿提案与主干值、logits 量级。
- `--vwidth T`:宽验证的"只接受第 0 行必须复现贪心"回归。

## 4.5 模型文件的选择(用户的更正)

- `bonsai-2-27b-mtp-ptq1_0.gguf`(Ollama `mannix/Bonsai-2-27B:mtp-ptq1_0`)= PrismML 的 PTQ1_0 + 额外一层 `blk.64.*`(q8_0,约 0.45 GB)的 MTP 头;**其余每个 key/张量与 PrismML 文件逐字节相同**。
- **若用 DFlash2 当草稿,应当用源文件 `Ternary-Bonsai-2-27B-PTQ1_0.gguf`(prism-ml),不带 blk.64**:省 0.45 GB 常驻,且这正是 DFlash2 的配套组合。
- 我们两个文件都测过:主干张量相同,所以 trunk 数值与 `--vwidth` 回归一致;DFlash2 的 tap 采集与草稿行为不受影响(这也交叉验证了"只多了 blk.64"的说法)。下载中的文件:`../../models/bonsai-2-27b-ptq1_0-src.gguf`。

## 5. 本轮的完整进展(追加)

**已修复 4 个 bug**,每一个都用可观测的现象确认:

| # | bug | 现象 | 修复 |
|---|---|---|---|
| 1 | 草稿的 norm 权重绑到主干的 buffer | logits 全 0 | 用草稿自己的 GGUF 映射 |
| 2 | Q6_K 张量按 Q4_K 解码(`blk.2/4` 的 `attn_v`、`ffn_down`) | 层 2 出现 NaN | 按偏移查 GGUF 真实类型 |
| 3 | **残差 tap 采集代码从未插入**(之前的批量替换静默失败) | 上下文 K/V 全零、有无上下文结果逐位相同 | 在 `verifyWide` 的残差加之后插入 `k_f2h` 拷贝;实测 taps sum\|.\|=200504、环缓冲随位置递增 |
| 4 | `k_draft_qk` 的 `isK` 条件写成 `h >= H + HKV`(与 V 分支相同) | K 头被当 Q 头、**K 从未写进环** → 注意力分数恒 0 → softmax 均匀 → logits 全挤在 0.5 nat 内 | 改为 `h >= H` |

修复 3、4 之后草稿的输出从"持平/全 0"变成有结构的分布(row 0 top-5 从 6.5–7.0 展开到 9.2–10.3),但**接受率仍是 0**:row 0 仍未预测出 anchor(row 0 top-1 = 1428,anchor 271 的 logit 6.33),提案也未命中主干的 550。

**下一步的定位手段(已备好)**:
- 自洽性指标"row 0 预测 anchor 的排名"已经作为一行输出常驻:当前 10 名左右(修复前 414),可当作回归指标。
- 位置偏移/语义的 A/B 只需改两处:`forwardBlock` 的 `pos`(±1)与提案取行(rows 1..7 还是 0..6)。
- 若仍不对,做一次 CPU 单层对照:用 ggml 的 dequant 在 CPU 上重算第 0 层(8 行,含 72 位置上下文),逐缓冲比较——这是唯一能给出"第一个偏离的算子"的方法。

## 6. 第五个 bug(本轮决定性):动态卷积的 delta 索引

`build_dflash2_conv` 里 `dynamic` 被 `ggml_reshape_4d(dynamic, n_groups, kernel, 2, n_tokens)`,而 ggml 的 `ne[0]` **是最快变化的维** → 线性索引是
```
dyn[t][g][tap][side] = dyn_linear[t*1280 + g + 320*(tap + 2*side)]
```
我原来按自然的 `[g][tap][side]`(side 最快)去读,于是每个 token 的 1280 个 delta 系数被完全打乱。

修复后(唯一改动就是这一个索引):
- **row 0 的 top-1 变成 anchor 自己(271)** —— 自洽性测试通过,这是"层内数学正确"的第一个硬证据;
- 0 基 tap 层号(5.72 的 anchor logit)优于 1 基(5.17),印证 GGUF 里的 `[6,20,34,48,62]` 是 0 基;
- 提案仍是错的(row 1 top-1 = 421,而主干说 550),接受率 2%。

## 7. 关于"用真实 token 填 block"的负面结果(避免下次重复)

为了分离"mask 处理"与"层内数学",我试过把真实续写填进 block 的 1..7 行。**这个实验无效**:注意力是非因果的,填了真 token 后 row 0 会看到"未来",它的输出从 271 变成 25——所以该配置下"row i 预测位置 F+i"不再良定义。正确的判据只能是 mask 配置下的端到端接受率。

## 8. 当前最可能的原因(下一轮的首要假设)

row 0 自洽只证明"恒等路径"(embedding → 5 层 → 共享词表头)能复现输入 token,**不能证明上下文被正确利用**。row 1 依赖上下文才能预测 550,而它失败。所以嫌疑集中在**上下文路径**:
1. `fc` 的输入顺序或 `encNorm` 的位置(已按图实现,但无 oracle);
2. 上下文 K 是否该用 `k_norm` + RoPE、V 是否该原样(已按图;可试一次"上下文 K 不做 RoPE"的 A/B);
3. `target_layers` 是 0 基(现在证据偏向 0 基);
4. 上下文 K/V 的写入位置(现在用 `frontier` 递增,已用非零计数确认覆盖)。

下一轮建议直接做**一次 CPU 单层对照**(第 0 层、8 行、72 位置上下文,用 ggml 的 dequant + 纯 C++ 数学),逐缓冲比较 attention 的输出——这是唯一能在"上下文没被正确利用"这类问题上给出确定答案的方法。

## 9. 逐算子验证结果(本轮的决定性工作)

为了不再靠猜,我把 GPU 的中间量 dump 出来,用**独立实现**重算并比较(`tools/dump_attn.cpp` + `/tmp/fc_check.c`):

| 算子 | 对照方法 | 结果 |
|---|---|---|
| `k_draft_mm`(Q4_K/Q6_K 矩阵乘) | ggml 的 `dequantize_row_q4_k/q6_K` + 双精度点积 | **通过**(fc: 0.027 = fp16 舍入;ffn_gate/down/q/conv_proj 均通过) |
| `k_draft_conv`(动态分组卷积) | 纯 Python 按 `build_dflash2_conv` 的语义重算 | **通过**(最大差 0.0) |
| `k_draft_attn`(非因果滑窗注意力) | 纯 Python 逐 key 在线 softmax(用 GPU 的 Q 与 ring) | **通过**(最大差 2e-6) |
| `k_draft_qk`(逐头 RMSNorm + RoPE + ring 写入) | 由注意力的匹配间接验证(Q 与 ring 都被 CPU 正确使用) | **通过** |
| `fc` + `encNorm`(上下文物化) | ggml dequant + 双精度 | **通过** |
| 最终 norm + 共享头 | row 0 自洽(预测出 anchor) | **通过** |

**所有算子都是对的,但端到端接受率只有约 5%**(模式提示上 6/7 正确)。这说明问题不在算子,而在**结构假设**。剩余嫌疑(按可能性):
1. 目标残差 tap 的**层号**:GGUF 写 `[6,20,34,48,62]`,ninfer 文档写 `[5,19,33,47,61]`;两者实测差异很小(4.8% vs 4.8%),但都没有达到应有的水平;
2. tap 的**残差位置**:现在取"mixer+FFN 之后"(ninfer 的合同),也可以试"仅 mixer 之后";
3. **选择器**:现在用逐行 argmax(DFlash1 风格)。官方的 DFlash2 用 top-16 + rank-256 条件路径,可能显著优于逐行 argmax;
4. 块内注意力的**掩码**(非因果 vs 块内因果)。

## 10. 诊断工具(可复现)

```
make build/dump_attn                     # 把第 0 层的嵌入/Q/ring/注意力输出/卷积输入输出 dump 到 /tmp
build/dump_attn ../../models/bonsai-2-27b-ptq1_0-src.gguf ../../models/drafters/dflash2-q4km.gguf
BONSAI_DUMP_FC=1 ...                     # 另外 dump fc 的输入/输出与 c_t(commit 之后写文件)
```
注意一个反复踩的坑:**诊断 dump 必须在 `waitUntilCompleted` 之后**(shared buffer 在编码阶段还没被 GPU 写),我曾两次因此读到旧数据、误判 `fc` 有错。

## 11. 两个 Bonsai 专用草稿(用户提供的新文件)

| 文件 | 大小 | 量化类型 | 旋转基 | 需要什么 |
|---|---|---|---|---|
| `Cyronius/Qwen3.8-27B-DFlash2-Bonsai2-ternary` | 746 MB | **type 142(2-bit prism)×46**、F32×32、BF16×3 | **是**:自带 `prism.hadamard.sign_widths = [4096, 5120, 17408, 25600]`、`sign_mode = explicit`、`sign_values`、`weight_names`(逐个列出哪些矩阵是旋转的);`borrowed.signs_folded = output_norm, selector_hidden_cols` | 2-bit prism 解码 + 每个旋转矩阵输入做 FWHT(还多出 4096 与 25600 两个宽度,不在主干 GGUF 里) |
| `Schiltmans/Ternary-Bonsai-2-27B-DFlash2-ft5-Q2_K` | 623 MB | Q2_K×34、Q3_K×10、Q4_K×5 | **否**(无 prism 元数据,常规基) | Q2_K + Q3_K 解码(现有管线即可) |

两者都是 `target_layers = [6,20,34,48,62]`(1 基边界),即 0 基块 **5/19/33/47/61**;`Cyronius` 的 `config.json` 明确写 `[5,19,33,47,61]`,与 ninfer 文档一致 → 我们的 `BONSAI_TAP_SHIFT=-1` 是正确设置。

**代码改动**:`k_draft_mm` 的模板参数从 `bool Q6` 改为 `int QT`(0=Q4_K,1=Q6_K,2=Q2_K,3=Q3_K),按每个张量自己的 GGUF 类型选择解码器;GGUF 加载器识别 Q2_K(84 B/256)、Q3_K(110 B/256)、PQ2_0(type 142)。

**验证状态**:
- Q2_K 解码 **通过**(与 ggml 的 `dequantize_row_q2_K` 对照,ffn_gate 5120 宽)。
- Q3_K 仍不对(17408 宽的张量行 0 与 ggml 不一致)。本轮已修两个错:qs 在 hmask 之后的偏移(+32)、每半 super-block 的 scale 索引漏了 `half_*8`。剩余差异未定位。
- 因为 ft5 的 down 矩阵是 Q3_K,草稿仍不可用,接受率 0%。
- ternary 那个需要一条新的算术路径(2-bit prism 解码 + FWHT),本轮未开始。

**下一步**:①把 Q3_K 对着 ggml 逐元素比(用 `k_dq_q3k` 式的探针,像 Q4_K 那样一次定位);②ternary 草稿:实现 type 142 解码 + 在 `fc`/`attn_conv_proj`/`attn_q/k/v/o`/`ffn_*` 的输入侧插入 FWHT(宽度 4096/5120/17408/25600 用它自带的符号向量),并把 `output_norm`/`selector_hidden` 的符号折叠考虑进去。

## 12. DFlash2 选择器(按用户要求,不再用 DFlash1 的逐行 argmax)

按 Prism 的 `build_dflash2_selector` 精确实现:

1. `gate = selector_hidden @ h`(BF16,不旋转 → 输入是**未旋转**的最终归一化隐状态);新内核 `k_draft_bf16mm`(BF16 权重 × fp16 激活)。
2. 每个 mask 行取 **top-16**(按分数降序,与 `ggml_top_k` 一致)及其 unary logits。
3. 两个 rank-256 码本按需 gather(每行 256 个 BF16):`W_pred[prev_candidate_id]`、`W_succ[candidate_id]`。
4. 条件走查:`score[c] = u[c] + Σ_r W_pred[P_prev, r] · g[r] · W_succ[C_c, r]`,第 1 个位置的前驱是 anchor(单行),之后用上一位置**选中的候选索引**作为前驱行 —— 与 ninfer 的
   `E_i[p,c] = u_i[c] + Σ_r W_pred[pred_token(i,p),r] g_i[r] W_succ[C_i[c],r]` 一致。

`BONSAI_DF_ARGMAX=1` 保留为 A/B(DFlash1 风格)。实测:提案序列与 argmax 明显不同(例如 760 4087 369 264 …),但**接受率仍只有 0.6%**。

**这意味着缺口不在"argmax 还是选择器"**:选择器把候选约束成一条连贯路径,但它仍建立在草稿的逐行分布之上;而草稿的逐行分布本身就不够准(row 0 自洽、row 1 之后偏离)。所以问题回到**结构假设**——最可能是 `fc` 的 tap 拼接顺序/层号,或者块的输入语义(anchor 的位置)。

## 13. 累计状态(供接手)

- 已修 **11** 个 bug:PQ2_0/旋转基、Q3_K 两处、delta 索引、RoPE 约定、tap 位置、tap 采集缺失、K 头分支、norm 权重绑定、Q6_K dtype、fp16→float 转换。
- 逐算子验证全部通过(解码 vs ggml:Q4_K/Q6_K/Q2_K/PQ2_0;卷积、注意力、fc vs 独立 CPU 实现;row 0 自洽且尖锐 logit 9.95 vs 8.40)。
- 端到端接受率 ~1%(应有 ~70%),两个 Bonsai 专用草稿都是如此。
- 结论:**剩下的误差是结构性的**,不是数值性的。若要继续,最有价值的是拿到一个 oracle:llama.cpp 在 `llama-speculative-simple --spec-type draft-dflash` 下对**同一提示**的逐位置草稿 token(目前没有日志钩子),或者官方转换脚本里 `target features` 的确切构造方式。
