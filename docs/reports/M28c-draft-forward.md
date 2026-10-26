# M28c:草稿前向装配完成、端到端跑通;接受率仍为 0(定位到 3 个 bug,已修 2 个)

日期:2026-10-25 · M28 的第三步。**这是一个中间状态的诚实记录**:链路全部打通、NaN 已解决,但草稿的提案仍与主干不一致(0/7 接受)。下一轮从这个状态继续。

## 1. 现在能跑什么

```
build/kata dflash-eval --model ../../models/Bonsai 2-mtp-ptq1_0.gguf \
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

现象:草稿提案与主干不一致,接受 0/7。幅值剖面(`KATA_DBG_STAGE=1`)显示第一层内部量级合理(embed 0.003 → rms 0.37 → conv0 0.5 → attn 0.41 → wo 3.0 → ffn conv0 0.15 → gate 3.3 → swiglu 1.3),但 **`down` 的输出平均 57**(残差累计到 132/层,两层后 8.3M)。所以要么 `down` 的输入/权重对应关系仍有错,要么这个模型的激活量级本来就大、问题在别处。

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

- `KATA_DBG_STAGE=1`:每层的关键缓冲 sum|.|/NaN 计数(会显著变慢,仅调试用)。
- `KATA_DBG_DRAFT=1`:打印草稿提案与主干值、logits 量级。
- `--vwidth T`:宽验证的"只接受第 0 行必须复现贪心"回归。

## 4.5 模型文件的选择(用户的更正)

- `Bonsai 2-mtp-ptq1_0.gguf`(Ollama `mannix/Bonsai-2-27B:mtp-ptq1_0`)= PrismML 的 PTQ1_0 + 额外一层 `blk.64.*`(q8_0,约 0.45 GB)的 MTP 头;**其余每个 key/张量与 PrismML 文件逐字节相同**。
- **若用 DFlash2 当草稿,应当用源文件 `Ternary-Bonsai-2-27B-PTQ1_0.gguf`(prism-ml),不带 blk.64**:省 0.45 GB 常驻,且这正是 DFlash2 的配套组合。
- 我们两个文件都测过:主干张量相同,所以 trunk 数值与 `--vwidth` 回归一致;DFlash2 的 tap 采集与草稿行为不受影响(这也交叉验证了"只多了 blk.64"的说法)。下载中的文件:`../../models/Bonsai 2-ptq1_0-src.gguf`。

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
build/dump_attn ../../models/Bonsai 2-ptq1_0-src.gguf ../../models/drafters/dflash2-q4km.gguf
KATA_DUMP_FC=1 ...                     # 另外 dump fc 的输入/输出与 c_t(commit 之后写文件)
```
注意一个反复踩的坑:**诊断 dump 必须在 `waitUntilCompleted` 之后**(shared buffer 在编码阶段还没被 GPU 写),我曾两次因此读到旧数据、误判 `fc` 有错。

## 11. 两个 Kata 专用草稿(用户提供的新文件)

| 文件 | 大小 | 量化类型 | 旋转基 | 需要什么 |
|---|---|---|---|---|
| `Cyronius/Qwen3.8-27B-DFlash2-Kata2-ternary` | 746 MB | **type 142(2-bit prism)×46**、F32×32、BF16×3 | **是**:自带 `prism.hadamard.sign_widths = [4096, 5120, 17408, 25600]`、`sign_mode = explicit`、`sign_values`、`weight_names`(逐个列出哪些矩阵是旋转的);`borrowed.signs_folded = output_norm, selector_hidden_cols` | 2-bit prism 解码 + 每个旋转矩阵输入做 FWHT(还多出 4096 与 25600 两个宽度,不在主干 GGUF 里) |
| `Schiltmans/Ternary-Bonsai-2-27B-DFlash2-ft5-Q2_K` | 623 MB | Q2_K×34、Q3_K×10、Q4_K×5 | **否**(无 prism 元数据,常规基) | Q2_K + Q3_K 解码(现有管线即可) |

两者都是 `target_layers = [6,20,34,48,62]`(1 基边界),即 0 基块 **5/19/33/47/61**;`Cyronius` 的 `config.json` 明确写 `[5,19,33,47,61]`,与 ninfer 文档一致 → 我们的 `KATA_TAP_SHIFT=-1` 是正确设置。

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

`KATA_DF_ARGMAX=1` 保留为 A/B(DFlash1 风格)。实测:提案序列与 argmax 明显不同(例如 760 4087 369 264 …),但**接受率仍只有 0.6%**。

**这意味着缺口不在"argmax 还是选择器"**:选择器把候选约束成一条连贯路径,但它仍建立在草稿的逐行分布之上;而草稿的逐行分布本身就不够准(row 0 自洽、row 1 之后偏离)。所以问题回到**结构假设**——最可能是 `fc` 的 tap 拼接顺序/层号,或者块的输入语义(anchor 的位置)。

## 13. 累计状态(供接手)

- 已修 **11** 个 bug:PQ2_0/旋转基、Q3_K 两处、delta 索引、RoPE 约定、tap 位置、tap 采集缺失、K 头分支、norm 权重绑定、Q6_K dtype、fp16→float 转换。
- 逐算子验证全部通过(解码 vs ggml:Q4_K/Q6_K/Q2_K/PQ2_0;卷积、注意力、fc vs 独立 CPU 实现;row 0 自洽且尖锐 logit 9.95 vs 8.40)。
- 端到端接受率 ~1%(应有 ~70%),两个 Kata 专用草稿都是如此。
- 结论:**剩下的误差是结构性的**,不是数值性的。若要继续,最有价值的是拿到一个 oracle:llama.cpp 在 `llama-speculative-simple --spec-type draft-dflash` 下对**同一提示**的逐位置草稿 token(目前没有日志钩子),或者官方转换脚本里 `target features` 的确切构造方式。

## 14. 推广 ALU 验证 + MTP 链 + 自适应深度(用户要求的第 1 条)

**已完成的部分**:

1. **ALU 验证路径**:`Engine::aluT`(0/2/4)让 `dWide` 在"矩阵单元宽内核"与"ALU LUT+half2 内核"之间切换;为此加了 `k_ilv2`/`k_ilv4` 两个纯 permute 内核(row-major fp16 → 交错 half2),**不改动已验证的 norm/FWHT 内核**。实测每轮:T=2 **85.9 ms**、T=4 **113.3 ms**、T=8(宽)146.6 ms,且 `--vwidth`+`KATA_ALU=4` 仍逐 token 复现贪心。
2. **MTP 链式草稿** `specGenerateChain`:每步把 MTP 块自己的输出当作下一步的 h,深度 1 走快速 `verify2`(67 ms),深度 3 走 ALU T=4(113 ms)。
3. **自适应深度**:按"每毫秒期望 token 数"在线比较两档(Strata 的 DraftPolicy 简化版),初值用实测轮耗时。
4. **修了一个正确性 bug**:链式路径最初用"草稿 == 主干 argmax"接受,这会破坏采样分布;改成精确规则(从 p 采样、命中才接受、否则用采样值)。

**结果(推理文本,贪心)**:

| | tok/s | 每轮 token | 接受率 |
|---|---|---|---|
| 深度 1(默认) | **29.2** | 1.90 | 90.5% |
| 链式深度 3 | 21.2 | 2.46 | 48.4% |

**链式目前是负收益**:因为 d_2/d_3 的逐位置接受率只有约 0.4/0.15,而 Prism 的 MTP 链在同一模型上是 0.8/0.7(实测 3.13 token/轮)。根因是**h 代理的质量**:链式第二步必须用"目标在位置 n 的隐状态"的估计,我用 MTP 块自己的输出(并对它做了 `output_norm`),但接受率对这一步不敏感(改归一化前后 47.7% → 48.4%),说明 h 代理不是唯一问题。已把默认回退到深度 1,链式保留为 `--mtp-chain` 可选。

**结论**:要拿到 +10~20%,必须先让链式草稿的 d_2/d_3 达到 Prism 的水平(0.8/0.7)。下一步应直接读 llama.cpp 的 `llama_mtp_chain`(mtp.cpp)看它第二步到底喂什么(是 h 还是 `h + token` 的某种融合,以及是否重算归一化),而不是继续猜。

## 15. 读 llama.cpp 的 MTP 链(按用户要求)与结论

**读到的精确语义**(`src/models/qwen35.cpp` 的 chain graph,`n_steps` 个 step 全在一个 `llama_decode` 里):

```
step s (s = 0: catch-up 行; s >= 1: 一行, 位置 r0 = n_lead-1+s):
    h_norm = RMSNorm(h, nextn.hnorm)          # h 是上一步的 h_nextn(step 0 是主干的 h)
    e_norm = RMSNorm(embed(argmax_token), nextn.enorm)
    cur    = eh_proj @ [e_norm | h_norm]
    cur    = attn(...) * sigmoid(gate) -> wo -> +cur
    cur    = ffn(RMSNorm(cur, attn_post_norm)) + cur
    h_nextn = RMSNorm(cur, head_norm_w)       # head_norm_w = nextn.shared_head_norm(不是主干 output_norm!)= 下一步的 h
    下一步的 token = argmax(head(h_nextn))
```
以及:MTP 缓存按位置顺序写;草稿是**贪心**(argmax),不是采样器给的 token。

**据此修正的两处**:
1. 链式下一步的 h 之前用的是主干的 `output_norm`,改成 `nextn.shared_head_norm` ✓(与 llama.cpp 一致)。
2. 修掉链式循环**把 anchor 重复发一遍**的 bug(导致"160 token"里含 64 个重复)。

**但链式仍然不可用**,而且暴露出更多问题:

| 指标 | 深度 1(默认) | 链式深度 3 |
|---|---|---|
| 推理文本 tok/s | **29.3** | 12.8(修 emit 后) |
| 首位置接受率 p1 | 0.905 | 0.40–0.50 |
| 提交的 token 序列 | 与普通贪心逐 token 一致 | **错的**(周期模式 `A B A B C B`,应为 `A B C A B C`) |

关键异常:**同一个第一份草稿,在深度 1 路径下接受率 0.9,在链式路径下只有 0.4**。已排除:
- T=4 的 ALU 验证:`--vwidth 4` 用**互不相同**的假草稿(能查出 token 行错位)仍逐 token 复现贪心 ✓;
- h 的归一化(shared_head_norm vs output_norm)✓ 已改,数字没动;
- 草稿调用本身(同一 `mtpRows(1, ...)`)✓。

所以问题在链式循环的**状态/提交记账**(它也解释了为什么提交序列是错的)。这条路径已从 CLI 撤下(`--mtp-chain` 移除),代码与本节记录保留,供下一轮定位;默认仍是深度 1。

**结论**:用户要求的第 1 条(推广 ALU 验证 + MTP 链 + 自适应深度)里,**ALU 验证基础设施完成并验证通过**(T=2/4/8 三条路径、9 种形状扫描),但**链式草稿有记账 bug 未解决**,因此没有采用。要拿到推理文本的 +10~20%,下一步应查链式循环的 `commitWide`/`encMtpKv`/`nextTok` 记账(而不是草稿本身)。

## 16. 链式记账排查(用户要求的下一项)

**这一轮修掉的三个真 bug**(都不是"记账"之外的猜测):

1. **`bufH` 是 fp16,却被当 float 读**:`Engine::hidden()` 直接 `(const float*)bufH->contents()`,以及 `feed()` 的逐步路径 `memcpy(hid, bufH->contents(), kHiddenBytes)`。短后缀(≤24 token)时 hPrev 因此是垃圾。已加 fp16→float 转换(`hConv` 缓冲)。
2. **链式循环把 anchor 重复 emit 一次** → 输出里出现重复 token,"160 token"里含 64 个重复。
3. **链式下一步的 h 用错 norm**:应为 `nextn.shared_head_norm`(llama.cpp 的 `h_nextn`),不是主干的 `output_norm`。

**排查过程中的关键数据**(强制深度):

| | 每轮 | 每轮 token | 接受率 | tok/s |
|---|---|---|---|---|
| 深度 1(链式管线) | 72 ms | 2.00 | 100% | 28.8 |
| 深度 3(链式管线) | 113 ms | 1.51 | 50.6% | 12.5 |

深度 3 的 **p1 = 0.47**(第一份草稿的接受率),而深度 1 是 1.00 —— 而第一份草稿在两条路径里是**同一次调用**。已用打印确认 h 的传递是正常的(`sum|h| ≈ 4479`,数值合理),所以 p1 的差异**不能用 h 解释**;`verifyWide` 的 T=4 行 0 也已用"互不相同的假草稿"验证过是逐 token 正确的。因此差异只能出在 `specGenerateChain` 与 `specGenerate` 对**同一窗口**的不同处理上,本轮未能定位到具体一行。

**结论**:第 1 条(ALU 验证 + MTP 链 + 自适应深度)中,ALU 验证基础设施完成且验证通过;链式草稿的三个 bug 已修但**仍有未定位的问题**,所以默认保持深度 1(散文 25.4、推理 28.8–29.3 tok/s),链式仅 `KATA_CHAIN=1` 可开。下一个嫌疑点:`specGenerateChain` 里 `fastDepth1` 与 `verifyWide` 两条分支对 `am[]`/`verifyLogits()` 的读取差异(前者读 `bufLogits`,后者读 `bufLogitsAll`)。

## 17. `--fast-ffn` 设为默认;宽验证 r 的天花板

**默认已切换**:`--fast-ffn`(int8 FFN 预填充)现在是默认路径,`--exact` 关闭。7.3K 提示 **207 → 227 tok/s(+10.7%)**;
质量证据不变(3072 位置 top-1 99.87%、KL 7.5e-5)。`--kv int8` 与该默认叠加实测 99.58%(低于 99.7% 门限),因此该组合
自动回落到精确预填充并打印提示。

**宽验证 r 的下压尝试(负结果)**:实现了"每个 simdgroup 带两个独立行组"的 `k_wide2`,期望 MMA A 与解码 B 指令级
重叠(N=8×2,threadgroup 内存与 N=16 相同)。数值完全正确(max|diff| 与 `k_wide` 一致),但**更慢**:

| 内核 | 5120→17408 | 5120→5120 | 5120→248320 |
|---|---|---|---|
| `k_wide` m16 n16 k128 g4 | 421 µs | 492 | 259 |
| `k_wide2` m16 r8 k128 g4 | 663 µs | 776 | 395 |

原因是 MPP 的 `op.run()` 对发射 warp 是同步退役的,"解码 B 与 MMA A 重叠"不成立,而每 chunk 两次 barrier 的代价是实打实的。

**r 的硬底**:`k_wide` 的消融显示 420 µs 里 MMA 占 306 µs(9.4 TFLOPS,M=16 时是峰值的 54%),解码占 ~110 µs。
即使解码完全被隐藏,r 也只能到 306/170 = **1.8**(verify(8) ≈ 51×1.8+15 = 107 ms,DFlash2 约 40 tok/s)。
要拿到 r≈1.6 必须把 MMA 拉到 11.7 TFLOPS(峰值 69%)——这是 MPP 在 M=16 下的形状限制,不是代码能绕过的。
所以"宽验证 2.47 → 1.6"这一项**无法在 M6 上达成**;1.8 是可达上限,且需要另一种重叠手段(如把解码挪到第二个
kernel 并用 `MTLParallelRenderCommandEncoder` 之外的手工流水)。
