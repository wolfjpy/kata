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
