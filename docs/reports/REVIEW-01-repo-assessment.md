> 本文以中文写成;**英文全译:[REVIEW-01-repo-assessment.en.md](REVIEW-01-repo-assessment.en.md)**,下方 `## Summary (EN)` 为摘要,双语文档索引见 [docs/README.md](../README.md)。
> *This document is in Chinese. An English summary follows; the bilingual index is at [docs/README.md](../README.md).*

## Summary (EN)

**What this is.** A read-through assessment of the whole repository: where it stands, what is left open, and what to do
next. Discussion only — no code was changed in this round.

**How it was produced.** All 21 documents, the 11 source files in `src/`, the 14 MSL files in `shaders/` and the 8 programs
in `tools/` were re-read, and every number that would be used to make a decision was cross-checked against the code and,
where possible, re-measured on today's machine.

**What changed the conclusions** (§3.4): three new measurements — combined switches fail the quality gate, a one-shot
process necessarily pays 3.3 s of setup, and the decode path's arithmetic does not line up with the other paths. Those,
plus the existing performance account, produce the plan in §5 and the milestone list in §6.

**Contents.** The repository inventory (~3,750 lines of own code plus vendored metal-cpp, file by file with line counts and
roles), the open issues graded by cost and payoff, and the resulting plan. This is the document to read if you want to know
what is *not* finished rather than what is.

---

# REVIEW-01:仓库通读评估、遗留问题与下一步方案

日期:2026-10-25 · 只讨论,不改代码。配套阅读:`CHECKPOINT-003.md`(设计/决策/里程碑结果)、`reports/M0–M17`、`MODEL_SPEC.md`。

## 0. 这份文档怎么来的

我把仓库里的 21 份文档、`src/` 的 11 个源文件、`shaders/` 的 14 个 MSL 文件、`tools/` 的 8 个程序全部重读了一遍,并把报告里的关键数字与代码逐条对照;对**任何我会用来做判断的数字**,要么在今天的机器上重新测过(交错 A/B),要么标注了它的原始来源和条件。文中"今天"指 2026-10-25 这次复核。

新测的三件事改变了我的判断(§3.4):组合开关的质量不达标、一次性进程必然付 3.3 s、解码与其他路径的算术不一致。这三条加上已有的性能账,构成了 §5 的方案和 §6 的里程碑。

## 1. 仓库全景

### 1.1 代码(约 3 750 行自有代码 + 自带的 metal-cpp)

| 文件 | 行数 | 角色 |
|---|---|---|
| `src/engine.cpp` | 517 | **四条前向路径**:`step`(单 token)、`prefill`(批量)、`verify2`(2 token 验证)、`mtpRows`(MTP 头);外加 `feed`/`encMtpKv`/`warmup`/`dKvQuant` |
| `src/engine.h` | 117 | 缓冲区句柄、每层权重偏移、公开 API、前缀缓存用的裸指针访问器 |
| `src/cli.cpp` | 257 | `run`/`chat`/`tokenize`/`cache`/`mtp-eval`/`verify-check`/`kl-eval` |
| `src/ckpt.cpp` | 166 | 前缀缓存的格式、恢复、异步保存、LRU |
| `src/main.cpp` | 107 | 开发入口(逐层对照、`--gen`、`--verify`、`--skip` 消融) |
| `src/spec.cpp` | 47 | MTP 投机解码主循环(精确投机采样) |
| `src/tokenizer.cpp` + `pretok.mm` | 128 | BPE + ICU 正则(唯一的 Objective-C++ 文件) |
| `src/chat.cpp` | 69 | 对话模板 + 采样器 |
| `src/gguf.cpp` / `gpu.cpp` | 124 | GGUF 解析(整文件 mmap 零拷贝)/ Metal 上下文 |
| `shaders/*.metal` | 1 484 | 其中生产用 ~1 050 行;`gemv_exp`(299)、`layout_probe`(59)、`wide`(137)是实验/探针 |
| `tools/*` | 336 | 参考运行时封装、6 个微基准、缓存回归脚本 |

代码风格一致:MSL 与主机共享 `config/kata_m6.h` 的 constexpr;加载时用 GGUF 元数据校验;所有缓冲区在 `load()` 里一次性分配;没有动态分配;`make` 6 秒全量构建。

### 1.2 文档(21 份,分三层)

- **权威层**:`CHECKPOINT-003.md`(最新状态、决策、被推翻的假设、路线图结果)、`MODEL_SPEC.md`(数学)、`reports/M9–M17`(本轮)。
- **历史层**:`CHECKPOINT-001.md`(1209 行设计阶段记录,含全部硬件探测与上游调研)、`CHECKPOINT-002.md`、`reports/M0–M7`。
- **证据层**:`docs/probes/`(28 个探针源文件)、`docs/data/`(GGUF 头导出)、`docs/reference/`(Prism/HF 参考代码)。

历史层有**两处会误导人的地方**(不影响正确性,但影响信任):
1. `reports/M3` 写着"GEMV 不受 ALU 限制,不需要重排布局或 LUT"。这对 T=1 成立,但 M4/M12 证明 T≥2 是 ALU 受限、且重排也确实没用(理由不同)。结论对、论据后来被替换了。
2. `reports/M5` 的 278 tok/s int8、`CHECKPOINT-002` §7 的预填充退化表(3639→201、7279→176)都是 M5b 之前的状态;现在的数字是 255–263 / 226(`--fast-ffn`)。读历史报告的人会拿到旧结论。

建议:给 M3、M5、CHECKPOINT-002 顶部加一行"已被 XXX 取代"的横幅。这是一行修改成本、显著的阅读体验收益。

## 2. 现状评价

### 2.1 做对的地方(有数据支撑)

| 判断 | 证据 |
|---|---|
| 零拷贝 + 单一常驻权重副本是对的 | 5.657 GB/token,与理论下界一致;没有第二份权重 |
| PTQ1_0 的解码方式选对了 | T=1 GEMV 126–134 GB/s(持续带宽的 89–94%),且和"只读不解码"的原生布局上限(124–132)几乎相同 → 没有浪费 |
| decode 的 roofline 已经吃掉 ~80% | 50.6 ms × 142 GB/s = 7.19 GB 的"等效读取",实际 5.657 GB + 状态 0.3 GB ≈ 5.96 GB → 83% |
| 预填充的 GEMM 已到矩阵单元峰值 | 1024 chunk 实测 16.6 TFLOPS(峰值 17.3,96%);非 GEMM 部分只有 1.0 s |
| 精确路径的质量没有问题 | 72 位置 KL 1.4e-7、top-1 72/72 |
| 投机解码是精确的 | 确定性草稿 + `p(argmax q)` 接受,分布不变;`verify-check` 830 位置 top-1 100% |
| int8 结论有边界 | 全 int8 不达标 → 只做 FFN(99.87%)→ 29K 上下文 +14% |
| 每个结论都写了证伪方法 | 探针、消融开关、`kl-eval`、`verify-check` 都能重跑 |

### 2.2 技术债(代码层,按影响排序)

1. **四条前向路径各自实现了一遍层循环**。`step` 与 `verify2` 是同一结构的两次拷贝(约 60 行 × 2),`prefill` 是第三个变体,`mtpRows` 是第四个。任何改动(比如 fp16 状态、ReplaySSM、新的注意力变体)都要改 3–4 处,改漏一处就是静默的数值不一致 —— 这正是本轮已经发生过一次的事故类型(`main.cpp` 只设了 `int8Mode` 没设掩码,导致三种模式实际都跑精确路径,测速和结论一度都是错的)。
2. **模式选择散落在热路径里**:`getenv()`(`KATA_SCALAR_DEC`、`KATA_I8_MODE`、`KATA_OLD_EXPAND`、`KATA_DEC_CHUNK`、`KATA_GM`、`KATA_KV`)在每次编码时读一次(有 static 缓存,开销可忽略),但这些开关**不在 `numericsTag()` 里**,只有 int8 集合和 `+kv8` 进去了。换句话说:环境和代码之间有若干条能改变数值、但不会让缓存失效的路径。`KATA_SCALAR_*` 目前被 `!kvInt8` 挡住,但这是巧合,不是设计。
3. **`sk()` 用字符串比较做消融**:每 token 约 600 次调用,每次线性扫描 `vector<string>`。开销可忽略(~6 µs),但把一个只用于性能剖析的东西放进了生产编码路径,是污染。
4. **没有可复现的数值基座**。`/tmp/ids72.txt`、`/tmp/golden72.bin`、`/tmp/ref128.bin` 都不在仓库里,session 里的 `/tmp` 一旦清空,所有"KL 1.4e-7 / top-1 72/72"的验收就无从重跑。仓库里唯一的自动化测试是 `tools/test_cache.sh`(11 项)。
5. **`engine.cpp` 是 517 行单文件**,且 `prefill` 里混着编码、计划、日志。没有中间表示,也没有"一个层 = 一个函数"的边界。
6. 小的不一致:`Makefile` 用 `-std=c++20`,文档写 C++17;`VERSIONS` 里 metal-cpp 的 commit 无法从 vendored 副本反查(没有 `.git`)。

### 2.3 功能缺口(与用户声明的用途直接相关)

用户的目标是"chat 和 agent 使用",目前缺三样:

1. **没有常驻服务**:每次 `run` 是一个新进程,**必然**付 3.3 s 的权重锁页(今天实测:三个连续进程各 3.55/3.33/3.40 s;文件页不会被留在 unified buffer cache 里)。agent 每步都启动一个进程的话,这 3.3 s 是纯开销。
2. **没有工具调用模板**:`chat.cpp` 只渲染 system/user/assistant,没有 `<tool_call>`/`<tool_response>` 段,也没有参数 JSON 的解析。agent 场景只能用 `--raw` 自己渲染。
3. **没有停止串、没有上下文滚动**:只有 EOG 停止;上下文满时直接报错,不会丢弃最旧的非系统轮次。

### 2.4 我这次读代码时确认的几处细节(不是 bug,但值得知道)

- `resetState()` 只清当前那一份 GDN 状态;另外两份(三缓冲)留着上一次的垃圾,但 `verify2` 在读之前一定会写,所以安全。这个不变量只存在于注释之外,建议写下来。
- 缓存检查点用 `--no-spec` 写的时候不含 MTP 的 K/V 与 `hPrev`,于是 `--no-spec` 与默认(开投机)的检查点**互不通用**。而 MTP 的 K/V 其实可以事后用 `encMtpKv` 重建(3 个矩阵的 GEMM,3K token 约 0.05 s)。
- 缓存的完整性只靠 `pread` 的返回值和 token 序列比对,没有校验和:半截文件能挡住,静默的位翻转挡不住。
- `verify2` 的 MTP 缓存行在被拒绝时会留脏,但下一个周期在读取前一定覆盖 —— 正确,但同样只存在于我的记忆里。

## 3. 性能账(今天的复核)

### 3.1 解码(64 token 上下文,MTP 关,交错 3 轮,单位 ms/token)

| 轮次 | 基线 | 去掉 `gemv_gate` | 去掉 `k_gdn_step` | 去掉 `k_attn_split` |
|---|---|---|---|---|
| 1 | 50.6(连测三次都是 50.6) | −9.6 | −2.8 | −0.1 |
| 2 | 51.4 / 52.5 / 52.6 | −11.2 | −2.8 | −0.3 |
| 3 | 52.3 / 52.6 / 52.9 | −10.1 | −3.4 | −0.6 |

分解(取 50.6 ms 的冷机值):

| 项 | 时间 | 上限 | 说明 |
|---|---|---|---|
| 权重读取(5.657 GB) | ~45 ms(GEMV 实测 126 GB/s) | 39.8 ms(142 GB/s) | 剩余 3.5–5 ms 是 ALU/延迟/tail |
| GDN 状态读写(302 MB) | ~3 ms | 2.1 ms | 纯带宽,只能压缩 |
| 其余(norm/FWHT/attention/conv/argmax) | ~2.5 ms | — | 依赖链上的串行部分 |
| **合计** | **50.6 ms = 19.8 tok/s** | **~42 ms = 24 tok/s** | 现实可达 ~46 ms(21.7 tok/s) |

**一次错误的期望要纠正**:顺序跑 19 个消融时,各项之和是 58 ms > 基线 53 ms。这不是加和公式错,而是**连续满载一分钟后 GPU 降频 ~10%**。任何单次消融的结论都必须有相邻基线(这也是 M17 里两项"看起来有效果"的优化被否掉的原因:交错之后没有差异)。

MTP 周期:验证 58.4 ms(5.657 GB → **97 GB/s**,比解码的 112 低,因为 T=2 的 LUT 内核是 ALU 受限的 113 GB/s,再加 1.2 ms 的双份状态写入和注意力)+ 草稿 5.7 ms(729 MB @128 GB/s)= 64.1 ms 换 1.60–1.66 个 token → 25 tok/s。
→ **若 T=2 的 GEMV 能到 T=1 的效率(126 GB/s),周期降到 ~55 ms,即 29 tok/s**。T=2 目前是每字节 6 条指令(half2 LUT),已是 ALU 上限;M12 也证明读取宽度不是瓶颈。所以这条路的结论是"到这里为止"。

### 3.2 预填充(1024 token chunk,今天两轮交错)

| 配置 | 总时间 | 其中 GEMM | GEMM 效率 |
|---|---|---|---|
| 精确 fp16 | 4.48 / 4.39 s | 3.46 / 3.39 s | 14.7 TFLOPS |
| `--fast-ffn` | 3.99 / 3.87 s | 3.10 / 3.01 s | — |
| 非 GEMM | 1.02 / 1.00 s | — | 展开 0.56 s + 其它 0.45 s |

(今天整体比前几天的 3.9 s 慢 12%,是降频,不是回归。)

| 项 | 占比 | 还能怎么降 |
|---|---|---|
| GEMM | 78% | 已到 fp16 峰值 96%;唯一杠杆是 int8(§5.E) |
| 权重展开 | 14% | 每 chunk 一次;层优先可让长提示只付一次(§5.D) |
| 其它(GDN/attention/norm/FWHT) | 11% | 已批量、已融合,单项都 <0.2 s |

7.3K 提示:精确 207–213 tok/s,`--fast-ffn` 226(+9.5%)。

### 3.3 长上下文

| 上下文 | fp16 KV | int8 KV |
|---|---|---|
| 7.3K,解码(普通 / MTP) | 17.5–18.0 / 21.4–21.5 | 18.1 / 21.3–21.5 |
| 29K,解码(普通) | 13.7 | **15.6(+14%)** |
| 29K,预填充 | 162 | 178(+10%) |

结论没变:KV 压缩只在 ≳16K 上下文有速度意义,其余是内存意义。

### 3.4 今天新发现的三个问题

**(a) 有损开关不能随意组合。** 各开关单独都过门限,叠加后不过:

| 配置(对 fp16 权重 + fp16 KV,3072 个位置) | 平均 KL | top-1 |
|---|---|---|
| `--fast-ffn` | 7.5e-5 | **99.87%**(3068/3072) |
| `--kv int8` | 1.3e-4 | **99.71%**(3063/3072) |
| **两者同时** | 2.0e-4 | **99.58%**(3059/3072)❌ |

误差近似线性叠加,门限是 99.7%,所以"两个有损开关组合"必须单独评测,并且 CLI 要么禁止、要么在同时打开时警告。这是质量问题,不是速度问题,但它决定了我们能不能把 `--fast-ffn` 设为默认。

**(b) 一次性进程永远付 3.3 s。** 三个连续进程中,`run --no-cache` 的首次预填充都是 3.3–3.6 s,文件页没有被保留;也就是说**每个 agent 步骤都在重复付权重锁页的钱**。缓存只能省计算,省不掉这 3.3 s(实测:整提示命中后 `restore` 0.03 s,但第一次解码要多花 ~2.7 s 去锁页)。

**(c) 不同路径的算术不一致。** 量化(72 位置):

| 对比 | 平均 KL |
|---|---|
| 预填充(fp16 激活)vs 逐 token(fp32 数学) | 1.4e-7(最大 7.4e-7) |
| 验证路径(half2 累加)vs 逐 token | 2.5e-7 |

都很小,但在近似平局处会让贪心输出翻转(我们在一段故事的第 690 个字符处见到过)。**"同一提示、不同长度/不同路径给不同答案"** 是用户可感知的行为不一致;而且缓存、`--kv`、`--fast-ffn` 都会改变走哪条路径。

## 4. 改进方案(按价值/风险排序)

每条写:目标、做法、预期、验收、风险。标 ★ 的是我建议做的。

### ★ A. 数值路径统一 + 可复现的测试基座(工程债,0 速度成本)

- **做法**:① `step` 改用 fp16 激活(与 `prefill` 同精度),用已实测同速的 `k_gemv_v1h_t1`(154 µs vs 148,噪声内);② 把层编码抽成 `encodeLayer(e, il, Mode)`(`Mode` = {decode, verify2, prefill-tok}),`step`/`verify2`/`prefill` 共用;③ `tools/make_golden.sh` 用 `ref_logits` 生成基座,`tools/test_all.sh` 跑:golden KL、`verify-check`、缓存回归、分词器一致性、跨路径贪心一致性(20 个提示 × 3 条路径)。
- **预期**:速度不变;消除"路径不同答案不同";此后每个改动都有机器可跑的回归。
- **验收**:`make test` 全绿;跨路径贪心一致性 20/20;`step` 与 `prefill` 的 KL ≤ 1e-6。
- **风险**:低。这是所有后续工作的前置条件。

### ★ B. 质量基座:任务级评测(决定能不能把有损开关设默认)

- **做法**:固定一组任务(20 道需要多步的算术/推理、8 段代码补全、20 段长文续写),贪心解码,分别在 exact / `--fast-ffn` / `--kv int8` / 组合下跑,比较**最终答案**是否一致(而不是 logits)。全套约 30 分钟/模式。
- **预期**:给"模型有没有变笨"一个用户能懂的证据;有它才敢改默认值。
- **验收**:每个模式的答案一致率 ≥ 95%,且差异逐条列出。
- **风险**:低(只是跑模型);但需要先定"任务集"是否算数(见 §7)。

### ★ C. 常驻服务 + agent 接口(功能缺口,用户的主用途)

- **做法**:`kata serve`:从 stdin/socket 读 JSONL 请求(`system`/`prompt`/`raw`/`max_tokens`/采样参数/`tools`),复用 Engine + PrefixCache,响应里带 timings;单线程排队(符合"单用户")。同一里程碑里补:Qwen3.8 的 tool-call 模板渲染 + 参数 JSON 解析、`stop` 串、上下文满时的滚动策略(丢最旧的非系统轮次并重建状态)。
- **预期**:agent 每步从 `3.3 s + prefill` 变成 `prefill`(缓存命中后 ~0.03 s);可用 `--raw` 以外的正规方式接工具。
- **验收**:100 个连续请求无泄漏;同一请求串行 3 次,第 2、3 次的首包 ≤ 0.2 s;工具调用的 JSON 能正确往返(用官方模板的 3 个示例比对)。
- **风险**:中(模板细节、状态滚动时的缓存失效)。

### ★ D. 层优先预填充(长提示 +12%)

- **做法**:外层循环层、内层循环 chunk,每个矩阵对整个提示只展开一次;激活按 chunk 常驻(8K:residual 164 MB + fp16 x′ 279 MB)。先做一个便宜的前置实验:chunk=2048/4096 配合重扫 `GM`(16/24/48),确认"chunk 变大就变慢"是否只是网格顺序没跟着调。
- **预期**:7.3K 提示 207 → ~235 tok/s(+13%);16K 以上线性受益。
- **验收**:与逐 chunk 的输出逐位一致;8K 预填充 ≥ +8%;常驻内存 +0.5–1 GB。
- **风险**:中(GDN 递推要跨 chunk 连续跑、注意力要按全前缀处理;一次性实现量不小)。

### ★ E. int8 GEMM 的 in-situ 效率(预填充 +7%)

- **做法**:in-situ 只有 ~20 TOPS(微基准 33.8,64%);原因是每 128 组都要做一次"int32→fp32 缩放累加"的 epilogue。可选:① 每 lane 预取全部 K 组的 scale 到寄存器;② K 循环双缓冲,让 `cInt` 的读取与 `cAcc` 的更新重叠;③ 激进方案:把每行的权重按行最大值重新量化成 int8(scale 只乘一次),代价是引入一次**权重**量化(与现有的激活量化不同),需要质量评测。
- **预期**:in-situ TOPS ×1.3 → `--fast-ffn` 从 +8% 到 +15%。
- **验收**:in-situ TOPS ≥ 26(即 1.3×);数值逐位一致(①②)或 KL 同量级(③)。
- **风险**:中(③ 有质量风险;①② 是纯性能工作)。

### F. fp16 GDN 状态(解码 +2.4%)

- **做法**:先用 fake-quant 探针(在 `k_gdn_step`/`k_gdn_rec_*` 之后把状态量化到 fp16 再写回)跑 `kl-eval` + needle;过门限再改内核与检查点格式(版本 +1)。
- **预期**:状态流量 302 → 151 MB/token,解码 −1.5 ms,MTP 周期 −1.2 ms。
- **验收**:3072 位置 KL ≤ 1e-3、top-1 ≥ 99.7%;needle 3/3。
- **风险**:低(有探针挡着),收益也小。

### G. 缓存工程化(体积、可移植、完整性)

- 链式增量检查点(消除"消息边界 + 结尾"两份的 2× 冗余);缺 MTP K/V 时用 `encMtpKv` 重建,让投机/非投机的检查点通用;可选 `--cache-verify`(xxhash)。
- **预期**:同样预算下检查点数量约翻倍;命中率与跨模式复用提高。
- **风险**:低。

### H. 小项(各 1.5–3%,排在最后)

采样感知草稿(+2–3%,模型已写好只差接入)、草稿 + 验证合一个 command buffer(+1.7%)、草稿词表子集(131072 行,+1.6%)。M17 已经把每一项的估算和理由写清楚了。

### 不建议做的(带数据,防止反复讨论)

| 想法 | 数据 |
|---|---|
| 重排权重布局 / 16 字节宽读取 | 暂存原型与原版同速(125–135 vs 127–134 GB/s);纯读取的 141 GB/s 带不进 GEMV(M12) |
| ~~更宽验证 + DFlash2~~ | **已推翻**:M14 拿 T=2 当基准而非带宽下限,原型内核有四处缺陷;见 `R2-dflash2-revisit.md`,已重开为 M25–M29 |
| 持久化 megakernel / ICB / 图重放 | 编码 0.7 ms/token(1.3%),dispatch 间隙 0.61 µs |
| CPU/ANE/SME 帮解码或展开 | CPU 写 DRAM 会把 GPU matmul 砍到 70%;ANE 的 MTP 依赖链串行(M0/CHECKPOINT-001) |
| 展开与 GEMM 重叠 | 实测更慢(3.9 → 5.6 s),缓存复用被打掉 |
| 二进制归档 / 预触摸页 / residency set | 稳态无收益,冷启动只是把 2.4 s 挪到别处 |

## 5. 下一步(M18–M24 里程碑)

| 里程碑 | 内容 | 验收 |
|---|---|---|
| **M18** | 数值路径统一(`step` 用 fp16 激活)+ `encodeLayer` 重构 + `make test`(仓库内 golden、跨路径贪心一致性、缓存回归) | `make test` 全绿;跨路径一致性 20/20;速度不降 |
| **M19** | 质量基座:任务级评测(算术/代码/续写),输出 4 种模式的一致率表 | 每个模式 ≥ 95%;差异逐条列出 |
| **M20** | `kata serve`(JSONL)+ tool-call 模板 + stop 串 + 上下文滚动 | 100 请求无泄漏;热请求首包 ≤ 0.2 s;3 个工具示例往返正确 |
| **M21** | 层优先预填充(先做 chunk/GM 前置实验) | 8K 提示 ≥ +8%;输出逐位一致 |
| **M22** | int8 GEMM in-situ 效率(①②;③ 视质量) | in-situ ≥ 26 TOPS;`--fast-ffn` ≥ +12% |
| **M23** | 有损开关的组合规则 + 默认值决定(`--fast-ffn` 是否默认、组合是否禁止) | 组合矩阵表;默认值有任务级证据 |
| **M24** | 缓存:链式增量、跨模式可移植、可选校验;fp16 GDN 状态(若 M19 判据通过) | 体积/命中率/质量三张表 |

顺序理由:M18 是基座(没有它,后面每一步都在流沙上);M19 决定 M22/M23 能不能落地;M20 是用户主用途的功能缺口;M21/M22 是两块剩下的真实性能;M24 收尾。

## 6. 需要你决定的事

1. **`--fast-ffn` 要不要设为默认?** 它单独 99.87%(过门限),但和 `--kv int8` 组合后 99.58%(不过)。三条路:① 默认开、组合改为互斥/警告;② 保持默认关(现状);③ 等 M19 的任务级评测再定。我倾向 ③。
2. **M20 的接口形态**:JSONL over stdin(最简单、无依赖)还是 unix socket + HTTP(接 agent 框架方便)?我倾向 stdin JSONL + 可选的 `--listen` unix socket。
3. **M21 愿意付 +0.5–1 GB 常驻内存吗**(64K 上下文时总占用会从 ~7 GB 涨到 ~8 GB,仍在 12.1 GB 工作集内)。
4. **M19 的任务集**:我提议 20 道算术/推理 + 8 段代码补全 + 20 段续写(约 30 分钟/模式);你如果有更在意的评测集(比如 GSM8K 子集、你自己的 agent 任务),现在指定最省事。
