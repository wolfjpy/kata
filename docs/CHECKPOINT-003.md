# CHECKPOINT-003:向 Strata / ninfer 学到的东西、实测数据、已确认的决定和路线图

日期:2026-10-25 · 自包含的恢复文档:**重启后只读本文件(以及它指向的文件)就能继续工作**。
前序文档:`CHECKPOINT-001.md`(设计阶段)、`CHECKPOINT-002.md`(已实现引擎的状态、文件地图、复现命令、被实测推翻的假设)、`reports/`(每个里程碑一份报告,M7 是 drafter 分析)。

## 0. 恢复指南(先读这里)

- 项目根:`<home>/super.engineering/projects/bonsai`。模型:`../../models/bonsai-2-27b-mtp-ptq1_0.gguf`。构建:`make`。
- 参考仓库(在项目外,持久):`../../refs/Strata`(commit `4b9e05a`)、`../../refs/ninfer`(`64d10fb`)、`../../refs/llama.cpp-prism`(`b3ebc47`,另已额外编译 `llama-speculative-simple`)。drafter 权重:`../../models/drafters/`(`dflash2-q4km.gguf` 1.14 GB、`bonsai2-2b-ptq1.gguf` 0.43 GB)。
- 如果参考仓库不见了:`git clone https://github.com/Niko1221/Strata`、`https://github.com/Neroued/ninfer` 到 `../../refs/`。
- 项目根里有一个未跟踪的文件 `1. **解码 GEMV 是 ALU 受限…(M4 §3)。T=.md`:是用户消息被保存下来的,不是我生成的,不要提交也不要删除。
- 工作规则(用户要求):回复用中文;只提交自己生成的 `docs/ shaders/ src/ tools/ Makefile VERSIONS config/ third_party/ README.md`,**绝不提交用户放在根目录的 PDF/PNG/笔记**;提交信息用英文;每个里程碑独立提交并附报告;最新版本规则(每个阶段开始时重新核对上游版本并记入 `VERSIONS`);所有有损选项必须过质量门限(KL ≤ 1e-3 nat、top-1 ≥ 99.7%,≥20K token 语料,用 `bonsai kl-eval`)且默认关闭。
- 本文 §7 的路线图是当前的工作计划,§8 是已确认的决定。下一步从路线图里第一个未完成的里程碑开始。

## 1. 当前性能基线(M6,Mac mini,GPU 持续满载 1 分钟无降频)

| 指标 | 数值 | 参照 |
|---|---|---|
| 解码,普通 | 19.8 tok/s | Prism llama.cpp 15.6 |
| 解码,MTP 深度 1(贪心 / 采样) | 25.0 / 22.7 tok/s | |
| 解码,MTP,7.3K 上下文 | 21.4–21.7 tok/s | |
| 预填充精确 fp16,512 / 1024 token | 245 / 257 tok/s | llama.cpp 208 (pp512) |
| 预填充精确 fp16,7.3K token | 213 tok/s | |
| 冷启动 | ~3 s(首次预填充 3.3 s,稳态 0.92 s) | 聊天模式已在后台线程热身 |

## 2. 两个参考仓库

### 2.1 Strata(Niko1221/Strata,MIT,C++/CUDA/HIP/SYCL/**Metal**)

产品化的本地推理平台(桌面应用、终端、OpenAI/Anthropic API)。引擎里有一个独立的 Metal 实现(`metal/`),在 **M5 Max 128 GB** 上跑 Qwen3.8-Flash-Next(MoE,Q2_0 37.6 GB)。关键文档:`docs/MAC.md`(Metal 引擎的全部测量和开关)、`docs/DETAILS.md`、`docs/MESSAGE_BOUNDARY_CACHE.md`、`docs/PROMPT_CACHE_TAIL.md`。关键源码:`metal/kernels/{mma,gdn,gdn_prompt,dense_mm,gemv,units,quant,alu_bench,gridbar}.metal`、`metal/src/gpu/runtime.cpp`、`src/spec/{suffix_drafter,draft_policy,controller}.cpp`。

测量(M5 Max):解码无 MTP 87.4 tok/s(llama.cpp 56.6);`--spec 2` + MTP + suffix drafter 119.5/108.2(散文/故事);suffix drafter 贡献约 +5%(114→119.5);窗口扫描:spec 2 对散文最好、spec 3 对混合问答最好、≥4 更差;预填充约 1430 tok/s(llama.cpp 887)。

它的设计(与我们相关的部分):
1. **宽验证内核 `mma.metal`**:"一个窗口 1–8 个 token 的 GEMV,在 simdgroup 矩阵单元上":每个 simdgroup 负责 8 行输出,每 128 宽的 K 块把自己的 8×128 权重解码进**私有**的 TG 内存 tile(`wt[SG][8*128]`,没有跨 simdgroup 屏障),再用 16 步 `simdgroup_multiply_accumulate`(A=8×8 权重块,B=8(k)×8(token))。验证窗口的代价几乎与窗口大小无关。
2. **GDN 验证不写状态**:`gdn.metal` 里 `commit=0` 的验证窗口不写状态,接受 m 个 token 后用同一个内核(`GDN_NTOK=m, commit=1`)重放;`0x100|k` 模式在已知接受数时原地提交。
3. 一个验证窗口一个命令缓冲区;**整个 MTP 草稿轮在一个命令缓冲区**(`STRATA_METAL_MTP_CHAIN`)。
4. **suffix drafter**(98 行):三元组哈希 → 最近 4 个结束位置,取最长匹配(≤32),不需要模型,微秒级。**`DraftPolicy`**:在线 EMA 学每个窗口大小的实测轮耗时与按匹配长度分 4 桶的接受率,按"每毫秒期望提交 token 数"在 MTP 窗口和 lookup 窗口之间选择(`E=1+q+…+q^k`,利润率 margin 3%);带按上下文长度分档的成本表。
5. 预填充:`dq_f16`(整个矩阵展开成 half,每 chunk 一次)+ MPP `matmul2d`(`gemm_f16<TM,TN,SG>`,B 用 `transpose_right`),chunk 2048(4096 仅 +2–4%)。`gdn_prompt.metal`:三段式(pre 并行、scan 状态在寄存器、post)。
6. 运行时:所有缓冲区 `storageModeShared`、hazard tracking `Untracked`、一个 `MTLResidencySet` 挂在每个队列上;图重编码约 0.17 µs/dispatch。`gridbar.metal` 只是可行性探针,引擎里没有使用持久化内核。
7. 上下文缓存:对话文件、**消息边界检查点**(在最后一条用户消息之前存一份,agent 编辑最后一条消息时不用重读整个历史)、`PROMPT_CACHE_TAIL`。
8. 流式 MoE 专家(SSD)、n-gram 表、Gyro 格式——与我们无关。

### 2.2 ninfer(Neroued/ninfer,Apache-2.0,C++/CUDA,仅 RTX 5090)

从零写的单卡引擎,Qwen3.5/3.6/3.8 + MTP + DFlash/DFlash2 + 视觉 + 前缀缓存 + 调度器。价值全在文档(`docs/maintainer/`):`dflash.md`(DFlash/DFlash2 的完整数学和实现合同)、`replayssm-gdn.md`(GDN 回放)、`linear-tuning.md`(线性层调优方法论)、`storage-layouts.md`(权重分平面布局)、`paged-kv-cache.md`、`engine-architecture.md`、`resource-scheduling-and-context-cache.md`、`docs/performance/qwen3.8-27b.md`(测量)。

测量(RTX 5090,Qwen3.8-27B,FP8 E4M3 row-256 KV,CUDA Graphs):
- 无投机解码:84.1 tok/s(groupwise-int,15.9 GB → 1.34 TB/s = 75% 名义带宽);NVFP4 74.1。
- MTP3:代码 193.5(3.29 token/轮)、故事 123.4(2.10)、结构化 214.7(3.65);DFlash2 K=7:代码 258.6(4.75)、结构化 348.9(6.41)、故事 115.0(2.11)、短推理 305.7(5.61;MTP3 196.1)、长推理 177–189(3.4–3.6;MTP3 149–155)。
- **DFlash2 一轮 ≈ 1.55× 单步解码**(代码:4.75 token / 258.6 tok/s = 18.4 ms,单步 11.9 ms)。
- 预填充:groupwise-int 3331 tok/s(7.7K 提示,约 160 TFLOPS ≈ 77% BF16 峰值);NVFP4 12.8K tok/s(FP4 张量核)。
- 线性层调优方法论:"热区间" T=1..128,优先级 T=1 > T=4,8 > 其它;每个 T 都测;以"持续读取天花板"(1674.5 GB/s,实测)为参照。

设计要点:
1. **DFlash2**:见 §4。
2. **ReplaySSM**:见 §5。
3. **权重分平面布局 `row_split_k128_v1`**(Q4/Q5/Q6/Q8):码平面、高位平面、scale 平面分开,每个平面 256 字节对齐,平面内按"行 0 组 0、行 0 组 1…"。16 字节向量读。Prism 的 CUDA 后端也有 planar PTQ1_0(重排的是 **激活**:Q8_1 → 8 个 16 字节平面 + 1 个 scale 平面),因为 CUDA 用 dp4a;Apple GPU 没有 dp4a。
4. **KV 格式**:BF16、INT8(group-64)、FP8 E4M3(row-256,即每个 head 每个 token 一个 scale)、NVFP4、K8V4;配有困惑度评测工具(`docs/perplexity.md`)。Qwen3.8 默认 FP8 row-256。
5. **优化草稿头 `--lm-head-draft`**:按 token 频率排名的 131072 行子集(全词表 248320 的 53%),只用于草稿提议,验证仍用全头。
6. **前缀/上下文缓存**:"一个可复用的检查点 = KV + 该 token 边界处完整的续写状态(GDN 状态等)";设备/主机共用容量,压力下回收保留的资源,暂停的请求可从快照恢复或重放。
7. CUDA Graph 精确 B 拓扑;投机单元分 Forward、Finish 两段图,CPU 在两段之间准备约束 mask。

## 3. 本轮在草稿 / 验证方面的实测(M7 报告有更多)

### 3.1 接受率(Prism `llama-speculative-simple`,Bonsai-2-27B 为目标,贪心)

| drafter | 散文 | 代码 | 推理(思考) |
|---|---|---|---|
| MTP 深度 1 | 1.6(我们引擎) | — | 1.86 |
| MTP 深度 3 | 1.87 | 3.2 | 3.13 |
| DFlash2(Q4_K_M,7 个草稿) | 2.2 | 5.1 | 5.1 |
| Bonsai-2 2B(普通草稿模型,4 个草稿) | 1.23 | 2.07 | — |

(单位:每轮验证产出的 token 数。)与 ninfer 在 Qwen3.8 上的数据一致,说明 Bonsai 作为 DFlash2 目标模型迁移良好。

### 3.2 宽验证的代价(`tools/bench_gemv2`,ffn_gate 5120→17408,每矩阵)

LUT+half2 GEMV:T=2 170–185 µs,T=4 296,T=6 397,T=8 590–598(3.2×)。矩阵单元原型(`k_wide_gemm`,TG 内存 fp16 块 + `matmul2d`,M=16,4 个 simdgroup 共享一个 20 KB 的 TG 块、TG 级屏障)831 µs:数值正确(最大误差 3e-6)但太慢,原因是每核只能驻留 1 个 TG、每个 K 块串行"加载→解码→屏障→matmul→屏障"。**Strata 的结构(每个 simdgroup 私有 tile)正是解决这一点的办法。**

### 3.3 矩阵路径的吞吐(M6,本轮新测,探针在 `docs/probes/`)

| 路径 | 吞吐 |
|---|---|
| 经典 `simdgroup_matrix` 8×8×8(half/float 相同,走 ALU) | **5.2 TFLOPS** |
| `matmul2d` half×half→float | 15.7–16.0 TFLOPS(64×128 / 4 SG) |
| `matmul2d` half×int8→float / half | 14.3–15.1 TFLOPS(**≈ fp16,没有 int8 加速**) |
| `matmul2d` int8×int8→int | 33.8 TOPS |

含义:Strata 的 `simdgroup_matrix` 做法在 M6 上直接照搬不够快(T=8 光 MMA 就要 ~80 ms);要做"每 simdgroup 私有 tile + `matmul2d`",且需要 M=8 的 tile(否则 T≤16 按 16 个 token 计费)。T=8 的 MMA 在 12.6 TFLOPS 下约 33 ms,与 ~40 ms 的带宽下限重叠,预期验证 ~55–70 ms。

### 3.4 运行时选项(临时拷贝里测的,稳态均无收益)

CPU 编码一个解码步 0.7 ms(1.3%);`MTLResourceHazardTrackingModeUntracked` 和 residency set 在稳态都在噪声内;residency set 让首次预填充降到 0.92 s,但进程总耗时仍 ~3.9 s(同样的 ~2.4 s 搬到了加载阶段,`requestResidency()` 同步等待 6.4 GB 锁页)。**冷启动 = OS 对 6.4 GB 映射的锁页,只能隐藏,不能消除。**(`docs/probes/residency_untracked_experiment.diff`)

### 3.5 int8 预填充的帕累托(24.5K token 语料,1536 个位置,`kl-eval`)

| 配置 | 平均 KL | 最大 KL | top-1 |
|---|---|---|---|
| 全部 int8(per-group 激活 scale) | 4.4e-4 | 0.34 | 99.54% |
| **只对 FFN gate+up(约 47% FLOPs)** | **6.1e-5** | 9.1e-3 | **99.80%** |
| gate+up+qkv+z+wq | 5.9e-4 | 0.76 | 99.67% |

误差主要来自喂给 GDN 循环状态的输入投影(qkv/z/wq);均值 KL 被少数离群位置主导,top-1 更稳。int8 GEMM 在实测里只有 ~21.5 TOPS(epilogue 开销),所以"只做 gate+up"的预填充收益约 8–10%。

## 4. DFlash2 规格摘要(给后面可能的移植用;原文 `refs/ninfer/docs/maintainer/dflash.md`)

- 草稿配置(Qwen3.8-27B 官方):H=5120,5 层,FFN 17408,Q/KV 头 32/8,head_dim 128,全部是局部注意力,滑动窗口 S=2048,RoPE θ=1e7,**目标层块 ID `[5,19,33,47,61]`**,双抽头动态分组卷积(组宽 16),候选选择器 top-16、rank 256。1.92B 参数(BF16 3.85 GB;GGUF Q4_K_M 1.14 GB,Q8_0 约 2 GB)。
- 条件:对每个已处理位置 t,取目标第 l 层块输出的**完整残差**(mixer+FFN 之后、下一层输入 norm 之前)`r_t^l`,拼接 `s_t=concat(r_t^l)`,`c_t = plain_rmsnorm(W_feature s_t, context_norm)`;每个草稿层把 `c_t` 投影成自己的上下文 K/V(`k_ctx = rope(plain_head_rmsnorm(W_ck c_t))`,`v_ctx = W_cv c_t`)。**上下文预填充只做特征投影和 K/V 投影,不让提示 token 走草稿的残差/MLP 栈。** 只需要保留最后 2048 个位置(环形 KV,5 层 × 40 MiB 总共)。
- 一轮:已提交前沿 F,锚点 token `a`;草稿输入 = `[a, MASK×K]`(位置 F..F+K,K=7 → 块宽 8);非因果注意力(上下文 + 块内),局部层 `|p_key − p_query| < S`;一次前向产出 K 个提议。头:per-column top-16 候选 → DFlash2 选择器 `E_i[p,c] = u_i[c] + Σ_r W_pred[pred,r]·g_i[r]·W_succ[C_i[c],r]` 做条件路径游走(温度>0 时按 softmax 采样,保留 q_i)。
- 验证:目标一次处理 `[a, d_1..d_P]`;贪心取匹配前缀 + 修正/奖励 token;采样时按 `min(1, p/q)` 接受、拒绝时从 `max(p−q,0)` 重采样(保持目标分布)。
- 每个提交的位置,其特征变成"待物化的上下文",在下一次草稿之前才做投影。
- 草稿的 5 层是 Qwen3 风格;`dynamic_grouped_conv`、`candidate_selector`、`context_kv_materialize`、`sliding_window_attention` 在 `refs/ninfer/include/ninfer/ops/*.h` 里有精确的数值合同。

## 5. ReplaySSM 摘要(原文 `refs/ninfer/docs/maintainer/replayssm-gdn.md`)

验证从已提交的检查点 `S_0` 出发,**逐 token 产生输出和 raw 转移记录 `R_t`,但不写状态**;接受 m 个后,从 `S_0` 顺序重放 `R_1..R_m` 得到 `S_m`。27B:一份状态 144 MiB(我们 151 MB)+ conv 历史 2.8 MiB;每 token 的记录 1.705 MiB(raw k/v + g/β)——是快照的 1/86。关键约束:**重放必须执行与验证相同的有限精度转移**(同样的归一化、门控、归约顺序、存储边界),否则状态会跨轮漂移。我们现在 T=2 时每个位置写一份完整状态(三缓冲),T=8 时不可接受(1.2 GB/轮)。

## 6. 提炼出的结论(每条都有数据)

1. 解码 GEMV:T=1 是带宽受限;T≥2 的验证才是 ALU 受限(CHECKPOINT-002 §4.1 已更正)。
2. T=1 的剩余 8–10% 来自读取模式:原生 28 字节块上的小读取 125–132 GB/s,16 字节向量读取 141–142 GB/s(`tools/weight_read_probe`)。要拿回只能重排权重(分平面,16 字节对齐)。
3. 预填充精确路径 GEMM 已到峰值的 ~96%;剩余非 GEMM 开销(展开 0.56 s + 其他 0.4 s)合计约 1 s,全消掉也只是 +28%。
4. `half × int8` 在 M6 上没有 int8 加速;唯一超过 fp16 速率的矩阵模式是 int8×int8,且需要激活量化。
5. 冷启动不能消除(锁页),可以隐藏(后台热身 / residency set 在后台线程请求)。
6. 降频:从冷机起连续满载 1 分钟(14 次 1024-token 预填充)耗时平坦;降频出现在连续满载数分钟之后,以及功耗更高的纯 matmul 探针上。
7. 在现有 ALU 内核下,深度 1 的 MTP 是最优点;更深的投机需要先有便宜的宽验证。

## 7. 路线图(里程碑;每个里程碑独立提交并附报告)

所有里程碑遵循:开始前核对上游版本;先写数据驱动的验收标准;实现;用 `verify-check`/`kl-eval`/`--cmp-logits` 验证数值;测速(交错 A/B、冷却);写 `reports/Mxx-*.md`;更新 `README.md` 状态表和 CHECKPOINT;提交。

| 里程碑 | 内容 | 验收标准 | 预期收益 |
|---|---|---|---|
| **M8** | 本文档(学习与决定的汇总) | — | — |
| **M9 ✅(2026-10-25,`reports/M9-prefix-cache.md`)** | **持久化前缀缓存**:在用户消息边界把 GDN 状态 + conv 状态 + KV(含 MTP 的 K/V)写到 SSD,按 token 前缀的哈希索引;新进程/新会话在长提示上复用;LRU + 磁盘预算 | 命中后恢复的 logits 与从头预填充逐位一致(或 KL ≤ 1e-6);20K token 提示的首包 ~80 s → ≤ 5 s | agent 场景首包提升 10×+ |
| **M10 ⏸(2026-10-25,`reports/M10-sampling-aware-draft.md`:离线估计只 +5 个点,闸门未达到,实现推迟)** | **采样感知草稿**:温度>0 时从 MTP 自己的分布 q 采样草稿,验证用 `min(1,p/q)` + 残差重采样;先离线估计 Σ min(p,q) | 采样模式接受率 49% → ≥ 58%,输出分布不变(用大量采样的频率检验) | 采样模式约 +8% |
| **M11** | **int8 KV**(每 head-row 一个 scale;`half×char` 原生 matmul2d,K 缩放 S 的列、V 缩放 P) | 24K 语料 KL ≤ 1e-3、top-1 ≥ 99.7%;32K 解码 +8–10%;内存减半 | 长上下文解码、64K 内存 |
| **M12** | **权重分平面重排**(`bonsai-pack` 工具生成 planar 文件,GEMV/展开/embedding 读 16 字节向量)| 数值逐位一致;普通解码 19.8 → ≥ 21.2 tok/s | 解码 +8–10% |
| **M13** | **FFN gate+up 的 int8 预填充开关**(`--fast-ffn`,有损,默认关) | 24.5K 语料 top-1 ≥ 99.7%(已测 99.80%),预填充 +8–10% | 预填充 +8–10% |
| **M14** | **宽验证微基准闸门**:`matmul2d` 每 simdgroup 私有 tile,M=8,权重 LUT 解码进私有 TG tile;测 T=4/8 | **T=8 每矩阵 ≤ 1.5 × T=2(≈ 260 µs @ ffn_gate)才算通过**;否则停止 DFlash2,在报告里结案 | 闸门 |
| **M15** | (仅当 M14 通过)ReplaySSM 重放 + 宽验证接入引擎 + 采样兼容 | verify-check:KL ≤ 1e-5;T=4/8 验证周期 ≤ 1.5× T=2 | DFlash2 的前提 |
| **M16** | (仅当 M15 完成)DFlash2 移植(Q8_0 或 Q4_K_M 草稿权重 + 宽 GEMV;上下文物化器、动态卷积、选择器、滑窗注意力) | 与 llama.cpp `draft-dflash` 的接受长度一致(代码 ~5.1);无损;代码/推理 ≥ 50 tok/s | 解码 2×(代码/推理) |
| **M17** | 小项:suffix drafter + 在线成本策略(有宽验证后才有意义);单命令缓冲区草稿+验证;草稿词表子集(131K 行) | 各自的收益 ≥ 1% 才保留 | ≤ 5% |

顺序理由:M9–M13 都是有数据支撑、风险低的收益;M14 是 DFlash2 的决策闸门,放在 M13 之后,但如果前面的里程碑出现阻塞可以提前做。

## 8. 已确认的决定(用户,2026-10-25)

1. **放入前缀缓存**(M9)。上下文覆盖范围由我选默认值:支持到引擎的上下文上限;磁盘预算默认 40 GB(256 GB SSD),LRU 回收,每个检查点 = 151 MB GDN 状态 + 6 MB conv + 64 KB × token 数的 KV(20K token ≈ 1.3 GB,32K ≈ 2.1 GB,64K ≈ 4.2 GB);都可配置。
2. **DFlash2 作为多天高风险项目,先用宽验证微基准做闸门**(M14 → M15 → M16)。
3. **引入有损的 FFN gate+up int8 开关**(M13),默认关闭,文档写明质量数据。
4. 用户的总体指示:没有进一步问题就按计划建设;每个里程碑做检查点并按 git 最佳实践提交(小而专注的提交、英文提交信息带正文、每个里程碑一份报告、不提交用户的文件)。

## 9. 风险与注意事项

- GPU 在连续满载数分钟后会降频:测速必须交错 A/B、重复、必要时冷却 1–2 分钟。
- 72 个位置的小测试会掩盖误差累积:量化类改动必须在 ≥20K token 语料上评估。
- 前缀缓存的正确性对象是"完整的续写状态",KV 页存在不等于命中:必须同时恢复 GDN 状态、conv 状态、MTP K/V 和 `hPrev`;chat 模板会在后续轮次剥离旧的思考内容,因此检查点位置应选在**用户消息边界**,且键里要包含模板渲染后的 token 序列。
- 64K 上下文常驻内存已贴近 12.7 GB;int8 KV(M11)会缓解。
- DFlash2 的验证需要 ReplaySSM 的"重放必须与验证有限精度完全一致"约束;我们目前的 `k_gdn_rec_v2` 与重放内核必须共用同一份算术。

## 10. 参考文件速查

| 主题 | 位置 |
|---|---|
| Strata Metal 宽验证内核 | `refs/Strata/metal/kernels/mma.metal` |
| Strata GDN 验证/重放 | `refs/Strata/metal/kernels/gdn.metal`、`gdn_prompt.metal` |
| Strata suffix drafter / 策略 | `refs/Strata/src/spec/suffix_drafter.cpp`、`draft_policy.cpp`、`include/strata/spec/*.hpp` |
| Strata Metal 运行时 | `refs/Strata/metal/src/gpu/runtime.cpp` |
| Strata 消息边界缓存 | `refs/Strata/docs/MESSAGE_BOUNDARY_CACHE.md` |
| ninfer DFlash/DFlash2 | `refs/ninfer/docs/maintainer/dflash.md`、`refs/ninfer/include/ninfer/ops/*.h`、`refs/ninfer/src/models/qwen3_5/execution/draft.cpp` |
| ninfer ReplaySSM | `refs/ninfer/docs/maintainer/replayssm-gdn.md`、`include/ninfer/ops/gdn_replay.h` |
| ninfer 权重布局 / 调优 | `refs/ninfer/docs/maintainer/storage-layouts.md`、`linear-tuning.md` |
| ninfer 上下文缓存 | `refs/ninfer/docs/maintainer/resource-scheduling-and-context-cache.md`、`paged-kv-cache.md` |
| 本轮探针 | `docs/probes/simdgroup_mma_probe.*`、`matmul2d_mixed_type_probe.cpp`、`residency_untracked_experiment.diff`;`tools/{weight_read_probe,kv_read_probe,bench_gemv2,probe_flash,alu_probe,layout_probe}.cpp` |
| 草稿接受率日志 | `/tmp/ss_*.log`(临时);汇总在 `reports/M7-speculation-options.md` |
