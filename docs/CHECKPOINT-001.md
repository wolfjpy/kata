> 本文以中文写成。英文摘要见双语文档索引 [docs/README.md](README.md)。
> *This document is in Chinese; see the bilingual index at [docs/README.md](README.md) for an English summary.*

# Kata-M6 推理引擎 — Checkpoint 001

- 写于:2026-10-23(本机时间 13:55 EDT 之后)
- 作用:**让一个没有任何历史记忆的新会话,仅凭本文件和 `docs/` 下的配套文件就能无缝接着干。**
- 阶段:**讨论/设计 + 硬件与数据探测**。**还没有写任何引擎代码**,也没有 commit(git 仓库 `main` 分支尚无任何提交,所有文件均为 untracked)。
- 用户最初明确说过 "do not implement yet. let's discuss the idea and solution together first."。之后用户的指示逐步给出了方案决策,但**没有明确说"开始实现"**。最后一次助手回复里问了三个待确认事项(见 §0.3),**用户没有回答,而是要求写本 checkpoint**。所以恢复后第一件事是向用户确认这三项(或在用户说"继续/go"后直接执行)。

---

## 0. 恢复指南(先读这一节)

### 0.1 一句话任务

为 **Ternary Bonsai-2-27B**(Prism ML 基于 Qwen3.8-27B 的三值量化版,Ollama 标签 `mannix/Bonsai-2-27B:mtp-ptq1_0`,6.4GB GGUF)写一个**只用 Apple Metal(C++ 加 MSL,metal-cpp)、不用 Python、只针对 Mac mini M6 16GB/256GB 这一台机器、速度是唯一目标**的推理引擎:prefill 极快,decode tok/s 最大。

### 0.2 与用户协作的规则(从对话中总结)

1. 用户用中英混合写作。**用户有一次明确要求"answer in chinese"**,之后的回复一直用中文,继续用中文回复(代码和标识符用英文)。
2. **永远使用所有东西的最新版本**(用户明确指令:"always make sure to use the latest version of everything.")。每个阶段开始前,都要重新核对 Apple SDK/系统、metal-cpp、Prism llama.cpp fork、BaseRT、模型仓库的最新状态,并把固定的版本/commit 记录到 `VERSIONS` 文件(待建)。无法核实的要明确标注。
3. **质量底线:速度优先,但不能让模型变笨。** 任何有损手段(int8 激活、KV 量化、fp16 GDN 状态、Q4 的 MTP 等)都必须通过质量门槛才能启用(§10.15)。
4. 先讨论再实现。写引擎代码前确认用户已经放行。
5. 并行 tool call 里**有依赖关系的命令不要并行**(本次对话里 `mkdir` 和 `cd` 发生过竞争)。
6. 回复保持简洁(系统规则);本类 checkpoint/报告除外。
7. `/tmp` 下的探测文件是临时的,已复制到 `docs/probes/`、`docs/data/`、`docs/reference/`。
8. python3 虽然存在(`/opt/homebrew/bin/python3`),**引擎和工具链一律不用 Python**。探测用 Swift 或 C++。(本次对话里有两处用 python 做一次性 JSON 解析的临时命令,不属于项目。)

### 0.3 待用户确认的三件事(上一条助手消息提出,尚无回答)

1. 建项目骨架:vendoring 最新 metal-cpp(固定 commit)、`config/kata_m6.h`、`VERSIONS`、`Makefile`。
2. `brew install cmake`(最新版),编译 Prism llama.cpp 最新 master 作基线和对照实现。
3. 之后进入 M0 微基准(§13)。

### 0.4 恢复后立刻执行的环境复核命令

```sh
sw_vers; xcode-select -p; DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer xcodebuild -version
export DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer
xcrun metal --version          # 期望 Apple metal version 32023.921 或更新
softwareupdate -l              # 期望 No new software available(否则记录新版本)
sysctl hw.memsize hw.perflevel0.physicalcpu hw.perflevel1.physicalcpu hw.perflevel2.physicalcpu
# metal-cpp 最新:
curl -s https://api.github.com/repos/apple/metal-cpp/commits?per_page=1 | grep -E '"sha"|"date"|"message"'
# Prism fork 最新:
curl -s "https://api.github.com/repos/PrismML-Eng/llama.cpp/commits?sha=prism&per_page=1" | grep -E '"sha"|"date"'
# 目标模型当前 manifest(确认权重 blob 没变,期望 174c9c6584e1... / 6397969920):
curl -sL -H 'Accept: application/vnd.docker.distribution.manifest.v2+json' https://registry.ollama.ai/v2/mannix/Bonsai-2-27B/manifests/mtp-ptq1_0
```

注意 GitHub API 未认证每小时 60 次,省着用。

### 0.5 本文件和配套文件索引

| 路径 | 内容 |
|---|---|
| `docs/CHECKPOINT-001.md` | 本文件 |
| `docs/data/gguf_header_dump.txt` | 目标 GGUF 的完整头部解析:51 个 KV、866 个张量(名称、维度、类型、偏移) |
| `docs/probes/gguf_dump.swift` | GGUF 头部解析器(Swift) |
| `docs/probes/pdf_extract.swift` | PDFKit 提取 PDF 文本(系统没有 `pdftotext`) |
| `docs/probes/gpu_probe.swift`, `gpu_family_probe.swift` | 设备能力探测 |
| `docs/probes/msl_language_version_probe.cpp` | 探测运行时编译器接受的最高 MSL 版本 |
| `docs/probes/scale_analysis.cpp` | 真实权重的 scale/trit 统计 |
| `docs/probes/matmul2d_probe.cpp` | MPP `matmul2d` 吞吐扫描(多类型、多 tile) |
| `docs/probes/matmul2d_int2b_probe.cpp` | `int2b`/`int4b`/fp8 格式 tensor 的编译探测(目前失败,见 §6.4) |
| `docs/probes/metalcpp_smoke.cpp` | 新版 metal-cpp 冒烟测试(Apple11、4_1、MTL4 对象) |
| `docs/reference/prism_ptq1_0_quant_dequant.c` | **PTQ1_0 量化/反量化的权威实现**(元素排列顺序) |
| `docs/reference/prism_metal_mul_mv_ptq1_0.metal` | Prism 的 Metal PTQ1_0 mat-vec(含 multicol) |
| `docs/reference/prism_metal_fwht.metal` | Prism 的 Metal FWHT(含融合 sign、融合 SwiGLU、threadgroup 变体) |
| `docs/reference/prism_metal_gated_delta_net.metal` | Prism 的 Metal GDN 递推内核(含状态快照/回滚相关) |
| `docs/reference/prism_metal_dequantize_ptq1_0.h` | PTQ1_0 的 Metal 反量化片段 |
| `docs/reference/prism_qwen35_graph.cpp` | Prism 的 qwen35 计算图(GDN、attention、MTP 的权威实现) |
| `docs/reference/hf_transformers_modeling_qwen3_5.py` | HF transformers 的 qwen3_5 实现(Apache-2.0,公式权威参考) |

项目根目录里**用户放入的参考资料**(不要修改):

| 文件 | 说明 | SHA-256(完整) |
|---|---|---|
| `实现Ternary Bonsai-2-27B (Qwen 3.8 27B) in.md` | 用户最初的需求原文(含内存层级表、baseRT 说明、PyTorch 参考链接) | `e3c0793037044bcd9fa2daa8380c042bd3b8d2a6774e1b13341a78d12729d77a` |
| `Apple Silicon 的 CPU 核心分三类:.md` | 用户的第二批笔记(CPU/ANE/Metal 4 编码/内存预算/各种要求) | `0fcc132431f106d05ff15a18ecd29d80207e4cd6ea99ef9ea34c17ad0f708835` |
| `Apple M6 vs M5 Chip Deep-Dive.png` | M5 对 M6 的 die shot(见 §6.1) | `de5fa4deea03868d96c145aa177ab830cb68902ef35ba4be04e36d2a58ef1606` |
| `qwen3.5.webp` | Sebastian Raschka 的 Qwen3.5 架构图(Gated Attention、Gated DeltaNet、3:1 比例、KV 随上下文增长对比) | `533a8c1a58564632a45923e42ecd1f085a00d84c2102d6676ac6807154dde10d` |
| `bonsai-2-27b-whitepaper.pdf` | Prism 白皮书(2026-09),14 页 | `aea10331ede3b34c34c21d1a45b80fd0fd6e231b3e8db7bd6346e20fcb8402c4` |
| `BaseRT-base.pdf` | BaseRT 第一篇论文,arXiv 2607.00501v1,11 页 | `722762bc076985b4da743dcf7cfe761e147c2f294e7905b1ac919a48f577eb54` |
| `baseRT.pdf` | BaseRT 第二篇论文(M5 Neural Accelerator),arXiv 2607.19438v1,8 页 | `28e5a6efacb3eee1fac0227e8571cd297636ec16be463f5c3c41085811d7120f` |
| `Metal-Feature-Set-Tables.pdf` | Apple 官方,2026-05-21,到 Apple10(M5)为止,18 页 | `9f31df15dd6827545702c5a0845f6e36e1889878cd0e534123bd70211e5c00a8` |
| `Metal-Shading-Language-Specification.pdf` | Apple 官方,MSL 4.1,2026-06-04,383 页 | `41538b30d2f1140a5b2a0c84ce0a9f7b67bf0c707e224cfea0bfe5a44aa26cf5` |

---

## 1. 用户的全部要求(按出现顺序,尽量保持原意)

### 1.1 最初需求(文件 `实现Ternary Kata…md`)

1. 目标模型就是 `https://ollama.com/mannix/Bonsai-2-27B:mtp-ptq1_0`。
2. 目标硬件就是 **Mac mini M6、16GB UMA、256GB SSD**。
3. **速度是唯一目标**:prefill 极快,decode tok/s 最大。
4. **只用 Apple Metal,完全不用 Python。**
5. 与硬件协同设计,极致优化。
6. 参考:用户自己写的 M6 存储层级表(寄存器/L1/L2/SLC/UMA/ANE,里面很多带宽数字没有出处,我不把它当设计依据,以实测代替)、baseRT 的设计思路(融合、CUDA Graphs 的对应物、专用 GEMM、零开销调度)、PyTorch 参考实现 `https://github.com/rasbt/LLMs-from-scratch/blob/cdbd33e6d57a71e20ea31434841095fac50cecd8/ch05/16_qwen3.5/qwen3_5_transformers.py`(**我还没有读这个文件**;已改用 HF transformers 的 qwen3_5 实现和 Prism 的 qwen35 图作公式参考,见 §12 的 TODO)。
7. Qwen3.8 基于 Qwen3.5 的架构:`https://huggingface.co/Qwen/Qwen3.8-27B`。
8. **"do not implement yet,先讨论"。**

### 1.2 第二轮追加

- 用 **MSL + C++,尽可能贴近 Metal,避免任何开销**。
- **极致针对这个特定模型和硬件,不为通用性牺牲速度。**
- 参考资料放在了项目目录里(上表)。
- "what do you think? answer in chinese"。

### 1.3 第三轮笔记(文件 `Apple Silicon 的 CPU 核心分三类:.md`,同时是第三条消息正文)

- CPU 核心分三类:E 核、P 核、Super 核(更强的超大核)。**要利用 CPU,和 GPU 并行做事**(tokenize 等放在 Super 核,避免抢带宽)。
- **用 Metal 4 的 command encoding(预编码)加速**,统一 command encoder,减少 CPU 开销。
- **针对这一个模型和这台机器写死,所有能提前决定的都提前做,不做动态分配。**
- `recommendedMaxWorkingSetSize`(12.1GB):在不影响性能的前提下**提前分配好最大内存,并验证、尝试调到最大**。
- 单个 buffer 上限 9.1GB:模型 6.4GB 放得下,但要注意分块,因为大小已知,提前分配。
- threadgroup memory 32KB:**提前计算,利用最大化**。
- cache line 128B:数据布局要填满 128B;28B 的三值块和 128B 不对齐是问题,**提前算好最佳对齐**。
- DRAM 带宽 153GB/s:"想尽一切办法减少内存数据搬运,压缩数据或把数据放进高速缓存"。
- 使用 KV cache。
- **用 CPU 和 ANE 并行地准备/预取/预计算,或者提前预计算好再加载**,因为模型是已知的。
- decode 阶段每 token 读取量的估算(用户贴了我之前的估算)。
- "可以把 draft mtp 放到 ANE 中,这样并行加速。"(**我否决了,理由见 §11**。)
- **使用 GPU 每个核心里新的神经加速器**:MPP / MetalPerformancePrimitives / MatMul2d。
- 显存预算:12.1GB 减 6.4GB 权重剩约 5.7GB;`sudo sysctl iogpu.wired_limit_mb` 可调,重启后失效。
- **量化 KV cache**,sub-4-bit KV 省读取带宽。
- **ANE:"Metal 无法调用 ANE,只能走 Core ML 或私有 API。这个是允许的。"**
- **单用户、只做文本、batch=1,不做视觉、continuous batching、paged KV。默认上下文 32K,最大 64K。**
- **使用场景:聊天和 agent。只做 CLI。**
- 寄存器双缓冲;persistent megakernel;**权重用 mmap + `MTLBuffer(bytesNoCopy)` + `MTLResidencySet` 常驻;零拷贝**。
- SSD prefix 缓存:把系统提示词对应的 GDN state 和 KV 存盘,下次直接加载。
- "解包可以放到 CPU 或 ANE 中吗?"(**回答:prefill 里放 CPU 可行;decode 里不行**,见 §10.8、§11。)
- **Hadamard 旋转/FWHT/sign flip 是 decode 里最大的非 matmul 开销,重点优化。**
- 零分配的 decode 循环、预分配到最大长度的 KV、跨多 token 摊销 CPU/GPU 同步、算子融合(借鉴 BaseRT)。
- **UMA 下不要为了用 threadgroup memory 而搬数据;对 `matmul2d` 的操作数直读 device memory。**
- **根据模型,直接使用硬件最原生支持的精度计算。**
- "tensor 的 scale plane 只支持 E8M0,放不下 fp16 group=128 的 scale,所以要在 epilogue 里按 K=128 分段累加。**double check 并重点优化。**"(**已核实,见 §5.7、§10.3、§10.8。**)
- **所有 shape 编译期写死,用一个 config 文件放在统一的地方。**
- Xcode license:**已经弄好了**。
- **使用 metal-cpp**,把控精细粒度,做到极致优化。
- **是否允许 prefill 的 int8 激活这一档?"优化速度为先,但不能降低模型质量、使模型变笨。"**(→ 默认走精确路径;int8 激活仅在质量门槛通过后启用。)

### 1.4 后续指令

- "metal-cpp 26 … this is the latest version of metal, right?"(→ 核实后发现不是,见 §14 更正日志。)
- "**always make sure to use the latest version of everything.**"
- "write what you have done so far as a very detailed long report and save as a md file as a checkpoint…"(→ 本文件。)

---

## 2. 对话与工作过程摘要(按时间)

| 轮次 | 用户 | 我做了什么 | 产出/结论 |
|---|---|---|---|
| 1 | 最初需求 | 探测硬件(`system_profiler`、Metal 设备属性);拉 Ollama manifest 与 GGUF 头(HTTP Range);解析 51 个 KV 和 866 个张量;读 Prism llama.cpp fork 的 `ggml-common.h`;读 Qwen3.8 config | 确认 M6 真实存在;确认量化格式 PTQ1_0 和 Hadamard 旋转;给出 roofline 和方案草图,提出一批待确认问题 |
| 2 | 加 MSL/C++、给 5 份 PDF 和 die shot | 用 PDFKit 提取 PDF 文本;精读白皮书、BaseRT 两篇论文、Feature Set Tables、MSL 4.1 规范、MPP 头文件 | 用论文数据校准预期(llama.cpp 基线在 M5 Pro 只用了约 65% 带宽、BaseRT 在 27B 上 prefill 只比 llama.cpp 快 27%);发现 `matmul2d` 支持 int8×int2b;发现 scale plane 只支持 E8M0 |
| 3 | 长笔记(§1.3) | 探测 metal-cpp、Metal 工具链、实测 `matmul2d` 吞吐、真实权重 scale 分析;12 点逐条回应 | **实测 M6 矩阵单元**(fp16 到 18.9 TFLOPS、int8 到 34 TOPS);scale 不可分解、无稀疏;否决 ANE 跑 MTP draft;提出 CPU 展开权重和 n-gram 草稿 |
| 4 | "metal-cpp 26 是最新吗" | 核实 SDK 27 头文件、探测 GPU family、探测 MSL 语言版本上限 | M6 是 **Apple11** family;MSL 4.1 是当前上限 |
| 5 | "永远用最新版本" | 全面版本审计;读 Prism/BaseRT 最新状态 | **发现 `apple/metal-cpp` GitHub main 已是 macOS 27.0.1**(纠正我之前的错误);Metal 离线编译器已被用户装好;Prism master 有大量相关提交 |
| 6 | 要 checkpoint | 补读 Prism 的 PTQ1_0 量化/反量化源码(**解决了 trit 排列顺序的未验证项**)、Prism 的 Metal 内核;确认 Prism 和 HF 对注意力输出门都用 sigmoid;复制探测与参考代码到 `docs/`;写本文件 | 本文件 |

---

## 3. 环境清单(2026-10-23 实测)

### 3.1 硬件(实测)

- Mac mini,`Model Identifier: Mac18,5`,`Model Number: MHQK4LL/A`,芯片 **Apple M6**。
- CPU:12 核 = **2 Super + 4 Performance + 6 Efficiency**(`hw.perflevel0/1/2.physicalcpu` = 2/4/6)。
- 内存 16GB(`hw.memsize` = 17,179,869,184)。
- GPU:**12 核**,Metal 4。**`supportsFamily` 对 apple9、apple10、apple11、metal4 全部为 true;M6 属于 Apple11 family**(早期我只测了 apple10,是我的疏漏,见 §14)。
- Metal 设备属性:`recommendedMaxWorkingSetSize` = **12,124 MB**;`maxBufferLength` = **9,093 MB**;最大 threadgroup memory = **32 KB**;每 threadgroup 最多 1024 线程;`hasUnifiedMemory` = true。
- `sysctl`:`hw.cachelinesize` = 128;`hw.perflevel0.l2cachesize` = 20,971,520;`hw.l2cachesize` = 8,388,608;`iogpu.wired_limit_mb` = 0(默认)。
- 磁盘:`/` 共 228GiB,剩余约 97GiB。
- Die shot(`Apple M6 vs M5 Chip Deep-Dive.png`,来自 SemiAnalysis):M6 有 12 个 GPU 核、**4 块各 4MB 的 SLC(共 16MB)**、2 个各 10MB 的 "S- & P-core L2"(共 20MB)、1 个 8MB 的 E-core L2、**2 个 ANE 簇(每簇标注 8 核,共 16 核)**、"S-Core AMX / E-core AMX"(CPU 侧矩阵协处理器)、8 个 16-bit LPDDR5X PHY(= 128-bit 总线)。用户给的 DRAM 带宽 **153GB/s 是理论值,我还没有实测**。

### 3.2 软件

| 项 | 版本 | 状态 |
|---|---|---|
| macOS | 27.0.1(`sw_vers` Build 26A434) | `softwareupdate -l`:No new software available |
| Xcode | 27.1(27A9275),路径 `/Applications/Xcode.app`;`xcode-select -p` 现在指向 Xcode | license 已由用户接受 |
| SDK | MacOSX27.0.sdk(另有 26.5、26、27) | 最新 |
| clang | Apple clang 21.0.0 | |
| Swift | 6.4 | |
| **Metal 编译器** | **`Apple metal version 32023.921 (metalfe-32023.921.6)`**,target `air64-apple-darwin27.0.0`;`metallib` = AIR-LLD 32023.921 | **用户已装好**(之前缺失:`xcodebuild -downloadComponent MetalToolchain` 需要先 `-runFirstLaunch`) |
| Metal 语言版本 | **4.1 是上限**:离线编译器 `-std=metal4.2`/`metal5.0` 报 invalid value;运行时编译器(`MTL::CompileOptions`)对 4.2、4.3、5.0、5.1 同样拒绝 | 与 MSL 规范 PDF 4.1(2026-06-04)一致 |
| Homebrew | 7.0.10 | 已安装 |
| cmake / ninja | **未安装** | 只有编译 Prism llama.cpp 基线时才需要 cmake;我们自己的项目计划用纯 `Makefile` |
| git / make | 有 | |

**离线编译命令(已验证对一个平凡 kernel 有效;含 MPP 的 kernel 尚未离线验证)**:

```sh
export DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer
xcrun metal -std=metal4.1 -c t.metal -o t.air && xcrun metallib t.air -o t.metallib
```

运行时编译(探测程序用的):`MTL::CompileOptions::setLanguageVersion(MTL::LanguageVersion4_1)`(新版 metal-cpp 有该枚举);`#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>` 在运行时编译器里可用。

### 3.3 上游最新状态(2026-10-23 核对)

| 组件 | 最新 | 备注 |
|---|---|---|
| **metal-cpp** | **GitHub `apple/metal-cpp` main,commit `8cdde68ae36a636c84cbb48ebcbc8f3b60251ea0`(2026-09-22,"metal-cpp for macOS27.0.1, iOS27.0.1")** | 带 `LanguageVersion4_1` 和 `GPUFamilyApple11`;冒烟测试通过。**官网 `metal-cpp_26.zip`(2026-01-30)已过时,不要用。** `MTL4BufferRange`(仅光线追踪加速结构用)、`MTLDeviceCertification` 暂无绑定,与我们无关 |
| Apple 文档 | MSL 规范 4.1、Feature Set Tables(2026-05-21) | 官网字节数与项目里的 PDF 完全一致。Feature Set Tables **不含 Apple11**,所以 M6 的能力要靠实测 |
| MPP Programming Guide | BaseRT 论文第二篇引用过(Version 1, March 2026) | 我猜的 URL 全部 404,**没找到**。如果用户有,请放进项目目录 |
| Prism llama.cpp | branch `prism` head `73bfe6cc485554822f80c473a3c0c4f25e80f41b`(2026-10-19);仓库 2026-10-23 仍有 push | 见 §8.2 |
| BaseRT | 仓库 `basecompute/baseRT` main `1bbd81e2af183e02e9c1ad7cbd9fe131e02520c1`(2026-10-23),**v0.4.1(今天发布)**;论文里的 v0.1.6 已过时 | 仓库是 CLI、`.base` 格式、C API、绑定;README 称引擎是**预编译安装**的,我没找到 kernel 源码。发布说明提到支持 M1/M3/M4/M5/M6、Qwen 3.8 |
| 模型 | Ollama `mtp-ptq1_0`:6.4GB,标签 digest `2eaf16c78c39`,一周前更新;权重 blob sha256 `174c9c65…`,6,397,969,920 字节 | 当前版本。HF `prism-ml/Ternary-Bonsai-2-27B-gguf` sha `b072e1d3b35a0a630cece372c2127528e0994386`,最后修改 2026-09-25 |
| Qwen3.8-27B | HF sha `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`,最后修改 2026-08-14 | config 即当前 |
| Prism 新模型 | Ternary-Bonsai-2 的 2B/4B/9B(gguf 和 mlx-2bit),2026-10-14/15 | 27B 没有更新版本。小模型可能当外部 drafter(未评估,tokenizer/架构兼容性未验证) |
| arXiv | 两篇 BaseRT 论文都只有 v1 | |

---

## 4. 项目目录当前状态

```
projects/kata/
├── .git/                      # 无任何 commit
├── .gitkeep
├── 实现Ternary Bonsai-2-27B (Qwen 3.8 27B) in.md     # 用户的原始需求
├── Apple Silicon 的 CPU 核心分三类:.md                 # 用户的第二批笔记
├── Apple M6 vs M5 Chip Deep-Dive.png
├── qwen3.5.webp
├── BaseRT-base.pdf  baseRT.pdf  bonsai-2-27b-whitepaper.pdf
├── Metal-Feature-Set-Tables.pdf  Metal-Shading-Language-Specification.pdf
└── docs/
    ├── CHECKPOINT-001.md      # 本文件
    ├── data/gguf_header_dump.txt
    ├── probes/*.cpp *.swift
    └── reference/*            # Prism / HF 的关键参考代码(带来源和 commit)
```

**尚未创建**:`src/`、`config/`、`third_party/`、`Makefile`、`VERSIONS`。

---

## 5. 目标模型的全部事实

### 5.1 来源与完整性

- Ollama 仓库:`mannix/Bonsai-2-27B`,标签 **`mtp-ptq1_0`**(6.4GB,text only,256K context window)。同仓库还有 `latest`(7.0GB,含视觉)、`pq2_0`(7.2GB)、`ptq1_0`(5.9GB,无 MTP)、各种 `vision`/`lvision`/`mtp-*` 变体,全部"1 week ago"。
- Manifest 请求:
  ```sh
  curl -sL -H 'Accept: application/vnd.docker.distribution.manifest.v2+json' \
    https://registry.ollama.ai/v2/mannix/Bonsai-2-27B/manifests/mtp-ptq1_0
  ```
  返回:`runner: llamacpp`,`format: gguf`;层:
  - config blob `sha256:a3f0e513e47b4526eaa4fef3a4367f7baf7c1960f1cd34a216546485e839ad07`(158 B)
  - **模型层** `sha256:174c9c6584e118766d5511e00a84fd8a31d146248915a726cfac8e6d140c6204`,**6,397,969,920 B**,`from: Ternary-Bonsai-2-27B-PTQ1_0-MTP.gguf`
  - license blobs `sha256:4bf96504…`(10,139 B)、`sha256:17e85d0a…`(410 B)
  - params blob `sha256:e7a4ccdae39a06680bcd804c07db5a7d6b155972642548bce8601abb5d3ca16f`(388 B):`{"min_p":0.05,"presence_penalty":0,"repeat_penalty":1,"temperature":1,"think_budget":"high","think_budget_message":"…","top_k":20,"top_p":0.95}`
- Blob 下载(支持 HTTP Range,探测时用的就是这个):
  `https://registry.ollama.ai/v2/mannix/Bonsai-2-27B/blobs/sha256:174c9c6584e118766d5511e00a84fd8a31d146248915a726cfac8e6d140c6204`
- 许可:Apache-2.0;"Created using Kata by Prism ML"(如果公开分发需署名);基于 Qwen3.8-27B(Apache-2.0)。
- 这个 GGUF 是第三方(`opencoti`/mannix)把 Prism 的 `Ternary-Bonsai-2-27B-PTQ1_0.gguf` 加上从 Qwen3.8-27B 的 `model-00018-of-00018.safetensors`(sha256 `1d3479509e21494658f9b64d317f5ea8e55c4025d28c702d6c4d0b356ce8ea06`)取出的 `mtp.*` 层(Q8_0)合成的。KV `opencoti.mtp.source` 记录了这一点。
- Prism 官方 HF 仓库 `prism-ml/Ternary-Bonsai-2-27B-gguf` 有:`F16`、`PQ2_0`(7.25GB)、`PTQ1_0`(5.93GB,无 MTP)、两个 mmproj、`KNOWN_ISSUES.md`(**还没读,应该读**)。

### 5.2 GGUF 文件结构(已验证)

- GGUF v3,小端;`tensor_count = 866`(0x362),`kv_count = 51`。
- 头部布局:魔数 `GGUF`(4B)、版本 u32、张量数 u64、KV 数 u64;每个 KV = 字符串 key(u64 长度 + 字节)、u32 类型、值(数组 = u32 元素类型 + u64 个数 + 元素);然后每个张量信息 = 名称字符串、u32 维数、u64 dims[]、u32 类型、u64 偏移(**相对数据区起点**)。
- **数据区起点 = 11,122,176**(32 字节对齐;用它按偏移取张量数据并解码,得到合理的 scale 与三值分布,说明正确)。
- **数据区总长 = 6,397,969,920 − 11,122,176 = 6,386,847,744 字节**。(我早期把整个文件长度减张量偏移得到 MTP 块 462MB,把头部也算进去了,**正确的 MTP 块是 451,319,808 B**,见 §14。)
- 张量类型计数:type 0(F32)360 个共 2,671,616 元素;type 8(Q8_0)8 个共 424,673,280 元素;type 30(BF16)96 个共 23,592,960 元素;**type 143(PTQ1_0)402 个共 26,869,760,000 元素**。

### 5.3 元数据(全部重要 KV)

```
general.architecture = qwen35        general.file_type = 143   general.quantization_version = 2
general.name = "Hf"  basename = "folded"  version = "v5"  size_label = 27B
general.sampling.top_k = 20  top_p = 0.95  temp = 1.0
qwen35.block_count = 65                         # 64 主干 + 1 MTP
qwen35.context_length = 262144
qwen35.embedding_length = 5120
qwen35.feed_forward_length = 17408
qwen35.attention.head_count = 24   head_count_kv = 4
qwen35.attention.key_length = 256  value_length = 256
qwen35.attention.layer_norm_rms_epsilon = 1e-06
qwen35.full_attention_interval = 4
qwen35.rope.freq_base = 10000000.0
qwen35.rope.dimension_count = 64                # partial rotary 25% of head_dim 256
qwen35.rope.dimension_sections = [11, 11, 10, 0]
qwen35.ssm.conv_kernel = 4   state_size = 128   group_count = 16   time_step_rank = 48   inner_size = 6144
qwen35.nextn_predict_layers = 1
prism.hadamard.version = 1
prism.hadamard.block_size = 1024
prism.hadamard.transform = normalized-sylvester-walsh-hadamard
prism.hadamard.axis = input-last-dimension
prism.hadamard.sign_mode = explicit
prism.hadamard.weight_names = [401 项: output.weight, blk.N.attn_qkv.weight, blk.N.attn_gate.weight, blk.N.ssm_out.weight, blk.N.ffn_*.weight, ...]
prism.hadamard.sign_widths = [5120, 6144, 17408]
prism.hadamard.sign_values = [28672 个 ±1]      # 5120+6144+17408 = 28672
prism.hadamard.inverse_weight_names = [token_embd.weight]
prism.hadamard.gdn_v_grouped = true
tokenizer.ggml.model = gpt2   tokenizer.ggml.pre = qwen35
tokenizer.ggml.tokens = [248320]   token_type = [248320]   merges = [247587]
tokenizer.ggml.bos_token_id = 248044   padding_token_id = 248044   eos_token_id = 248046   add_bos_token = false
tokenizer.chat_template = (Jinja,内含)
opencoti.mtp.source = "Qwen/Qwen3.8-27B mtp.* (model-00018-of-00018.safetensors sha256 1d34…), convert_hf_to_gguf --mtp ..."
```

HF `config.json`(Qwen3.8-27B)里另有:`eos_token_id` 248044、`image_token_id` 248056、`video_token_id` 248057、`vision_start/end` 248053/248054、`mamba_ssm_dtype: float32`、`mtp_num_hidden_layers: 1`、`mtp_use_dedicated_embeddings: false`、`tie_word_embeddings: false`、`attn_output_gate: true`、`output_gate_type: "swish"`(**见 §12 风险 R1**)、`rope_parameters.mrope_interleaved: true`、`mrope_section [11,11,10]`、`partial_rotary_factor 0.25`、`rope_theta 1e7`、`layer_types`(64 项,每第 4 项为 `full_attention`)、视觉塔(27 层,hidden 1152,out_hidden 5120;**我们不做视觉**)。

### 5.4 架构(由 config 和张量推出)

- 64 个主干层:**48 个 Gated DeltaNet(GDN,linear attention)+ 16 个 full attention**,full attention 在 blk 3、7、11、…、63(每 4 层的第 4 层)。`blk.64` 是 MTP 层。
- 每层后面都有同一个稠密 SwiGLU FFN(5120→17408→5120,三个矩阵)。
- full attention:`attn_q` 的输出维度 12288 = 24 头 × 256 × 2(**query 和输出 gate 在一个投影里**);`attn_k`/`attn_v` 输出 1024 = 4 头 × 256;`q_norm`/`k_norm` 是 256 维 RMSNorm;`attn_output` 输入 6144 = 24 × 256;GQA 比例 6:1;注意力输出乘 `sigmoid(gate)`(**Prism 和 HF 都是 sigmoid,config 里写 swish,见 R1**);partial RoPE 只作用于前 64 维。
- GDN:`attn_qkv` 输出 10240 = Q 2048(16 头 × 128)+ K 2048 + V 6144(48 头 × 128);`attn_gate`(即 z,输出 6144);`ssm_alpha`/`ssm_beta`(5120→48,**bf16,不旋转**);`ssm_conv1d`(4 × 10240,f32);`ssm_a`(48,看起来已预变换为 `A=-exp(A_log)`:Prism 的 Metal kernel 直接用 `g = a[h] * softplus(g_raw + dt_bias[h])`,`β = sigmoid(raw)`,衰减 `exp(g)`;**精确含义以源码为准**);`ssm_dt.bias`(48);`ssm_norm`(128);`ssm_out`(6144→5120)。每个 V 头有一个 128×128 的 fp32 状态;K 头(16)按 3 倍重复到 48 个 V 头。
- Raschka 架构图(`qwen3.5.webp`)给出的 GDN 结构:Q、K 经 conv 后做 L2 Norm;V 经 conv 加 SiLU;α 经 softplus 门、β 经 sigmoid;"Gated Delta Rule";RMSNorm;再乘 SiLU(z) 门;Linear 输出。**精确公式要以 HF 的 `modeling_qwen3_5.py` 和 Prism 的 `qwen35.cpp` 为准。**
- MTP 层(`blk.64`,Q8_0):`nextn.eh_proj`(10240→5120,拼接 embedding 与 hidden 后投影)、`nextn.enorm`、`nextn.hnorm`、`nextn.shared_head_norm`、一个完整的 full-attention 子层(q/k/v/o、q_norm/k_norm)、一个完整的 FFN。**没有单独的 lm_head,复用 `output.weight`。**

### 5.5 张量清单(代表项,完整见 `docs/data/gguf_header_dump.txt`)

| 张量 | 形状 [in, out] | 类型 | 说明 |
|---|---|---|---|
| `output.weight` | [5120, 248320] | PTQ1_0 | lm_head,偏移 0,278,118,400 B |
| `output_norm.weight` | [5120] | F32 | |
| `token_embd.weight` | [5120, 248320] | PTQ1_0 | **旋转后的**嵌入,查表后需反向 FWHT;偏移 278,138,880,278,118,400 B |
| GDN `blk.N.attn_qkv.weight` | [5120, 10240] | PTQ1_0 | |
| GDN `blk.N.attn_gate.weight` | [5120, 6144] | PTQ1_0 | z 门 |
| GDN `blk.N.ssm_out.weight` | [6144, 5120] | PTQ1_0 | 输入需 6144 宽的 FWHT |
| GDN `blk.N.ssm_alpha/beta.weight` | [5120, 48] | BF16 | 不旋转 |
| GDN `blk.N.ssm_conv1d.weight` | [4, 10240] | F32 | |
| GDN `blk.N.ssm_a` / `ssm_dt.bias` | [48] | F32 | |
| GDN `blk.N.ssm_norm.weight` | [128] | F32 | |
| attn `blk.N.attn_q.weight` | [5120, 12288] | PTQ1_0 | query + gate |
| attn `blk.N.attn_k/v.weight` | [5120, 1024] | PTQ1_0 | |
| attn `blk.N.attn_output.weight` | [6144, 5120] | PTQ1_0 | |
| attn `blk.N.attn_q_norm/k_norm.weight` | [256] | F32 | |
| 每层 `ffn_gate/up.weight` | [5120, 17408] | PTQ1_0 | |
| 每层 `ffn_down.weight` | [17408, 5120] | PTQ1_0 | 输入需 17408 宽的 FWHT(17 个 1024 块) |
| 每层 `attn_norm`/`post_attention_norm` | [5120] | F32 | |
| `blk.64.*`(MTP) | 同上形状 | **Q8_0** | 共 8 个 Q8_0 张量,451,319,808 B |

401 个被旋转的张量 = 400 个层内矩阵(GDN 层 6 个 × 48 = 288,attention 层 7 个 × 16 = 112)加上 `output.weight`;`token_embd` 另列为**反向**旋转(PTQ1_0 类型共 402 个 = 401 + `token_embd`)。

### 5.6 PTQ1_0 格式(**权威定义,已读 Prism 源码确认**)

来源:`docs/reference/prism_ptq1_0_quant_dequant.c`(Prism `ggml/src/ggml-quants.c` 第 2203–2300 行)和 `ggml-common.h`。

```c
#define QK_PTQ1_0 128
typedef struct {
    uint8_t qs[24];   // 5 trits per byte -> 120 values
    uint8_t qh[2];    // 4 trits per byte ->   8 values
    ggml_half d;      // fp16 scale        (放在块的最后,字节 26-27)
} block_ptq1_0;       // 28 bytes per 128 weights = 1.75 bpw
```

- 每个字节是 **256 为分母的 base-3 小数**:编码时 `q = Σ digit·3^k`(5 位三进制数,digit = 值+1,即 -1/0/+1 → 0/1/2),然后 `q = ceil(q × 256 / 243)`;解码第 n 个 trit:`xi = ((uint16)(uint8)(byte × pow3[n]) × 3) >> 8`,值 = `xi − 1`,`pow3 = {1,3,9,27,81,243}`,n=0 是**最高位 trit**。
- 一个 128 元素块里的元素排列(`ptq1_0_stages = {32,16,8}`):
  1. `qs[0..15]`(16 字节)→ **元素 0..79**:元素索引 `e = n×16 + m`(n=0..4,m=0..15),即 `qs[m]` 的第 n 个 trit。
  2. `qs[16..23]`(8 字节)→ **元素 80..119**:`e = 80 + n×8 + m`(n=0..4,m=0..7),即 `qs[16+m]` 的第 n 个 trit。
  3. `qh[0..1]`(2 字节)→ **元素 120..127**:`e = 120 + n×2 + h`(n=0..3,h=0..1),即 `qh[h]` 的第 n 个 trit。`qh` 编码时是 4 个 trit 再乘 3(等价于末位补 0 的 5 位数),解码方式同上。
- 反量化值 = `(xi − 1) × d`。量化器的 `d = 该组 |x| 最大值`(对已经是三值的 checkpoint 则无损,`d` 即该组 scale)。
- 一行(一个输出神经元)的所有块**连续存放**,`K/128` 个块;行与行连续。
- 白皮书称 PTQ1_0 为 1.76 bpw/5.93GB(含非三值的少量张量);纯三值是 1.72 bpw。

### 5.7 对真实权重的数据分析(`docs/probes/scale_analysis.cpp`,取自 `blk.3.attn_k` 和 `blk.0.ffn_down`)

| 指标 | `attn_k` [K=5120, N=1024] | `ffn_down` [K=17408, N=5120] |
|---|---|---|
| trit 分布 (-1 / 0 / +1) | 0.336 / 0.328 / 0.336 | 0.336 / 0.328 / 0.336 |
| scale 最小 / p1 / 中位 / p99 / 最大 | 0.00848 / 0.01059 / 0.017365 / 0.023621 / 0.035614 | 0.007267 / 0.009018 / 0.011108 / 0.014496 / 0.029037 |
| 不同的 fp16 scale 值个数 | 1578 / 40960 | 1713 / 696320 |
| 非正 scale | 0 | 0 |
| 行内 max/min scale 比(均值 / 最大) | 1.339 / 1.665 | 1.542 / 3.282 |
| 对 `log scale` 做 **行因子×组因子** rank-1 拟合:rms / 平均相对误差 | 0.0658 / 5.25% | 0.0898 / 6.97% |
| 仅按行拟合:rms | 0.0667 | 0.0898 |
| 与 2 的幂相差 <1.4% 的 scale 占比 | 4.5% | 0.1% |

**结论(用户要求的 double check):**
1. **三值是均匀的 1/3,没有稀疏性**,跳零毫无意义。
2. scale **不是 2 的幂**,Metal 4.1 tensor 的 block-scale plane(只支持 `metal_fp8_ue8m0_format`、第一维块大小固定 32)**放不下**。
3. scale **不能分解**成行因子×组因子(拟合误差 5% 到 7%,有损),也不能折叠进激活。
4. 所以**逐 K=128 组的 scale 处理是精确实现的必要条件**。唯一的免 epilogue 办法是把权重展开成 fp16(`w = s·t`,fp16 乘 ±1/0 是精确的),代价是权重字节翻倍到 2B/权重——这只能用在 prefill(见 §10.8 路径 A)。
5. 不同 fp16 scale 值很多只是因为 fp16 在这个动态范围里的格点就有约 2000 个,**不是码本**。

### 5.8 字节账(精确)

(数据区 6,386,847,744 B)

| 项 | 字节 | GB |
|---|---|---|
| `output.weight`(lm_head) | 278,118,400 | 0.278 |
| `token_embd.weight` | 278,118,400 | 0.278 |
| 主干 64 层(含少量 bf16/f32 小张量) | 5,379,291,136 | 5.379 |
| MTP 块 `blk.64`(Q8_0) | 451,319,808 | 0.451 |
| **decode 每 token 需读**(主干 + lm_head) | 5,657,409,536 | **5.657** |
| 每个 MTP draft 需读(MTP 块 + lm_head) | 729,438,208 | 0.729 |

- GDN 层约 84.9MB(三值 83.7MB + bf16 alpha/beta 0.98MB + 小张量);attention 层约 81.5MB。48 个 GDN 层约 4.07GB,16 个 attention 层约 1.30GB。
- 每层权重元素数:GDN 层约 382.7M(qkv 52.4M + z 31.5M + out 31.5M + FFN 267.4M),attention 层约 372.2M(q 62.9M + k/v 各 5.2M + o 31.5M + FFN 267.4M)。**prefill 每 token 计算量 = 2 × (48×382.7M + 16×372.2M) = 2 × 24.33G MACs ≈ 48.7 GFLOP**(lm_head 只对最后一个 token 算)。
- **GDN 状态**:每层 48 头 × 128 × 128 × 4B = 3,145,728 B,48 层共 150,994,944 B(151MB);conv 状态(核 4,3 个历史输入 × 10240 通道 × f32)约 123KB/层,共 5.9MB。
- **KV(fp16)**:每 token 16 层 × K、V × 4 头 × 256 × 2B = **65,536 B/token**;int8 加每 token 每头 scale 约 33,024 B/token。32K 上下文 fp16 = 2.15GB;64K int8 ≈ 2.16GB。MTP 层自己的 KV 是 4,096 B/token(fp16)。
- 每 token 的 embedding 行(三值)约 1.1KB。

### 5.9 分词器与采样

- GPT-2 风格 byte-level BPE,`pre = qwen35`,词表 248,320,merges 247,587,`add_bos_token = false`,bos/pad = 248044,GGUF 里 eos = 248046(HF config 里 eos = 248044)。
- Chat template 在 GGUF 里(Jinja)。计划用 C++ 手写这一个模板,不引入 Jinja 引擎。Qwen 模板会把**历史轮次的 thinking 内容剥掉**(prefix 缓存的快照点要选在最后一个 user 消息的边界)——**这一点我是凭对 Qwen 模板的了解,还没有对着模板核实**,TODO。
- 默认采样:temperature 1.0、top_k 20、top_p 0.95、min_p 0.05、presence_penalty 0、repeat_penalty 1;`think_budget: high`。

---

## 6. 硬件探测与实测结果

### 6.1 内存层级(已知)

| 层级 | 已知事实 | 来源 |
|---|---|---|
| GPU 寄存器/L1/L2 | **未知**,Apple 不公开 | — |
| Threadgroup memory | 32KB(每 threadgroup 最大),16B 对齐 | Metal 设备属性、Feature Tables |
| Cache line | 128B | `sysctl hw.cachelinesize` |
| CPU L2 | Super+P 共享 2×10MB(sysctl `perflevel0` 报 20MB);E 核 8MB | die shot、sysctl |
| SLC | 16MB(4×4MB),全芯片共享(CPU/GPU/ANE) | die shot |
| DRAM | 16GB LPDDR5X,128-bit,理论 **153.6GB/s**(**未实测**) | 用户给出,与 die shot 的 8×16-bit PHY 吻合 |
| 可用 GPU 工作集 | 12,124 MB(默认);`sudo sysctl iogpu.wired_limit_mb=N` 可调高,重启失效 | Metal 设备属性 |

### 6.2 `matmul2d` 实测(探测程序 `docs/probes/matmul2d_probe.cpp`)

- 方法:运行时编译 MSL(Metal 4.1),`#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>`;`tensor_inline` 从 device 裸指针构造(`tensor<device T, dextents<int32_t,2>, tensor_inline> tA(A, dextents<int32_t,2>(K, M))`,第 0 维是最快变化维);`matmul2d_descriptor(TM, TN, dynamic_extent)`,`matmul2d<desc, execution_simdgroups<SG>>`;A、B、C 都从 device 读/写;dispatch 网格 `(N/TN, M/TM)`,`tgid.x`→N,`tgid.y`→M,每个 threadgroup `32×SG` 线程;`M=N=K=4096`;计时用 `commandBuffer.GPUEndTime − GPUStartTime`,取 8 次中去掉前 2 次后的最小值。
- **注意事项**:输入用 `memset(0x11)` 的常数填充,可能影响功耗/频率行为,**需要用随机数据复测**;`tensor_inline` 可能比 `tensor_handle` 慢;**还没有试 K 方向手动分块、threadgroup 预取、relaxed_precision、真实形状(M=512~2048 token × N=17408 × K=5120)**。这些都是 M0 的工作。

**原始结果**(单位:TFLOPS 或 TOPS;`TM×TN` = 每 threadgroup 输出 tile;SG = 每 threadgroup 的 simdgroup 数):

| 类型 (A × B → C) | tile / SG | 吞吐 |
|---|---|---|
| half × half → float | 64×32 / 4 | 5.57 |
| | 128×64 / 4 | 9.73 |
| | 64×64 / 4 | 10.68(另一轮 10.73) |
| | 128×128 / 4 | 6.47 |
| | 64×128 / 4 | 15.87 |
| | 32×128 / 4 | 16.69 |
| | 128×32 / 4 | 4.46 |
| | 64×96 / 4 | 15.95 |
| | 64×64 / 2 | 14.36 |
| | 64×128 / 8 | 11.36 |
| | 128×128 / 8 | 10.45 |
| | **32×64 / 1** | **18.94** |
| half × half → half | 64×64 / 4 | 11.36 |
| bfloat × bfloat → float | 64×64 / 4 | 10.27 |
| half × int8_t → float | 64×64 / 4 | 11.97 |
| half × int8_t → half | 64×64 / 4 | 12.37 |
| half × uchar → float | 64×64 / 4 | 11.74 |
| bfloat × int8_t → float | 64×64 / 4 | 11.66 |
| **int8_t × int8_t → int** | 64×64 / 4 | 22.13 |
| | 128×64 / 4 | 20.17 |
| | 64×256 / 4 | 18.48 |
| | 32×128 / 4 | 28.51 |
| | 64×128 / 8 | 22.99 |
| | 64×128 / 2 | 16.54 |
| | **64×128 / 4** | **33.87** |
| | 128×128 / 8 | 20.61 |
| uchar × uchar → int | 64×64 / 4 | 21.57 |
| | 128×64 / 4 | 20.85 |
| | 64×128 / 4 | 34.05 |
| | 128×128 / 4 | 15.12 |

**解读:**
- **int8×int8→int32 比 fp16 快约 1.8 倍**(34 对 19)。`half × int8` 和 fp16 同速(硬件内部转 half),所以"精确的 half 激活 × int8 权重"路径和 fp16 路径速率相同。
- **tile 形状和 SG 数影响极大**(fp16:4.5 到 18.9;int8:15 到 34),M0 必须系统地扫。
- 类型注意:**有符号 8 位必须写 `int8_t`,写 `char` 会 static_assert 失败**。
- 这些是**未调优的数字**;规范/头文件列的组合里,`char × char → int` 等写成 `char` 的组合在本机编译器上不可用,`int8_t` 版本可用。
- 由此推算 prefill 的 matmul 上限:48.7 GFLOP/token,fp16 路径 18.9 TFLOPS → **约 388 tok/s**;int8 路径 34 TOPS → **约 698 tok/s**(只算 matmul;实际按 60%~70% 估约 250 和 450,**这是推算**)。作参照:Prism 白皮书中 M5 Pro 上 llama.cpp 的 pp512 为 397 tok/s。

### 6.3 其它 MPP 事实(规范/头文件)

- `matmul2d` 支持的类型组合(MSL 4.1 规范第 17.2.1 节,约第 351–354 页):`half/bfloat/float` 彼此组合;`int8_t/uint8_t` 与 half/bfloat/float 组合;`int8×int8→int32`、`uint8×uint8→int32`;**Metal 4.1 新增:`int8×int2b_format→int`、`half/bfloat×int2b_format→half/bfloat/float`,以及 int4b、fp4(e2m1)、fp8(e4m3/e5m2)格式**。
- 格式 tensor 的限制(Feature Set Tables 第 17 页):`Int2/UInt2` 行大小对齐 32 元素,行跨度对齐 512 元素,切片 4 元素,buffer 偏移 128B,支持 block scaling。
- **block-scale plane 只支持 `metal_fp8_ue8m0_format`,块的第一维固定为 32**(规范 2.22.2.2)。
- Metal 4.1 新增:acquire/release 内存序、packed numeric、multiplane tensor、`interleave/deinterleave`(整数函数)。
- MSL 里**没有 async-copy/prefetch**;simdgroup 宽度 32;`simdgroup_matrix` 从 Apple7 起可用。
- BaseRT 第二篇论文的结论:操作数直接从 device memory 读 `matmul2d` 最快,不必经 threadgroup memory 中转。

### 6.4 `int2b_format`/`int4b`/fp8 tensor 探测(`docs/probes/matmul2d_int2b_probe.cpp`)

- 类型本身存在,`#include <metal_packed_numeric>` 后可用。
- **但 `tensor<device int2b_format, dextents<int32_t,2>, tensor_inline>(devicePtr, extents)` 编译失败(no matching constructor)**。格式 tensor 大概必须通过 `MTLTensor` + Metal 4 argument table 绑定,或用别的构造器(**未查明**)。
- 因此 int2b/fp8 的吞吐**还没有测**,是 M0 的一项。

---

## 7. 参考资料摘要

### 7.1 Prism 白皮书(`bonsai-2-27b-whitepaper.pdf`,2026-09)

- 架构同 Qwen3.8-27B:约 75% 线性注意力、25% full attention。
- 权重:三值 {-1,0,+1} + fp16 group-128 scale,**固定旋转基**(1024 块 Hadamard,固定 ±1 符号)。有效 1.76 bpw/5.93GB(PTQ1_0),PQ2_0 为 2.16 bpw/7.25GB。**白皮书称 PTQ1_0 在 Apple silicon 上不比 PQ2_0 快**(解包更费算力)。
- 保持高精度、不旋转、不量化的张量:`in_proj_a/b`(即 `ssm_alpha/beta`)、`conv1d`、`A_log/dt_bias/norm.weight`、所有 RMSNorm、`q_norm/k_norm`,共 26.2M 参数(占 0.0976%)。
- 激活侧 FWHT:"sign flip + 分块 FWHT"在 batch=1 decode 里是**最大的非 matmul 开销之一**;Metal 上把 sign flip 融进 FWHT 的加载路径。
- 基准(llama.cpp Metal,PQ2_0):M5 Pro tg128 **27.7**、pp512 **397** tok/s(约 201GB/s 权重流量);M5 Max 46.8 / 765;M4 Pro 18.0 / 125(旧版未旋转构建)。PTQ1_0 在 M5 Pro 为 27.1 / 369。
- 评测:thinking 模式、20 项基准、平均 83.9(FP16 基线 85.4,保留 98.2%);输出预算最高 81,920 token——**所以实际使用里 decode 的上下文会很长,KV 读取量不能忽略**。KV 压缩(sub-4-bit)早期结果有希望。
- 路线图:让 PTQ1_0 在所有平台上和 PQ2_0 一样快;降低旋转开销。

### 7.2 BaseRT 第一篇(M3/M4,arXiv 2607.00501v1)

- 纯 C++ + Metal,无框架。设计:数据驱动的架构描述符(我们**不要**,我们要写死)、**零分配 decode 循环**、预分配到最大长度的 KV、大量手写融合 kernel、per-chip 配置、tiled GEMM(simdgroup matrix)、FlashAttention。
- 结果:在 ≥3B 的模型上 decode 只比 llama.cpp 快 1.04–1.18 倍(带宽受限);小模型最高 1.56 倍。**稠密模型 prefill 反而比 llama.cpp 慢 5%~10%**。uzu(用 MPSGraph、可能走 ANE)的 prefill 只高 1%~13%。

### 7.3 BaseRT 第二篇(M5 Neural Accelerator,arXiv 2607.19438v1)

- M5 每个 GPU 核有一个 Neural Accelerator,通过 Metal 4 tensor API/`matmul2d` 使用,**只用于 prefill**(decode 带宽受限,加速器没用)。M5 对外仍报告 `apple9`,不能靠 family 判断。
- M5 Pro 上 **Qwen3.6-27B Q4**(与我们同架构族):decode **18.1**(llama.cpp 15.4,MLX 18.4)tok/s;prefill pp128–2048 为 440–498 tok/s(llama.cpp 358–392)。**最好的原生实现在 27B 稠密上 prefill 只比 llama.cpp 快 27%。**
- 由此校准:27B Q4 的 18.1 tok/s × 约 15.3GB ≈ 277GB/s,约占 M5 Pro 峰值(**我凭印象写 307GB/s,未核实**)的 90%。所以**调优良好的 kernel 能到约 90% 带宽**,而 Prism 的 llama.cpp 在 M5 Pro 上只到约 65%。
- 未来工作里提到:"投机解码把一部分 decode 变成批验证,可用 tensor core"——正是我们的 MTP 路线。

### 7.4 Feature Set Tables(2026-05-21)

- 到 Apple10(M5)为止,**没有 Apple11(M6)**。关键:Apple7 起有 SIMD 归约和 simdgroup matrix;Apple9 起 64-bit atomics;浮点 atomics 从 Apple7;最大 threadgroup 线程数 1024,threadgroup memory 32KB;ICB(compute)从 Apple3;residency set 从 Apple6;argument tables/command allocators/tensors/command barriers 等 Metal 4 特性从 Apple7。
- 第 17 页的 tensor 数据类型限制表(见 §6.3)。

### 7.5 MSL 4.1 规范(383 页)

- 要点位置:2.21 Packed Numeric Type(第 79 页)、2.22 Tensors(第 85–103 页,`tensor_blockwise` 第 88–90 页)、6.4 整数函数(`interleave/deinterleave`)、6.10/6.16 barrier 与 atomic 内存序(第 299–301 页)、17.2.1 MPP 矩阵乘数据类型(第 351–354 页)。

### 7.6 其它

- `qwen3.5.webp`:Raschka 的 Qwen3.5 架构图,3:1 线性/full 层比;"Gated DeltaNet modules don't grow the KV cache size"。
- Prism 的 Metal PTQ1_0 内核里有一段很有价值的注释(`docs/reference/prism_metal_mul_mv_ptq1_0.metal`):
  - 早期映射让每个线程取 16 个连续权重,在 base-3 布局下相当于**同样的 24 个 qs 字节被重复读 5 次**;现在让每个线程**拥有整字节**并消耗该字节的全部 5 个 trit,一个块的 26 字节只被 8 个线程各读一次。
  - 把 trit 提取写成**纯 float 运算**:`u = b/256`,`g_k = floor(3^k·u)`,`t_n = g_{n+1} − 3·g_n`;点积合并成 `Σ_{k=1..4} g_k·(y_{k−1} − 3·y_k) + g_5·y_4`,其中系数只依赖激活 `y`(由调用者预先准备)。内循环每个 trit 一次 `floor` 加一次 `fma`。**注释明确说"这个 ISA 不能同时发射整数和浮点指令",所以整个解码都留在 float 管线里。**
  - 这是 M0 解包基准的出发点。

---

## 8. 上游生态

### 8.1 Ollama / Hugging Face

见 §3.3、§5.1。

### 8.2 Prism llama.cpp fork(`PrismML-Eng/llama.cpp`,分支 `prism`)

当前 head `73bfe6cc485554822f80c473a3c0c4f25e80f41b`(2026-10-19)。对我们最有价值的提交/分支:

| 条目 | 内容 | 对我们的意义 |
|---|---|---|
| **#358** | llama:**GDN rollback by replay**、**one-graph MTP draft chain**、**K/V-only MTP prompt pass**、sync trim(port) | 验证了"GDN 状态用重放回滚(lazy commit)"可行;**prefill 阶段 MTP 层只需要算 K/V,不用算输出** |
| **#354** | 保留 pending rollback 跨 `seq_cp`;**MTP draft 深度默认 2(CUDA)** | verify 宽度应取 3(1 个 + 2 个 draft)而非 4;这是 CUDA 默认值,Metal 要自己调 |
| **#225** | metal:**dense mat-vec 一次处理 5 个 PTQ1 行**(原来 4 行),"让每个暂存的激活块在另一行上复用" | 和我提的"8 行×1024 K tile"要在 M0 里对比 |
| **#290** | arm:NEON + i8mm `vec_dot` for PTQ1_0("trits are decoded in 8-bit lanes: ((q + (q >> 1)) >> …") | CPU 展开权重(prefill 路径)的参考 |
| **#369** | cuda:staged GDN verification kernel | GDN verify 的参考 |
| #313 | common:top-k 直接从 logits 取 | GPU 采样的参考 |
| 分支 | `metal41-direct-optin`(9-30)、`metal-fwht-tg-512`(9-28)、`perf/fwht-large-blocks`、`perf/fwht-2048-wide-prism`、`hadamard-folded-runtime`(10-10)、`feat/metal-gdn-rows-write-fold`、`cuda/yukon-gdn-verify-20261017` | 实现阶段读 |
| 其它分支名 | `dflash-*`、`feat/dfly-drafter`、`dspark-*`(块扩散/其它 drafter 的实验) | 备选 drafter,暂不考虑 |

Metal 内核文件(分支 `prism`,`ggml/src/ggml-metal/kernels/`):`mul_mv.metal`(PTQ1_0 mat-vec,含 multicol)、`mul_mm.metal`、`gated_delta_net.metal`、`misc.metal`(`kernel_fwht`、`kernel_fwht_tg`,含融合 sign flip 与融合 SwiGLU)、`norm.metal`、`fa.metal`、`conv.metal`、`dequantize.h`;图构建在 `src/models/qwen35.cpp`;`ggml-metal-ops.cpp` 里有 `ggml_metal_op_fwht_swiglu` 融合。关键节选已复制到 `docs/reference/`。

### 8.3 BaseRT

见 §3.3 和 §7.2/7.3。可作为第二条基线(**未验证它能否直接跑 PTQ1_0 的 Qwen3.8 bundle**)。

---

## 9. 性能模型(roofline)

**图例:[实测] [推算] [假设]**

### 9.1 Decode

- 每 token 读取 **5.657GB** [精确]。153.6GB/s 理论峰值下上限 **27.2 tok/s**;按 90% 效率(≈138GB/s)约 **24.4 tok/s** [推算,90% 是据 BaseRT 在 M5 Pro 上的表现假设的];按 Prism llama.cpp 在 M5 Pro 上约 65% 的效率,本机基线大约 **13~14 tok/s**(PQ2_0 7.25GB 为准)[推算,必须在本机实测基线]。
- MTP 投机:每步 = 主干 5.657GB + 每个 draft 0.729GB。1 个 draft 时每步 6.386GB,在 138GB/s 下约 46.2ms;每步产出 `1+α` 个 token,α=0.6~0.8 → **约 33~38 tok/s** [假设 α]。
- 额外流量:GDN 状态读写 302MB/步(约 5%,fp32);KV 读取随上下文线性增长(32K 时 fp16 约 2.1GB,会让 decode 慢约 37%);但 lazy-commit 之后状态流量**不随 verify 宽度增长**。
- CPU 帮不上 decode 的忙:CPU 与 GPU 共用同一条 DRAM,多读不会增加总带宽。

### 9.2 Prefill

- 每 token 48.7 GFLOP。`tok/s ≈ 有效 TFLOPS / 0.0487`。权重读取被大 batch 摊薄(每个 512~2048 token 的 chunk 读一遍),所以 **prefill 完全受算力限制**。
- 实测 `matmul2d` 未调优上限:fp16 18.9 TFLOPS(→ 388 tok/s)、int8 34 TOPS(→ 698 tok/s)[实测 + 推算]。

### 9.3 校准参照(来自论文)

| 平台 | 实现 | decode | prefill |
|---|---|---|---|
| M5 Pro | Prism llama.cpp,Bonsai 2 PQ2_0 | 27.7 | pp512 397 |
| M5 Max | 同上 | 46.8 | 765 |
| M5 Pro | BaseRT,Qwen3.6-27B Q4 | 18.1 | 498(pp2048) |
| M5 Pro | llama.cpp,Qwen3.6-27B Q4 | 15.4 | 392 |

---

## 10. 当前设计(已和用户讨论达成方向,细节待 M0 实测定案)

### 10.1 总原则

1. 模型专用、机器专用:所有 shape、tile、分块、核数假设、容量**编译期常量**,集中在一个配置头里;不做运行时分支、不做动态分配、不做通用性抽象。
2. **启动时一次性做完所有决定**:内存预分配、pipeline AOT、argument table 预绑定、整个 decode 步录进 ICB。
3. decode 受带宽限制:一切围绕"每 token 少读一个字节"和"让 DRAM 一直忙"。
4. prefill 受算力限制:一切围绕"让矩阵单元一直忙",把不属于矩阵乘的事挪到 CPU 或融合掉。
5. 质量门槛优先于速度(§10.15)。

### 10.2 代码、构建、目录(**待用户确认后创建**)

- 语言:**C++17 + MSL(`-std=metal4.1`)**,通过 **metal-cpp**(最新 GitHub main,固定 commit)访问 Metal。不用 Swift、不用 Python。
- 构建:**纯 `Makefile`**,不用 cmake(cmake 只为编译 Prism llama.cpp 基线才装)。MSL 离线编译为 `.metallib`(AOT),pipeline 可用 binary archive/pipeline dataset 序列化,启动时不做 JIT。
- 统一配置:`config/kata_m6.h`(`constexpr`,C++ 与 MSL 共用,用 `#ifdef __METAL_VERSION__` 区分),包含全部 shape、tile、核数假设、内存布局、容量、版本。**`kata-pack` 打包时对照 GGUF 元数据做 `static_assert`,不匹配即拒绝。**
- `VERSIONS` 文件:记录固定的 SDK、metal-cpp commit、Prism commit、模型 digest 等。
- 计划中的二进制(**仅 CLI**):`kata-pack`(GGUF → 重排后的权重文件)、`kata`(运行时:chat/agent 用的 CLI)、`kata-bench`、`kata-verify`(对 Prism llama.cpp 做 logits/KL 对比)。
- 核心规则:热路径零分配;tokenizer 和 chat template 用 C++ 手写;采样放 GPU。

### 10.3 权重格式与布局(**128B 对齐的精确解**)

背景:PTQ1_0 的 28B 块(26B 载荷 + 2B scale)与 128B cache line 不对齐。

**提议(M0 验证)**:因为所有输入维度 K 都是 1024 的倍数(5120、6144、17408;这是 Hadamard 块大小带来的),取 **tile = 8 行 × 1024 个 K**:
- 载荷 = 8 行 × 8 组 × 26B = **1664B = 13 条 128B line**(整数!)。
- scale(分离存放)= 64 个 fp16 = **128B = 1 条 line**。
- 合计 14 条 line = 1792B,正好 1.75 bpw,**无任何 padding**。
- 一个 tile 恰好对应一个 Hadamard 块。
- 一个 simdgroup 的 32 个 lane 每个读 13 个 dword(52B),repack 时把 dword 交错摆放,使每条 load 指令读满一整条 128B line。

注意:**Prism #225 的 Metal 内核用 5 行 tile**,我提议 8 行;两者在 M0 里对比。另外 trit 在字节内的位置、字节在块内的顺序是**我们的 repack 工具可以自由重排的**,只要保证数值等价(§5.6 是原始顺序);按"每线程拥有整字节、全 float 管线解码"(§7.6)来设计新排列。

scale 处理(用户要求的"epilogue 按 K=128 分段累加"):
- decode(GEMV):scale 与载荷同流读取,在寄存器里按组乘加;精确、免费。
- prefill:路径 A(CPU 把权重展开为 fp16,`w=s·t`,GPU 做 half×half,**无 epilogue**,精确)或路径 B(CPU 展开为 int8,GPU 做 int8×int8→int32,**每 128 个 K 做一次 scale 的 epilogue**:用一个 K=128 的 `matmul2d` 得到新的 cooperative tensor,再逐元素乘 scale 加进主累加器;激活量化到 int8 有损,受质量门槛约束)。epilogue 开销估算:每 128 个 K 做 `TM×TN` 次 FMA,对 `TM×TN×128` 次矩阵 MAC,比例 1:128,矩阵单元比 ALU 快数倍,约占 6%~12%(**推算**)。

### 10.4 decode kernel 与融合

- GEMV 的 x 向量放 threadgroup memory 的情况:5120 宽(fp16 10KB)和 6144 宽(12KB)放得下;**17408 宽 fp16 要 34KB > 32KB**,所以 `ffn_down` 要么分两半处理,要么不放 threadgroup memory 而直接从 L1/L2 读 device memory。
- 同输入的权重拼接成一个矩阵、一次 dispatch:GDN 的 `qkv|z|α|β`(α、β 是 bf16,在同一 kernel 里单独处理);attention 的 `q|k|v`;FFN 的 `gate|up` 行交错(SwiGLU 在 epilogue 里做)。
- 全模型只有约 **8 种 GEMV shape**:5120→10240+6144(+α β)、5120→12288+1024+1024、6144→5120(两种来源)、5120→17408(×2)、17408→5120、5120→248320。每种一个手写 kernel,**网格按 12 核对齐,避免尾部效应**(例如 TG 数取核数的整数倍/远多于核数以便动态均衡)。
- 解码走 **float 管线**(Prism 的发现:此 ISA 不能同时发射 int 和 float):`floor(3^k·u)` 一次 floor + 一次 fma 每 trit。另两个候选:threadgroup LUT、**activation-side LUT**(为每 5 个激活预计算 243 项部分和,每个权重字节查一次表加一次加法)。M0 对比。

### 10.5 Hadamard(重点优化)

- 三个 sign 向量(5120/6144/17408 宽)**所有层共享**。`sign ⊙ γ`(RMSNorm 权重)在离线打包时合并成一个常量向量(`γ_l ⊙ d`),运行时只乘一次。(对 6144 和 17408 宽的输入只有 `d`。)
- **不再有独立的 norm 和 FWHT kernel**:每个 GEMV 在自己的 prologue 里完成"读 fp32 残差 → 求平方和 → 乘 `γ⊙d` → FWHT → 转 fp16"。残差只有 20KB,从 L2 读,开销约 1µs。
- `ffn_down`(17 个块):**按 1024 块做 split-K**,每个 threadgroup 只变换自己那一块(开销降到 1/17),部分和用 fp32 atomic 加进残差(求和顺序不定,结果最后几位有差异,可接受;也可以用确定性的 per-block 部分和再求和)。
- FWHT-1024 = H32 ⊗ H32:5 级在寄存器内,5 级用 `simd_shuffle_xor`(或 threadgroup 转置),M0 对比两种写法。
- 每层 dispatch 目标:GDN 层约 5 个、attention 层约 3~4 个;预计共省掉每 token 约 400 次小 dispatch,约 5%~7% 的 token 时间(**推算**)。
- embedding:`token_embd` 保持三值常驻,GPU 上查表后做反向 FWHT(一行 5120,几乎免费)。因为下一个 token id 在 GPU 上,GPU 必须能自取。

### 10.6 GDN

- 状态 fp32(config `mamba_ssm_dtype: float32`),每层 3.1MB。fp16/bf16 状态仅在质量门槛通过后考虑。
- **lazy commit**:verify 不写回最终状态,保留步前状态,只保存每个 token 的 k/v/门控;下一步的 GDN kernel 开头,在寄存器里**重放已被接受的 token 的更新**,再处理新 token。状态流量恒为 150MB 读 + 150MB 写,与 verify 宽度无关。已被接受 token 的**输出在 verify 时已经算对了**,只推迟状态提交。(与 Prism #358 "GDN rollback by replay" 同思路。)
- prefill:chunk=64 的并行算法(块内矩阵乘 + 三角求解,块间扫描状态)。
- conv 状态(核 4)随状态一起处理。

### 10.7 Attention 与 KV

- 只有 16 层 full attention。KV 预分配到最大长度(64K),连续布局(单用户不需要 paged)。
- decode:split-K flash-decoding(GQA 6 个 Q 头共享 1 个 KV 头);固定 split 数,多余的 threadgroup 直接退出(ICB 里 dispatch 网格固定)。
- prefill:head_dim=256 的 flash-attention,tile 受 32KB threadgroup memory 限制,靠寄存器和 `matmul2d`;causal,GQA。
- KV 量化:**默认先试 int8 + 每 token 每头 scale**;fp8 E4M3 可被 `matmul2d` 原生消费但 3 位尾数误差更大,按同一门槛评估;sub-4-bit 属探索项。
- RoPE:partial(64/256 维),文本场景下 mrope 退化成标准 RoPE(`mrope_section [11,11,10]` 只对视觉有意义——**这一点我凭对 Qwen3.5 的了解,实现时需对着 HF/Prism 核实**)。

### 10.8 投机解码(MTP + CPU n-gram)

- 依赖链:`verify_n → h_n → MTP draft → verify_{n+1}`,**严格串行**。MTP 在 GPU 上跑(ANE 无法并行,见 §11)。
- verify 宽度固定(静态 shape),建议 **3**(1 个 MTP draft + 最多 2 个来自 CPU n-gram;参考 Prism #354 的深度 2),M0 之后再调。decode 受带宽限制,verify 3 个 token 和 1 个 token 读的权重一样多;**风险:M=3 时 ALU 解包是否仍然受带宽限制**,待 M0 确认。
- **CPU n-gram/prompt-lookup 草稿**:agent 场景(改代码、回显 JSON/工具输出)有大量复制自上下文的片段,CPU 查 n-gram 零 DRAM 流量;CPU 没赶上就不填。
- prefill 阶段 MTP 层**只算 K/V**(Prism #358)。
- 可选:MTP 层从 Q8_0 重新量化到 Q4(约 230MB),因为验证是精确的,只影响接受率,不影响结果;需实测接受率。
- 备选 drafter:Prism 新发布的 Kata-2 2B(约 0.44GB 三值),未评估。

### 10.9 prefill 路径

- 大 tile GEMM,`matmul2d` 操作数直读 device memory,每个 threadgroup 的 tile 形状按 M0 扫描结果定。
- **CPU 展开权重(用户的"解包放 CPU"想法,在 prefill 里成立)**:CPU 在 GPU 计算第 L 层时,把第 L+1 层从 PTQ1_0 展开成 staging 缓冲(双缓冲)。每层约 383M 个权重;按 1024-token chunk、fp16 路径 70% 效率,每层 GPU 约 59ms,CPU 只需每秒展开约 6.5G 个权重,写出约 13GB/s,DRAM 额外占约 17%。**这是推算,M0 要实测 NEON 吞吐和与 GPU 的 DRAM 争用。** 参考 Prism #290 的 NEON/i8mm 解码。
  - 路径 A:展开成 fp16(`w=s·t`,精确,2B/权重,每层约 766MB,双缓冲约 1.5GB)。
  - 路径 B:展开成 int8(1B/权重,每层约 383MB)+ int8 激活 + 每 K=128 epilogue。
  - 决策取决于 M0:int8 是否真的比 fp16 快(已测到 1.8 倍)、epilogue 实际开销、质量门槛。
- chunk 大小建议 ≥1024,摊薄 staging 流量。
- 长 prefill 的 command buffer 要拆分,避免 GPU watchdog。
- 只对最后一个 token 算 lm_head。

### 10.10 命令编码(Metal 4)

- 启动时:AOT 编译 pipeline;预绑定 `MTL4ArgumentTable`;把整个 decode 步(约 64×5 个 dispatch 加头尾)录进 **ICB**(compute ICB 从 Apple3 支持,但**ICB 里的 dispatch 网格是固定的**,不支持 indirect 参数——**这一点我凭印象,需 M0 验证**);配好 `MTLResidencySet` 和固定 barrier。
- 每 token 的 CPU 成本:更新一个小参数 buffer(位置、上下文长度)+ 提交一个 command buffer。
- 采样(temp/top-k/top-p/min-p)和下一步 token 反馈全在 GPU;CPU 只异步取回 token id 流式输出。
- 跨多 token 摊销 CPU/GPU 同步(BaseRT 的做法)。
- persistent megakernel(用 atomic 做全局屏障):Metal 4.1 的 acquire/release 内存序和 device scope 屏障使它在内存序上可定义,但**仍无前向进度保证**,而且 BaseRT 论文显示大模型下 dispatch 开销占比不大 → **后期实验,不进主路线**。

### 10.11 内存预算(12,124 MB 工作集)

| 项 | 大小 |
|---|---|
| 权重常驻(层 5.379 + lm_head 0.278 + embedding 0.278 + MTP 0.451) | 6.39GB |
| GDN 状态,双份(lazy commit/回滚) | 0.30GB(+ conv 状态约 6MB) |
| KV:64K token int8(若质量门槛通过),否则 32K fp16 | 约 2.2GB |
| prefill staging 双缓冲(fp16 展开) | 1.5GB |
| 激活 scratch(chunk 1024) | 0.3GB |
| MTP 的 KV(64K,fp16) | 0.27GB |
| **合计** | **约 11.0GB**,余量约 1.1GB |

- 尝试:`sudo sysctl iogpu.wired_limit_mb` 调高并验证不降性能(用户要求"验证尝试调到最大")。
- 权重加载方式待 M0:**mmap + `bytesNoCopy`** 零拷贝、但文件映射页可能在内存压力下被换出;**匿名页 + `F_NOCACHE` 读入 + residency set** 更稳但冷启动要读 6.4GB。**这是一个风险点。**
- `maxBufferLength` 9.1GB > 权重总量,可以做成一个大 buffer 内按偏移切片。

### 10.12 CPU 的角色(Super/P/E)

- **做**:prefill 展开权重(P+Super 核)、流式 tokenize、n-gram 草稿、detokenize/流式输出/stop 检测、prefix 快照 IO(E 核)。
- **不做**:decode 里的任何每层关键路径工作(CPU↔GPU 同步延迟 ≥10~50µs)。
- 热线程用 `QOS_USER_INTERACTIVE`,倾向调度到 Super/P 核(macOS 没有严格亲和性)。
- 另有 CPU 侧矩阵协处理器(AMX/SME,die shot 标注),**不使用**(只用 Metal 的约束)。

### 10.13 ANE 决策

- 用户允许走 Core ML 或私有 API。**我的结论:不放进主路线。**
  - **MTP draft 放 ANE:否决**(依赖链严格串行,且 ANE 不增加 DRAM 带宽)。
  - **ANE 当 prefill 协处理器**:唯一合理位置(按层流水线:ANE 处理前几层、GPU 处理后几层)。代价:Core ML 持有自己格式的权重副本(我的判断,需实测)、没有 Python 就要手写 Core ML 的 protobuf/MIL、16 个 ANE 核的实际 TOPS 不清楚、BaseRT 论文里 uzu 走 ANE 的 prefill 收益只有 1%~13%。
  - 本机 `AppleNeuralEngine.framework` 在 `/System/Library/PrivateFrameworks`,`coremlc` 在 Xcode 工具链里(已确认存在)。
  - **排到最后**,GPU 路径完成后再做可行性探测。

### 10.14 SSD 与预计算清单(`kata-pack` 离线完成)

- 重排后的权重文件(§10.3 布局),存 SSD(剩余约 97GiB)。
- 合并常量向量 `γ_l ⊙ d`;`A = −exp(A_log)` 等预变换(已在 GGUF 里);RoPE cos/sin 表(64K 位置 × 32 频率 fp32 约 16.8MB,或实时计算);分词器的 BPE 表预处理成扁平数组/完美哈希以便 mmap;chat template 编译成 C++ 代码;每个 kernel 的静态工作划分表。
- **prefix 缓存**:把系统提示词/历史对应的 GDN 状态(151MB)+ KV 存盘;8K 前缀约 690MB,SSD 读约 0.14 秒(按 5GB/s 估),对比重新 prefill 约 27 秒(按 300 tok/s 估)。快照点打在**最后一个 user 消息的边界**(见 §5.9 的核实 TODO)。聊天/agent 的循环里前缀通常是追加式的,不需要回滚。

### 10.15 质量门槛(**提议,待用户确认具体阈值**)

- 参考:Prism llama.cpp(fp16 激活)的 logits。
- 每个有损手段(int8 激活、KV 量化、fp16 状态、Q4 MTP)必须在**≥10 万 token 的混合语料(文本、代码、长推理)上**:对参考的平均 KL ≤ 0.001 nat/token,top-1 一致率 ≥ 99.7%;另加 32K 上下文的 needle 测试(验证 KV 量化)。
- 不用端到端 benchmark 做门槛:HumanEval+ 164 题在 thinking 模式下每题约 1 万 token,在这台机器上要约 15 小时。
- 默认路径:fp16 激活,残差/GDN 状态/softmax 用 fp32(**精确路径 A**)。int8 激活默认关闭。

---

## 11. 已做的决定与被否决的想法

| 想法 | 决定 | 理由 |
|---|---|---|
| C++ + MSL + **metal-cpp(最新 GitHub main)** | 采纳 | 用户指定;已验证 |
| Metal 4 命令编码/ICB | 采纳 | 用户要求;M0 验证 ICB 细节 |
| 所有 shape 写死到一个 config 头 | 采纳 | 用户要求 |
| 启动时预分配最大内存 | 采纳 | 用户要求;预算见 §10.11 |
| 利用 CPU 与 GPU 并行 | 采纳,**但只放在算力受限的阶段** | CPU 与 GPU 共用 DRAM 带宽,decode 里帮不上 |
| 解包放 CPU | **prefill 里采纳;decode 里不行** | decode 里解包后的数据变大 4~5 倍,DRAM 流量增加,违背目标 |
| 解包放 ANE | 否决 | ANE 不是通用解包引擎 |
| **MTP draft 放 ANE 并行** | **否决** | 依赖链严格串行(draft 需要主模型本步的 hidden state);ANE 不增加带宽 |
| ANE 作 prefill 协处理器 | 排最后 | 权重副本、无 Python 构建 Core ML、收益不确定 |
| int2b_format 作常驻权重 | 否决 | 每 token 多读约 1.2GB,decode 慢约 17% |
| **只常驻一份 PTQ1_0 重排权重** | 采纳 | decode 直读;prefill 由 CPU 展开 |
| 用 `char` 类型做 int8 matmul2d | 否决 | 编译失败,必须 `int8_t` |
| scale 折叠进激活/用 E8M0 | 否决 | scale 不是 2 的幂、不可分解(§5.7) |
| 跳零/稀疏 | 否决 | 三值均匀分布 1/3 |
| persistent megakernel | 后期实验 | 无前向进度保证,收益不明 |
| CPU n-gram 草稿 + 加宽 verify | 新增建议 | 零 DRAM 流量 |
| GDN lazy commit(重放) | 新增建议 | 状态流量与 verify 宽度无关;Prism #358 同思路 |
| int8 激活 prefill | **有条件**:质量门槛通过才启用 | 用户:不能让模型变笨 |
| 用 `matmul2d` 操作数直读 device memory | 采纳 | 用户要求 + BaseRT 结论 |
| 数据驱动的架构描述符(BaseRT 的做法) | 否决 | 我们要写死 |
| 视觉、continuous batching、paged KV | 不做 | 用户明确 |

---

## 12. 未解决的问题、风险与 TODO

### 12.1 风险

| # | 风险 | 如何解决 |
|---|---|---|
| **R1** | **注意力输出门**:HF `config.json` 写 `output_gate_type: "swish"`,但 **HF transformers 的 `modeling_qwen3_5.py`(`attn_output * torch.sigmoid(gate)`)和 Prism 的 `qwen35.cpp`(`ggml_sigmoid(gate)`)都用 sigmoid,且没有任何实现读取该 config 项**;Qwen3.8 仓库里没有自带 modeling 代码 | M2 里同时实现两种,对 Prism 参考的 logits 做 KL 比较;也可查更新的 transformers/vLLM 实现 |
| R2 | 带宽实际可达多少(理论 153.6GB/s) | M0.1 |
| R3 | 解包是否追得上 DRAM(尤其 verify M=3) | M0.2 |
| R4 | M6 矩阵单元的真实吞吐(上面的数字未调优、用常数数据) | M0.3 |
| R5 | `int2b`/fp8 格式 tensor 如何构造、吞吐多少 | M0.3(需 `MTLTensor`) |
| R6 | ICB 能否表达变长 attention、dispatch 间隙多大 | M0.4 |
| R7 | 权重常驻方式:mmap 文件映射页在内存压力下是否被换出 | M0.8 |
| R8 | MTP 真实接受率(α=0.6~0.8 是假设) | M4 |
| R9 | Hadamard 的符号顺序、归一化、`gdn_v_grouped` 约定要与 Prism 逐位对齐 | M2,参考 `docs/reference/prism_*` |
| R10 | Apple11(M6)能力不在任何官方文档里 | 全靠 M0 实测 |
| R11 | 长 prefill command buffer 触发 GPU watchdog | 拆分 command buffer |
| R12 | `KNOWN_ISSUES.md`(Prism HF 仓库)没读 | 读 |

### 12.2 TODO(读/核实)

1. 读 `docs/reference/hf_transformers_modeling_qwen3_5.py` 和 `prism_qwen35_graph.cpp`,**精确写出 GDN 与 attention 的公式**(我对 GDN 的记忆:`g = A·softplus(a + dt_bias)`,衰减 `exp(g)`,`β = sigmoid(b)`,Q/K 做 L2 norm 并乘 `1/√d_k`,causal conv1d + SiLU,递推 `S ← S·exp(g)·(I − β k kᵀ) + β v kᵀ`,输出 `o = S q`,再经 gated RMSNorm(`silu(z) ⊙ norm(o)`)——**这些是回忆,必须对着源码核实**)。
2. 读 Raschka 的 `qwen3_5_transformers.py`(用户给的链接)。
3. 读 `prism_metal_mul_mv_ptq1_0.metal`、`prism_metal_fwht.metal`、`prism_metal_gated_delta_net.metal` 的细节,以及 `mul_mm.metal` 的 PTQ1 路径。
4. 对照 Qwen chat template 核实"历史 thinking 被剥掉"。
5. 读 Prism HF 的 `KNOWN_ISSUES.md`。
6. 找 MPP Programming Guide(March 2026);问用户是否有。

---

## 13. 路线图与下一步

### 13.1 立即要做(按顺序,待用户放行)

0. 复核环境(§0.4)。
1. 向用户确认 §0.3 的三件事。
2. 建项目骨架:`third_party/metal-cpp`(固定到 `8cdde68ae36a636c84cbb48ebcbc8f3b60251ea0` 或更新的 main)、`config/kata_m6.h`、`VERSIONS`、`Makefile`、`src/`、`tools/`。
3. `brew install cmake`;克隆 Prism llama.cpp 最新 master(当前 `73bfe6cc4855…`),开启 Metal 编译;下载 HF 的 `Ternary-Bonsai-2-27B-PTQ1_0.gguf`(5.93GB)和 Ollama 的 MTP GGUF(6.4GB);跑 `llama-bench` 的 tg128/pp512,**记录本机基线**(用于所有后续对比,也是 logits 参考)。
4. M0 微基准。

### 13.2 M0 微基准清单(每项都要写通过标准)

| 编号 | 内容 | 要回答的问题 |
|---|---|---|
| M0.1 | GPU 从 DRAM 读流的实际带宽:buffer 1~4GB(超过 SLC),TG 数/线程数/向量宽度/每 lane 在途 load 数的扫描;128B 对齐 vs 不对齐 | 实际能到 153.6 的百分之多少(决定 decode 上限) |
| M0.2 | 三值 GEMV(真实尺寸 17408×5120 等):(a) Prism 的 float 管线 `floor` 解码,(b) TQ1_0 式乘移位,(c) threadgroup LUT,(d) activation-side LUT,(e) 2-bit 对照;行 tile 4/5/8;**M=1 与 M=3**;tile 布局(§10.3) | 能否达到 M0.1 的 ≥90%;选 trit 打包重排方案 |
| M0.3 | `matmul2d` 扫描,**随机数据**、真实形状(M=512/1024/2048 × N=17408 × K=5120 等)、tile/SG 扫描、操作数在 device vs threadgroup、`tensor_handle` vs inline、fp16/bf16/int8、K=128 分段 epilogue 的开销、`relaxed_precision`、`int2b`/fp8(需 `MTLTensor`)、与 `simdgroup_matrix` 对照 | prefill 上限;fp16 还是 int8;epilogue 真实开销 |
| M0.4 | 400 个小依赖 dispatch:Metal 3 encoder、Metal 4 encoder、ICB;barrier 开销;GPU 时间线上的间隙;persistent kernel 软件屏障(acquire/release) | 每 dispatch 开销;ICB 能否表达固定网格 + 提前退出;是否值得 megakernel |
| M0.5 | FWHT-1024 的写法对比(shuffle+寄存器 / threadgroup 转置);17408 split-K 加 fp32 atomic 的开销 | Hadamard 融合方案的取舍 |
| M0.6 | GDN 状态更新的带宽(每层 3.1MB 读+写),SLC 命中行为 | lazy commit 的实际成本 |
| M0.7 | CPU NEON:PTQ1_0 → fp16/int8 展开的每核吞吐、多核扩展性、与 GPU 的 DRAM 争用 | CPU 展开路径可行性 |
| M0.8 | 内存驻留:mmap+`bytesNoCopy` vs 匿名页+`F_NOCACHE`;`iogpu.wired_limit_mb` 调高;内存压力下是否被换出;分配接近 12.1GB 的表现 | 权重加载与驻留方案 |
| M0.9 | (最后)ANE 可行性:Core ML 的 5120×17408 线性层吞吐;与 GPU 同时跑的争用 | ANE 是否值得做 |

产出:一张**"M6 实测真值表"**,替代用户给的那张层级表。

### 13.3 里程碑

- **M1**:GGUF 解析、`kata-pack`、分词器、零拷贝加载;单个三值 GEMV 对 CPU 参考做正确性验证。
- **M2**:完整前向(先写朴素但正确的 kernel),logits 对齐 Prism 参考;解决 R1(注意力门)、R9(Hadamard 约定)。
- **M3**:decode 优化(融合、ICB、GPU 采样),目标是带宽 roofline 的 ≥85%。
- **M4**:MTP 投机解码 + GDN lazy commit;CPU n-gram 草稿。
- **M5**:prefill(`matmul2d` GEMM、CPU 展开 staging、chunked GDN、flash-attention)。
- **M6**:长上下文(KV 量化)、SSD prefix 缓存。
- **M7**:ANE 可行性探测(可选)。

---

## 14. 更正日志(我在对话中说错或不准的地方)

| # | 之前的说法 | 更正 |
|---|---|---|
| 1 | "metal-cpp 26 是目前发布的最新版" | **错**。只查了官网 zip;Apple 官方 GitHub `apple/metal-cpp` main 已是 macOS 27.0.1(2026-09-22,commit `8cdde68a…`),带 `LanguageVersion4_1` 和 `Apple11`,不需要补丁头 |
| 2 | "GPU family 到 apple10" | M6 是 **Apple11**(apple9/10/11/metal4 全为 true);官方文档只列到 Apple10 |
| 3 | MTP 层 462MB | 我把整个文件长度(含 11.1MB 头部)减去偏移了。**正确值 451,319,808 B**;每个 draft 读取量 0.729GB(不是 0.74) |
| 4 | "Metal 编译器组件缺失,需要 `-runFirstLaunch`" | 用户已解决,`xcrun metal` 现在可用 |
| 5 | 用 `char` 作为 int8 matmul2d 类型 | 必须用 `int8_t` |
| 6 | "`half × int8 → float` 不支持" | 用 `int8_t` 后**支持**(与 fp16 同速) |
| 7 | 头文件注释里列出的 `int8×int8→int32` | 用 `char` 写不可用,`int8_t`/`uchar` 可用 |
| 8 | 早期方案里 Swift 做宿主 | 用户要 C++ + metal-cpp,已改 |
| 9 | ANE 一开始建议"完全排除" | 用户说允许;改为"排最后做可行性探测",MTP-on-ANE 仍否决 |
| 10 | 之前未核实 PTQ1_0 元素排列顺序 | 已读 Prism 源码确认(§5.6) |

---

## 15. 附录 A:复现命令

```sh
# 0. 目录
cd <repo root>

# 1. 取 GGUF 头(前 12MB)并解析(Swift)
B=https://registry.ollama.ai/v2/mannix/Bonsai-2-27B/blobs/sha256:174c9c6584e118766d5511e00a84fd8a31d146248915a726cfac8e6d140c6204
curl -sL -r 0-12000000 -o /tmp/kata_head.bin $B
swiftc -O docs/probes/gguf_dump.swift -o /tmp/gguf_dump && /tmp/gguf_dump > /tmp/out.txt     # 脚本里硬编码读 /tmp/kata_head.bin

# 2. 按偏移取任意张量(数据区起点 DS=11122176;偏移来自 docs/data/gguf_header_dump.txt)
DS=11122176; OFF=594124800; LEN=19496960         # 例:blk.0.ffn_down.weight
curl -sL -r $((DS+OFF))-$((DS+OFF+LEN-1)) -o /tmp/ffn_down.bin $B
clang++ -O2 -std=c++17 docs/probes/scale_analysis.cpp -o /tmp/scl && /tmp/scl /tmp/ffn_down.bin 17408 5120   # 参数:K N

# 3. 取最新 metal-cpp,编译探测程序
curl -sL -o /tmp/mcpp.zip https://github.com/apple/metal-cpp/archive/refs/heads/main.zip && unzip -q -o /tmp/mcpp.zip -d /tmp/latest
M=/tmp/latest/metal-cpp-main
clang++ -O2 -std=c++17 -I$M docs/probes/metalcpp_smoke.cpp -framework Metal -framework Foundation -framework QuartzCore -o /tmp/smoke && /tmp/smoke
clang++ -O2 -std=c++17 -I$M docs/probes/matmul2d_probe.cpp -framework Metal -framework Foundation -framework QuartzCore -o /tmp/mmprobe && /tmp/mmprobe

# 4. PDF 提取文本
swiftc -O docs/probes/pdf_extract.swift -o /tmp/pdfx && /tmp/pdfx Metal-Shading-Language-Specification.pdf /tmp/msl.txt
```

`matmul2d_probe.cpp` 里的配置列表(`cfgs`)被反复改写过,当前保存的是最后一轮(fp16 与 int8 的 tile/SG 扫描);它的 kernel 生成器是核心:

```cpp
// kernel 模板(MSL,由 kern() 用 snprintf 生成)
kernel void k(device TA* A [[buffer(0)]], device TB* B [[buffer(1)]], device TC* C [[buffer(2)]],
              constant uint3& dims [[buffer(3)]], uint2 tgid [[threadgroup_position_in_grid]]) {
  int M = dims.x, N = dims.y, K = dims.z;
  tensor<device TA, dextents<int32_t,2>, tensor_inline> tA(A, dextents<int32_t,2>(K, M));
  tensor<device TB, dextents<int32_t,2>, tensor_inline> tB((device TB*)B, dextents<int32_t,2>(N, K));
  tensor<device TC, dextents<int32_t,2>, tensor_inline> tC(C, dextents<int32_t,2>(N, M));
  constexpr auto desc = matmul2d_descriptor(TM, TN, static_cast<int>(dynamic_extent));
  matmul2d<desc, execution_simdgroups<SG>> op;
  auto mA = tA.slice(0, tgid.y*TM);
  auto mB = tB.slice(tgid.x*TN, 0);
  auto mC = tC.slice(tgid.x*TN, tgid.y*TM);
  op.run(mA, mB, mC);
}
// 头部:#include <metal_stdlib> <metal_tensor> <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
//       using namespace metal; using namespace mpp::tensor_ops;
// 编译选项:opt->setLanguageVersion(MTL::LanguageVersion4_1)(旧版 metal-cpp 用 (MTL::LanguageVersion)262145)
// dispatch:dispatchThreadgroups(MTL::Size(N/TN, M/TM, 1), MTL::Size(32*SG, 1, 1))
```

## 16. 附录 B:一些容易再次踩的坑

- 并行 tool call 里有依赖关系的步骤会竞争(例如先 `mkdir` 再 `cd`)。
- `xcrun metal` 需要 `DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer`(或 `xcode-select` 已指向 Xcode;现在已指向)。
- GitHub API 未认证限速 60 次/小时。
- 官网的 `metal-cpp_26.zip` 是旧的,要用 GitHub main。
- MSL 里 int8 类型写 `int8_t`,不要写 `char`。
- 格式 tensor(`int2b_format` 等)不能从裸指针构造。
- GGUF 张量偏移是**相对数据区起点**(11,122,176),不是相对文件开头。
- `read` 工具读不了 PDF,用 `docs/probes/pdf_extract.swift`;图片(png/webp)可以直接 `read`。
- 用户的内存层级表里的带宽数字(寄存器/L1/L2/ANE SRAM 的 TB/s)没有出处,**不要当设计依据**。

## 17. 附录 C:Qwen3.8 config 的关键字段(原文节选,取自 HF `config.json`)

```json
"text_config": {
  "attn_output_gate": true, "output_gate_type": "swish",
  "full_attention_interval": 4, "head_dim": 256, "hidden_act": "silu",
  "hidden_size": 5120, "intermediate_size": 17408,
  "linear_conv_kernel_dim": 4, "linear_key_head_dim": 128, "linear_num_key_heads": 16,
  "linear_num_value_heads": 48, "linear_value_head_dim": 128,
  "mamba_ssm_dtype": "float32", "max_position_embeddings": 262144,
  "mtp_num_hidden_layers": 1, "mtp_use_dedicated_embeddings": false,
  "num_attention_heads": 24, "num_hidden_layers": 64, "num_key_value_heads": 4,
  "partial_rotary_factor": 0.25, "rms_norm_eps": 1e-06,
  "rope_parameters": {"mrope_interleaved": true, "mrope_section": [11,11,10],
                      "partial_rotary_factor": 0.25, "rope_theta": 10000000, "rope_type": "default"},
  "tie_word_embeddings": false, "vocab_size": 248320 }
```

---

*Checkpoint 001 结束。下一个 checkpoint 应在完成 M0 或项目骨架后写入 `docs/CHECKPOINT-002.md`,并保持同样的自包含格式。*

---

## 18. 增补 A(2026-10-23 晚):prefill 真实形状实测,以及"路径 A(CPU 展开成 fp16)"的结论

**起因**:用户问:prefill 受算力限制,那么路径 A(CPU 把权重展开成 `w = s·t` 的 fp16,GPU 做 half×half,无 epilogue、数值精确)虽然权重字节变多,是不是合适的方案?

**结论:是,合适。** 带宽变大对 prefill 不是瓶颈;但有三个前提(见 18.3)。

### 18.1 实测(`docs/probes/matmul2d_prefill_shape_probe.cpp`)

- 形状:FFN gate/up,**N=17408、K=5120**(fp16 权重 178MB、int8 权重 89MB,远大于任何缓存);M = 512 或 2048 个 token。
- **随机数据**(激活随机,权重随机 {-1,0,+1}×0.01),不再用常数填充;`matmul2d` 操作数直读 device memory;输出 float;运行时编译 Metal 4.1;8 次取最小(去掉前 2 次)。
- 扫描:tile(TM×TN, SG)× 网格顺序(N 方向最快 / M 方向最快)× B 布局(row-major / tile-major)。

**每个配置里最好的(及典型差的)结果:**

| 类型 | M | 最好配置 | 吞吐 | "权重只读一次"所需带宽 | 最差配置的吞吐 |
|---|---|---|---|---|---|
| fp16 (half×half→float) | 512 | 64×128 / SG4 / N-fast / row-major | **17.4 TFLOPS** | 34 GB/s | 4.3(32×128 / M-fast) |
| fp16 | 2048 | 32×64 / SG1 / N-fast / tile-major | **17.2 TFLOPS**(row-major 15.7) | 8.4 GB/s | 3.7(32×128 / M-fast) |
| int8 (int8×int8→int32) | 512 | 64×128 / SG4 / N-fast / tile-major | **36.9 TOPS** | 36 GB/s | 14.4 |
| int8 | 2048 | 64×128 / SG4 / N-fast / tile-major | **35.7 TOPS**(row-major 32.9) | 8.7 GB/s | 12.8 |

**解读:**
1. **fp16 展开权重时,真实形状(权重 178MB,必须从 DRAM 流入)仍能跑到 15.7~17.4 TFLOPS,约为前面 4096³ 实测最高值 18.9 的 83%~92%**。权重只需要 8~34GB/s(M=2048 时仅 8GB/s),远低于 DRAM 的约 138GB/s。所以 **prefill 确实是计算受限,路径 A 的"字节变多"几乎没有代价**。
2. **int8 在真实形状上达到 33~37 TOPS,约为 fp16 的 2.1 倍**,比之前 4096³ 的 34 还略高。
3. **tile 形状/网格顺序/布局的影响极大**(fp16 在 3.7 到 17.4 之间,int8 在 12.8 到 36.9 之间),而且**没有简单规律**:`N-fast` 大多更好,`tile-major` 布局有时更好、有时更差,`32×128 / M-fast` 最差。M0.3 必须系统扫描,不能凭直觉选。
4. 吞吐上限的推算(每 token 48.7 GFLOP,只算 matmul):fp16 路径 17 TFLOPS → **约 350 tok/s**;int8 路径 36 TOPS → **约 740 tok/s**。实际再乘 60%~80%(GDN、attention、norm、FWHT 等):A 路径约 **210~280 tok/s**,B 路径约 **440~590 tok/s**(**推算**)。
5. 局限:只测了 matmul 本身,输出写 float 142MB(实际会融合 epilogue 写 fp16);没有同时让 CPU 在展开权重(争用 DRAM,属于 M0.7);没有测更小 M(<512)。

### 18.2 为什么"压缩失去意义"不成立

- 压缩的目的是**减少每 token 读的字节**,这只在 decode(带宽受限)里有价值。
- 我们**只常驻一份** 1.75bpw 的 PTQ1_0 重排权重(6.39GB)。fp16 只是**短暂的 staging**(最多一两层、甚至一两个矩阵的量),整个模型展开成 fp16 要 48.7GB,本来也放不下。
- prefill 里每个权重字节被 M 个 token 复用,每字节对应 `M×(2 FLOP)` 的计算;机器的平衡点约 137 FLOP/字节(19 TFLOPS ÷ 138GB/s),M=512 时 fp16 权重也有 512 FLOP/字节,所以带宽不是问题。

### 18.3 采用路径 A 的三个前提(设计要求)

1. **展开成本必须摊薄 → prefill 采用"层优先"(layer-major)而不是"chunk 优先"。**
   - chunk 优先(处理完一个 chunk 的所有层再处理下一个 chunk):每个 chunk 都要把 64 层的权重重新展开一遍(48.7GB fp16 写入/ chunk)。chunk=1024 时 GPU 约 3 秒,展开约 1~2 秒、占 DRAM 约 20%;chunk 小于约 300 token 时展开成为瓶颈。
   - **层优先**:先把整个 prompt 的所有 chunk 依次通过第 L 层,再进入第 L+1 层。每层权重**每个 prompt 只展开一次**,总展开量固定为 48.7GB(约 1~2 秒,与 GPU 计算完全重叠),与 prompt 长度无关。
   - 代价:需要保存整段序列的残差:N×5120×4B(fp32),32K 约 0.67GB,64K 约 1.34GB。GDN 状态和 KV 本来就是逐层写一次,天然兼容;MTP 层的 K/V-only 预填充排在最后一层之后即可;prefix 快照照常。
   - 内存:64K 时用"逐矩阵"的 staging 槽位(最大一个矩阵 fp16 178MB,3~4 个槽 <0.8GB)代替"双层 staging 1.5GB",可以省出这部分。
2. **短 prompt / 小批量要有另一条路径。** 展开一层的成本(CPU 约 25ms/层,按 30GB/s 写带宽估)相当于该层约 **500 个 token** 的 matmul 时间。所以 N ≲ 300~600 的 prefill(agent 里常见的"追加几百 token 的新一轮")不该走展开路径,而要走 **GPU 内核内解包 base-3 权重**的路径(中间 M 档,待设计)。分档(估计,M0 校准):M≤4 decode/verify GEMV;4<M≲512 GPU 内核内解包;M≳512 层优先 + CPU 展开。
3. **tile/调度/布局必须靠 M0.3 扫描定案**(18.1 第 3 点)。特别是权重 staging 的布局(row-major 还是 tile-major)由 CPU 展开时决定,可以免费选最优者。

### 18.4 路径 B(int8)的位置

- 2.1 倍的算力优势是真实的,但它要求激活量化到 int8 并做逐 K=128 的 scale epilogue,**有损**,必须通过 §10.15 的质量门槛。
- 因此:**路径 A 是默认的、精确的 prefill 路径;路径 B 是质量门槛通过后的加速档。**
- 路径 C(`half × int8` 权重 → float,1B/权重,速率同 fp16,需要 epilogue)只在 staging 内存吃紧时才有意义,否则被 A 取代。

### 18.5 新增/修改的 TODO

- M0.3 增加:真实形状 + 随机数据 + tile/顺序/布局全扫描;小 M(16、64、128、256)的曲线;确认 fp16 与 int8 各自的最优配置。
- M0.7 增加:CPU 展开 fp16 的写带宽上限,以及它和 GPU matmul 同时运行时的互相影响(这决定层优先方案里"重叠"是否真的免费)。
- 设计待办:"M 中间档(4<M≲512)的 GPU 内核内解包 matmul"需要单独设计(解包到 threadgroup memory 或寄存器,再喂 `matmul2d`/`simdgroup_matrix`)。

---

## 19. 增补 B(2026-10-23 夜):persistent megakernel、DRAM 实测、CPU SME、ANE 私有 API、外部 drafter

用户新增指令(摘要):(1) 利用模型全部事实,针对这台硬件调优 tile/网格顺序/布局;(2) **用 persistent megakernel 减少 overhead、减少 dispatch**;(3) 问:能否让 CPU 提前预处理下一个 token、能否融合成一个大内核;(4) 问:备选 drafter(Kata-2 2B)能否放在 ANE/CPU 上提前做;(5) **尝试使用 CPU 的矩阵协处理器(AMX/SME),如果能加速**;(6) **尽量使用 ANE,如果能加速**。

### 19.1 新增探测程序(`docs/probes/`)与原始输出

| 文件 | 作用 |
|---|---|
| `persistent_kernel_probe.cpp`,输出 `docs/data/persistent_kernel_probe_output.txt` | 全局屏障/共驻上限、依赖 dispatch 链开销、带宽上限、persistent vs per-phase dispatch |
| `sme_throughput_probe.cpp`、`sme_dual_unit_probe.cpp` | CPU SME 外积吞吐、单元数量(**内联汇编**) |
| `ane_private_class_probe.mm` | ANE 私有类与设备信息 |

构建:`clang++ -O2 -std=c++17 -I<metal-cpp> persistent_kernel_probe.cpp -framework Metal -framework Foundation -framework QuartzCore`;SME 探测必须 `-march=armv9.2-a+sme2`。

### 19.2 DRAM 实测带宽(M0.1 的雏形)

- 只读流式读取 1.61GB 随机数据(超出所有缓存),`uint4` 合并访问,每 thread 一次 8 个 load:**144~146 GB/s ≈ 理论 153.6GB/s 的 95%**(G=32 TG × 256 thread 时最高,146.2GB/s)。
- **所以 §9 里"90% 效率≈138GB/s"偏保守**:纯读取上限约 **146GB/s**。decode 上限随之更新:每 token 5.657GB → 5.657/146 = 38.7ms → **25.8 tok/s**(未用 MTP);MTP 每步 6.386GB → 43.7ms,每步 1.6~1.8 个 token → **约 36.6~41 tok/s**(α=0.6~0.8 是假设)。真实 kernel 达不到纯读取的 95%,仍以 M0.2 的实测为准。

### 19.3 persistent megakernel 实测(**结论:本机上不如逐 phase dispatch**)

测试:96 个 phase,每 phase 流式读 16MiB(≈115µs,接近 decode 真实的每个矩阵的量级)。

**A. 全局屏障与共驻**(Metal 4.1 `atomic_fetch_add_explicit(..., memory_order_acq_rel, mem_flags::mem_device)` + 有界自旋)

| 每 TG 线程数 | 可共驻的 TG 数 | 说明 |
|---|---|---|
| 256 | **144 可以,160 失败** | = 12 核 × 12 个 TG |
| 1024 | **36 可以,48 失败** | = 12 核 × 3 个 TG |

→ **每个 GPU 核最多驻留 3072 个线程**(与 threadgroup 大小无关)。persistent kernel 的网格必须 ≤ 12 × (3072 / 每 TG 线程数),否则死锁(探测里靠有界自旋检测到;失败一次要耗 40~50 秒的自旋超时)。

屏障延迟(每次):T=256:12~36 个 TG 约 **1.4µs**,48 个 1.8µs,64 个 2.4µs,96 个 3.5µs,128 个 4.4µs,144 个 4.7µs;T=1024:12 个 TG **0.45µs**,24 个 0.53µs,36 个 0.67µs。→ **屏障成本随 TG 数线性增长**(约 30ns/TG,原子争用)。

**B. 依赖 dispatch 链**:微小 kernel(48 TG×256 线程)背靠背 100/400/1600 个,**每个 dispatch 只有 0.61µs**(GPU 端空隙;不含 CPU 编码时间,但 ICB/Metal 4 预编码可消除后者)。

**C. 带宽(1.61GB,16MiB/phase,各 G×T 配置)**

| 方案 | 带宽(GB/s)(G=32×256 / 64×256 / 128×256 / 16×1024) |
|---|---|
| V0 单个 dispatch,无屏障(上限) | 146.2 / 145.4 / 145.4 / 144.3 |
| **V1 每个 phase 一个 dispatch** | **146.5 / 146.5 / 145.9 / 145.5**(与 V0 持平,损失 <0.2%) |
| V2 persistent + 全局屏障 | 143.1 / 130.1 / 131.9 / 135.8(**慢 2%~11%**) |
| V3 persistent + 屏障前预取下一 phase 的前 16 个 load | 138.4 / 133.2 / 105.3 / 132.2(**更差**;G=128 时寄存器压力使占用率下降) |

**解读:**
1. 每个 dispatch 边界只有约 0.6µs,而且 16MiB 的 phase 之间**几乎不丢带宽**;每 token 约 330 个 dispatch × 0.6µs ≈ **0.2ms,约占 token 时间的 0.5%**。**§10.5 里"约省 5%~7%"的估计没有得到支持**,应下调为约 0.5%。
2. persistent kernel 的屏障(0.45~4.7µs)比 dispatch 边界(0.6µs)还贵,而且 V2 实测更慢(损失更像来自"等最慢的 TG"以及自旋访问同一 cache line,而不是屏障本身的延迟)。
3. 屏障前预取没有帮助:DRAM 已经被当前 phase 的 load 打满,屏障等待很短。
4. **结论:decode 主路线保持"每 kernel 一个 dispatch + ICB/Metal 4 预编码",不采用 persistent megakernel。** 融合 norm/FWHT 进 GEMV prologue 仍然值得做,但理由是减少显存往返和小 kernel 的尾部,而不是省 dispatch。
5. 用户明确要求使用 megakernel:**保留为可选实验**。设计 kernel 时把每个算子写成可复用的 "phase 函数",如果端到端 profile 显示 dispatch 间隙/尾部损失 ≥ 约 3%,再把相邻小算子拼成 persistent 段(只在"小算子密集"的区域,例如 GDN 的 conv→门控→递推→gated norm,且 G 取 12~36 个 1024 线程 TG,屏障 <0.7µs)。
6. 局限:phase 的工作量是均匀的;真实 kernel 有异构工作和跨 TG 的数据依赖,结论可能略有不同,需要用真实 kernel 复测。

### 19.4 "让 CPU 提前预处理下一个 token / 融合成一个大内核"

- **下一个 token 的 id 要等采样结束才知道**,所以不能提前处理;只有 draft token 是提前知道的,但它们的预处理(embedding 查表 1.1KB + 5120 宽反向 FWHT + 第 0 层的 norm/FWHT)总共只有微秒量级,在 GPU 上和主 kernel 融合更省(没有跨设备同步,同步延迟 ≥10~50µs)。**结论:不做 CPU 预处理,embedding 留在 GPU 步内。**
- 全部预计算(能提前做的)已在 `kata-pack` 清单里(§10.14):常量向量 `γ⊙d`、RoPE 表、权重重排等。
- "融合成一个大内核":见 19.3。

### 19.5 外部 drafter(Kata-2 2B)能否放 CPU/ANE 提前做

- 与 MTP 不同,**外部 drafter 不依赖主模型的 hidden state**,只依赖 token 序列,所以**可以并行**:在主模型 verify 第 n 步时,drafter 可以假设"全部被接受"继续往前起草(被拒绝就丢弃)。
- 规格(HF):`Ternary-Bonsai-2-2B-PTQ1_0.gguf` **0.43GB**,基础模型 `Qwen/Qwen3.5-2B`(18 层线性注意力 + 6 层 full attention);4B 另有 `PQ2_0-MTP-Q8_0`(1.27GB)和 **`DFlash2-BF16`(1.58GB,块 drafter)**。tokenizer 是否与 27B 完全一致**未验证**(同为 Qwen3.5 家族,大概率一致)。
- 瓶颈仍是 DRAM:每起草一个 token 要读 0.43GB。k=3 时每步多读约 1.3GB(+20%)。用 α_token=0.6 估计:每步产出 a+a²+a³ ≈ 1.18 个额外 token,即约 2.18 个/步;每 GB 产出 2.18/6.96 = 0.313 个 token,对比 MTP(1 个 draft,α=0.7)的 1.7/6.39 = 0.266 → **约 +18%,且依赖 α_token=0.6 这个假设**。
- **放 CPU 比放 ANE 合适**(不需要 Core ML;CPU 上 PTQ1_0 已有 NEON/i8mm 路径可参考 Prism #290;一个 token 读 0.43GB,在 CPU 可达带宽下约几毫秒)。**ANE 路线不值得**(要手写 Core ML 模型)。
- **结论:作为 M4 之后的实验**(先做 MTP + CPU n-gram);评估项:α_token 的真实值、CPU 起草对主模型带宽的挤占、tokenizer 一致性。

### 19.6 CPU 矩阵协处理器(SME/AMX)

**特性(`sysctl hw.optional.arm.*`)**:`FEAT_SME`、`FEAT_SME2`、**`FEAT_SME2p1`**;`SME_F32F32`、`SME_B16F32`、`SME_F16F32`、`SME_I8I32`、`SME_I16I32`、`SME_BI32I32`、`FEAT_SME_F16F16`、`FEAT_SME_B16B16`、`FEAT_SME_F64F64`、`FEAT_SME_I16I64`、**`FEAT_SME_LUTv2`**、`FEAT_SME_F8F16`、`FEAT_SME_F8F32`;另有 `FEAT_I8MM`、`FEAT_BF16`、`FEAT_DotProd`、`FEAT_FP16`;`sme_max_svl_b = 64`(流式向量长度 512 bit)。`hw.perflevel0` = Super(2 核,L1d 128KB,L2 20MB),`perflevel1` = Performance(4 核,L1d 64KB),`perflevel2` = Efficiency(6 核,L1d 96KB,L2 8MB)。

**重要的工程事实:ACLE 的 `__arm_new("za")` 在 macOS 上会 SIGILL**(编译器生成的 ZA lazy-save 序言读 `TPIDR2_EL0`,macOS 不支持)。**必须用内联汇编**直接 `smstart`/`smopa`/`fmopa`/`smstop`(`smstart`/`smstop` 会清零 Z/P 寄存器,所以要在 `smstart` 之后再初始化,并把 v0–v31、p0–p15 声明为 clobber)。`docs/probes/sme_throughput_probe.cpp` 是可用的模板。

**实测(寄存器内循环,4 个独立外积/迭代,峰值上限,不含访存)**:

| 指令 | 1 线程 | 2/3/4/6 线程聚合(UI QoS) |
|---|---|---|
| `fmopa` f32 | 2.21 TFLOPS | 2.26 / 2.22 / 2.21 / 2.28 |
| `fmopa` f16→f32(widening) | 2.26 TFLOPS | 2.26 / 2.23 / 2.23 / 2.28 |
| `bfmopa` bf16→f32 | 2.26 TFLOPS | 2.26 / 2.24 / 2.23 / 2.29 |
| **`smopa` s8→s32** | **4.50 TOPS** | 4.52 / 4.42 / 4.49 / 4.58 |

- **聚合吞吐不随线程数增加**(同簇内矩阵单元是**共享**的,不是每核一个)。
- **有两个独立单元**:1 个 UI 线程 + 1 个后台线程同时跑 s8,UI 4.53 + BG 2.33 = **聚合 6.86 TOPS**;2 UI + 2 BG = 7.26。即 Super/P 簇共享的单元 ≈ 4.5 TOPS(int8)/2.26(fp16),E 簇的单元约为一半(约 2.3 TOPS int8 / 约 1.2 TFLOPS fp16)。**CPU 侧总计约 7 TOPS int8、约 3.5 TFLOPS fp16(峰值)**。(QoS 不保证线程落在哪个簇,"BG=E 簇"是推断。)
- 对比 GPU:fp16 约 17 TFLOPS、int8 约 36 TOPS → CPU 的峰值相当于 GPU 的约 **+20%**。真实 GEMM(要访存、tile 复用)按 60%~80% 估,**净增约 +10%~15% prefill**(**推算**)。
- 用法设想:prefill 里按输出维 N 切分,GPU 算约 88%,CPU 用 SME 的 f16→f32 `fmopa`(或 int8 `smopa`)算约 12%;CPU 本来就在展开权重(路径 A),展开出的 fp16 权重可以直接给 SME 用。代价:手写汇编 kernel、占用 CPU 带宽、与 GPU 争 DRAM、功耗/散热。
- SME2 的 `LUTI2/LUTI4`、`SME_LUTv2` 可加速 CPU 展开 **2-bit 打包**;base-3(每字节 5 个 trit)没有直接对应的查表指令,仍用 NEON。
- **决定:列为 M5 之后的可选加速(优先级:低于 GPU 路径、CPU 展开权重、n-gram 草稿)。**

### 19.7 ANE

**实测(私有类探测,`ane_private_class_probe.mm`,`dlopen("/System/Library/PrivateFrameworks/AppleNeuralEngine.framework/AppleNeuralEngine")` 成功)**:
- 存在:`_ANEInMemoryModel`、`_ANEInMemoryModelDescriptor`、`_ANEClient`、`_ANEModel`、`_ANERequest`、`_ANEIOSurfaceObject`、`_ANEDeviceInfo`、`_ANEProgramForEvaluation`、`_ANEBuffer`、`_ANEQoSMapper`、`ANEServicesLog`;`_ANECompiler` 类名不存在(编译可能在别的类里)。
- `_ANEDeviceInfo`:`hasANE = 1`、**`numANEs = 2`、`numANECores = 32`(= 2 个 ANE × 16 核,与用户表里的 "Dual 16-Core ANE" 一致)**、`aneArchitectureType = h18g`、`aneSubType = h18`、`aneBoardType = 784`。
- 注意:早期的探测程序用 `objc_msgSend` 而没 `#import <objc/message.h>` 导致编译错误,以及 `objc_copyClassList` 遍历导致段错误,改用 `objc_getClass` 按名字查就稳定。

**接入路径(不需要 Python)**:私有 `_ANEInMemoryModel` 接受 **MIL 文本**程序加权重 blob(社区的 ANE 逆向项目的做法,**我没有验证过在 macOS 27 上能否跑通**),避免了手写 Core ML protobuf。风险:私有 API 随系统更新可能失效。

**ANE 能在哪里加速(推理)**:
- decode:**不能**(ANE 不增加 DRAM 带宽,GPU 已经用到 95%)。
- prefill:FFN 占每层约 70% 的 FLOPs(267.4M / 约 380M)。若 ANE 持续吞吐为 r_a、GPU 为 r_g=17T,让 ANE 承担一部分,最佳加速 = 1 + r_a/r_g。受限因素是**内存**:ANE 需要自己格式的权重副本(2-bit 调色板化的整个 FFN 约 4.3GB,int4 约 8.6GB,int8 约 17GB),和 GPU 的 6.39GB 常驻副本同时存在,16GB 机器放不下全部;只能让**一部分层**(例如 ≤20 层)的 FFN 走 ANE,上限约 1.28 倍(= 1/(1−0.7×20/64))。
- 外部 drafter:见 19.5,不值得走 ANE。

**决定(用户:"尽量使用 ANE,如果能加速的话")——作为带明确止损条件的实验(M0.9)**:
1. 用 `_ANEInMemoryModel` 跑通一个最小的 MIL 矩阵乘(fp16,权重 blob)。
2. 测 5120→17408 的 token 数 512/1024 时的持续吞吐(fp16,若支持再测 int8/调色板化权重),并确认 17408 这样超过 16384 的输出维是否受限。
3. **与 GPU 同时运行**,测互相干扰和 DRAM 争用。
4. **通过条件**:ANE 并发持续吞吐 ≥ 约 6 TFLOPS 等效,且 GPU 吞吐下降 <10%。通过后才做"FFN 按 chunk-wave 流水到 ANE"的设计;不通过则记录并放弃。

### 19.8 针对这台硬件的调优方式:离线自动调优 `kata-tune`

- 全部形状已知(约 8 种 GEMV/GEMM shape × M 档位),所以**不靠手工猜,用自动调优器在本机扫出最优值并冻结为常量**:`kata-tune` 对每个 `(阶段, M 档位, N, K)` 扫描并生成 `config/tuned_m6.h`(`constexpr`)。
- prefill 的扫描维度:`matmul2d` 的 tile(TM、TN)、每 threadgroup 的 simdgroup 数、网格顺序(N 快/M 快/分块 swizzle)、权重 staging 布局(row-major/tile-major)、K 方向分块、操作数位置、fp16 路径 vs int8 路径。M 档位:{512, 1024, 2048, 4096}。
- decode 的扫描维度:每 simdgroup 的行数(4/5/8)、每 TG 的 simdgroup 数、TG 总数(取 12 的倍数,且 ≤ 12 × 3072/线程数)、load 向量宽度和在途 load 数、展开度、解包方案(float 管线 `floor` / LUT / activation-side LUT)。M 档位:{1, 2, 3, 4}。
- 已有的原型:`matmul2d_prefill_shape_probe.cpp`(prefill 扫描)和 `persistent_kernel_probe.cpp`(带宽扫描)。

### 19.9 对前面各节的更新

- §9.1:achievable DRAM 带宽 ≈ **146GB/s(95%)**;decode 上限 25.8 tok/s;MTP 约 36.6~41 tok/s。
- §10.5 和 §10.10:persistent megakernel 的收益**不成立**(见 19.3);"省 400 次 dispatch 约省 5%~7%"下调为约 0.5%。
- §10.12:CPU 新增可选角色"SME 协算"(优先级低)。
- §10.13:ANE 作为带止损的 M0.9 实验。
- §12:新增风险 R13:`kata-tune` 的扫描空间大,需要限定在真实形状和少量 M 档位上。

---

## 20. 增补 C(2026-10-23 深夜):tokenizer 核实、CPU/GPU 重叠的干扰实测、FWHT 融合的真实开销、持续吞吐

用户新指令:(1) 告知 Qwen3.5 与 Qwen3.8 的 tokenizer 相同(引用第三方文章);(2) **用预先规划好的调度,隐藏通信/同步/取数,让 CPU、GPU、ANE 都不空闲**;(3) 再次要求**把 norm/FWHT 融合进 GEMV prologue**;(4) cmake 已通过 brew 安装。

### 20.1 tokenizer 核实(不依赖第三方文章,直接比对主数据)

- HF 仓库 `vocab.json`、`merges.txt` 的 SHA-256 在 **Qwen3.5-2B / 9B / 27B 与 Qwen3.8-27B 之间完全一致**。
- `tokenizer.json`:Qwen3.5 全系列一致;Qwen3.8 的 `normalizer`、`pre_tokenizer`、`post_processor`、`decoder`、`model`(BPE)与 3.5 **逐项一致**,差异只有 **多了 7 个特殊 token(id 248070–248076:`<|audio_start|>`、`<|audio_end|>`、`<tts_pad>`、`<tts_text_bos>`、`<tts_text_eod>`、`<tts_text_bos_single>`、`<|audio_pad|>`)**。
- `tokenizer_config.json`:唯一差异是 **chat template**(3.5 为 7,756 字符,3.8 为 8,952 字符)。我们应使用 27B GGUF 里的 3.8 模板(§5.9 的"历史 thinking 被剥掉"的核实仍是 TODO,要对着 3.8 模板核实)。
- GGUF 层面:`prism-ml/Ternary-Bonsai-2-2B-gguf` 与 27B GGUF 的 `tokenizer.ggml.tokens`(248,320)、`token_type`、`merges`(247,587)的 FNV-1a 哈希**完全相同**(`bd0c60744edf9fb7`、`cbf29ce484222325`、`e157cff7e1d26575`)。2B 的结构:`qwen35`,24 层、hidden 2048、FFN 6144、`full_attention_interval = 4`(18 个 GDN + 6 个 attention)。
- **结论:2B 可以在 token id 层面直接作 27B 的外部 drafter**(prompt 用 27B 的模板渲染成 id 后喂给两者即可,drafter 不需要自己的模板)。
- 探测程序:`docs/probes/gguf_tokenizer_scan.swift`。(tokenizer.json 的结构 diff 用了一次性的 python 小脚本,不属于项目,结果已记录在此。)

### 20.2 CPU 与 GPU 同时工作的干扰(**最重要的新发现:推翻了 §18/§10.9 的"CPU 展开权重几乎免费重叠"**)

探测程序 `docs/probes/overlap_interference_probe.cpp`(`ov3`):GPU 持续跑 prefill 形状的 fp16 matmul(M=2048,N=17408,K=5120,30 个连续),CPU 同时用 NEON 把 PTQ1_0 展开成 fp16(写入 Metal-shared staging buffer)。CPU 展开函数在 `expand_block`(按 §5.6 的块布局,qs[0..15] 的每个字节产生 5 个 16 元素向量,qs[16..23] 产生 5 个 8 元素向量,qh 标量)。

**CPU 单独运行(展开吞吐)**:1 线程 **21.8 G weights/s**(写出 43.7GB/s);2/4/6 线程约 **29.2~29.9 G weights/s(写出约 59GB/s,CPU 写带宽上限)**。需求(层优先 chunk=1024 时)只有 6.5 G weights/s,**单个核就够**。

**GPU 基线**:30 个连续 matmul = 13.07~13.6 TFLOPS(见 20.4)。

**并发时 GPU 的相对吞吐(基线 ≈ 13.1 TFLOPS,有 ±10% 噪声)**:

| CPU 工作 | CPU 写出 | GPU 相对吞吐 |
|---|---|---|
| 纯 ALU 自旋(2 或 6 线程,无内存流量) | 0 | ≈100%(117%/114%,噪声内) |
| NEON 展开,**缓存内**(2 或 6 线程,不碰 DRAM) | 0 | ≈100%(114%/117%) |
| fp16 普通存储 | 13 GB/s | **69%~73%** |
| fp16 **非临时存储(`stnp`)** | 13 GB/s | **86%~90%** |
| int8 普通存储 | 6.5 GB/s | 94% |
| int8 普通存储 | 13 GB/s | 70%~75% |
| fp16 普通存储 | 24.5 GB/s | 47% |
| fp16 非临时存储 | 26 GB/s | 61% |
| fp16 非临时存储,2 线程 | 52 GB/s | 43% |

**解读:**
1. **纯 CPU 计算(包括 6 个核满负荷)不拖慢 GPU** → 不是功耗/频率共享,是**内存系统干扰**。
2. 干扰正比于 **CPU 写出的字节数/s**(int8 13GB/s 与 fp16 13GB/s 的降幅相同),**与权重个数无关**。
3. 降幅远大于 DRAM 带宽占比(26GB/s 只占 18% 的带宽却让 GPU 掉 39%)。我的解读(**推断,未证明**):混合读写让 DRAM 效率下降、延迟升高,而 GPU matmul 的复用依赖 SLC/L2,CPU 的写流还会冲刷缓存;非临时存储能减轻一半以上,但不能消除。
4. E 核(`QOS_CLASS_UTILITY`)与 P/Super 核的效果相同(QoS 不保证线程落在哪个簇)。
5. 注意:早期用 `__builtin_nontemporal_store` 的版本 CPU 只达到 0.72 G w/s(代码生成很差),结论无效;用内联汇编 `stnp q0,q1,[x]` 才有效。

### 20.3 FWHT/norm 融合的真实开销(**推翻了"融合进每个 GEMV 的 prologue"**)

探测程序 `docs/probes/fwht_prologue_probe.cpp`(`pro2`):流式读 36MiB 权重(gate+up 的量级),x 向量宽 5120。

| 方案 | 额外开销 |
|---|---|
| **融合进每个 GEMV TG 的 prologue**(朴素的 threadgroup-memory radix-2 FWHT:读 fp32 残差 → 求平方和 → ×γ⊙d → 5 个 1024 块的 10 级 FWHT → fp16) | **+40~+345µs / kernel,即 +19%~+70%**,TG 越多越差(每个 TG 都重复做,且 prologue 期间 DRAM 闲置) |
| 同上但先预取前 8 个权重 load 再做 prologue | 并未改善(+26%~+98%) |
| **独立的优化版 FWHT kernel**(5 个 TG,每个负责一个 1024 块;块内 2 级寄存器 + 5 级 `simd_shuffle_xor` + 1 次 threadgroup 转置 + 3 级寄存器) | **4.83µs / kernel**(含 0.6µs dispatch 间隙);与 CPU 参考相比最大绝对误差 0.0037(输出最大值 14.2,即 fp16 舍入) |
| 两个依赖 dispatch:[fwht5 → GEMV(x 预先在 device)] vs 只有 GEMV | +3~+59µs(噪声大,理论值约 5µs) |

**解读:**
- 融合的"节省"只是省一次 10KB(fp16)向量的读写——它在 L2 里,几乎不花钱;而融合的代价是**每个 TG 重复一遍 5 个块的 FWHT 且阻塞该 TG 的权重流**。即使把融合版 FWHT 优化到同样水平,每个 TG 仍要串行处理 5 个 1024 块(独立 kernel 是 5 个 TG 并行,每个 1 块),延迟仍高 3 倍以上。
- **决定:用一个专门的小 kernel 做"RMSNorm 缩放 + `γ⊙d` + FWHT + 转 fp16"**(rms 的平方和在该 kernel 里算);GEMV 的 prologue 只做"把 10KB 的 fp16 x′ 从 L2 搬进 threadgroup memory"。
- 成本估算:每层约 4 个 FWHT 边界(GDN/attention 输入 5120、`ssm_out`/`attn_output` 输入 6144、FFN 输入 5120、`ffn_down` 输入 17408)× 64 层 ≈ 256 次 × 约 5µs ≈ **1.3ms,约占 token 时间的 3%**(推算)。后续可通过合并相邻边界、把 17 块的 17408 宽 FWHT 并行化来压缩。
- 局限:朴素版的实现细节(TG memory 的 bank 冲突、除法/取模索引)可能让融合版比"理论最优的融合"差;但"优化版融合"的延迟下限仍高于独立 kernel(见上),所以结论稳健。另:本探测的 TG memory 用量 30KB,导致每核只能驻留 1 个 TG,M0 基线本身只有 77~135 GB/s。

### 20.4 持续吞吐(突发值不能直接用于规划)

- **DRAM 流式读取持续带宽**(13 秒,1.61GB 缓冲区反复读):**142.0~142.6 GB/s,13 秒内无下降**(突发 146)。decode 上限更新:**5.657GB / 142 = 39.8ms → 25.1 tok/s**;MTP(6.386GB/步,α=0.6~0.8)约 **35.6~40.0 tok/s**。
- **fp16 `matmul2d` 持续吞吐会降频**:3 个连续 matmul 约 17.3 TFLOPS → 10 个 15.9 → 30 个 15.5 → 100 个 14.6 → **300 个(约 8.7 秒)12.6 TFLOPS**。另一次测量 30 个 13.07~13.6。**规划 prefill 要用持续值 ≈ 12.6~13 TFLOPS,而不是 17**。
- **int8 `matmul2d` 持续稳定**:3 到 300 个连续 matmul 都是 **35.5~36.1 TOPS**(`docs/probes/sustained_matmul_probe.cpp`)。**持续状态下 int8 ≈ fp16 的 2.8 倍**(35.8 / 12.8)。
- 修订后的 prefill 估计(每 token 48.7 GFLOP,matmul 上限 → 乘 60%~80% 得实际,推算):

| 路径 | matmul 上限 | 实际估计 |
|---|---|---|
| A:fp16 展开(精确) | 13/0.0487 ≈ 267 tok/s | **约 160~215 tok/s** |
| B:int8 激活(有损,需质量门槛) | 35.8/0.0487 ≈ 735 tok/s | **约 440~590 tok/s** |

- 因为持续状态下 int8 是 fp16 的 2.8 倍,**"int8 激活能否通过质量门槛"成为 prefill 最大的单项杠杆**。
- 一个未测量的折中思路:激活用"两个 int8 切片"表示 16 位整数(`x ≈ hi·256 + lo`,每个 128 组一个 scale),做两次 int8 matmul;精度接近或优于 fp16 激活,速度约 fp16 的 1.4 倍(35.8/2/12.8),再扣 epilogue(每组要合并两个累加器)后可能只剩约 1.2 倍,收益有限,优先级低。

### 20.5 修订后的调度原则("让 CPU/GPU/ANE 都不空闲")

**原则:共享的稀缺资源是 DRAM 加缓存系统,不是计算单元。** "忙"的衡量标准是 **DRAM 和矩阵单元有没有空闲**,而不是每个单元都在跑。调度器要为每个阶段做一份**带宽/写突发账本**:

1. **decode**:DRAM 是瓶颈,GPU 独占;**其它单元只能做不碰 DRAM 的事**(n-gram 查找、token 流式输出、采样回读,工作集都在缓存里)。让 GPU 永不空等:整步预编码成 ICB;提前提交 2 个 command buffer;下一步的位置/token 由 GPU 自己更新;CPU 通过环形缓冲异步读 token;CPU 产生的 n-gram 候选放进"邮箱",GPU 在步首读取(有就用,没有就不等)。
2. **prefill(层优先)**:GPU 的矩阵单元是瓶颈;**不再让 CPU 展开权重作为默认**(20.2:CPU 写流让 GPU 掉 12%~40%)。默认改为 **GPU 端的展开 kernel,每层每个 prompt 只做一次**:读 83MB 打包权重 + 写 766MB fp16(按 142GB/s 估约 6ms/层,64 层约 0.37 秒/prompt,**推算,未实测**),串行放在该层 matmul 之前。对比:CPU 非临时存储展开以 13GB/s 重叠,GPU 掉约 12%,约 3.7 秒内损失约 0.45 秒,**与 GPU 端展开基本同量级,但更复杂**;只有在 CPU 写速率限制在 ≤6.5GB/s 且用 `stnp` 时才值得(GPU 损失约 5%~10%)。
3. **CPU 的不碰 DRAM 工作**(缓存内):分词、n-gram、chat template、detokenize、采样回读、调度/记账。**SME 协算**需要从 DRAM 流入 staging 权重,按 20.2 的规律它会拖慢 GPU,**净收益很可能为负**——降为"仅在实测并发干扰后再决定"。
4. **ANE**:同样要在 M0.9 里测**并发时对 GPU 的干扰**(它也要从 DRAM 流入权重)。
5. **启动/空闲阶段**:此时 GPU 不忙,可以让 SSD→DRAM 预载 prefix 快照、权重预热(page-in)、预热 pipeline。这类 DRAM 写入只在不与 GPU 计算重叠时才免费。
6. **隐藏 CPU↔GPU 同步**:同 §10.10(ICB、queue-ahead、GPU 端采样/位置更新、异步 token 环形缓冲)。dispatch 边界本身只有 0.6µs,不是问题;CPU↔GPU 的来回(≥10~50µs)才是要避免的。

### 20.6 其它

- **cmake 4.4.4 已安装**(`/opt/homebrew/bin/cmake`);**ninja 没有**(不需要,Prism 基线用 make 即可)。
- 还没做的、仍待用户放行:建项目骨架、编译 Prism llama.cpp 基线、M0。
- 修订前文:§10.5(FWHT:不再融合进 GEMV prologue,改独立优化 kernel,约 3% token 时间)、§10.9/§18.3(prefill 权重展开默认改为 GPU 端 kernel;CPU 展开降为备选)、§9(持续带宽 142GB/s、fp16 持续 12.6~13 TFLOPS、int8 持续 35.8 TOPS)、§19.6(SME 协算净收益存疑)。

---

## 21. 用户后来放进项目根目录的文件(均为用户消息的原文副本,不含额外信息)

| 文件 | 内容 | SHA-256 前 16 位 |
|---|---|---|
| `Yes, the tokenizers for Qwen 3.5 and Qwe.md` | 用户关于"Qwen3.5 与 Qwen3.8 tokenizer 相同"的消息,以及调度/预取、融合 norm/FWHT、cmake 已安装的指令(§20) | `481cfc2e7372796e` |
| `因为有目标模型的全部事实, 根据具体的硬件来优化, tile 形状:网格顺序:布.md` | 用户关于按硬件调优 tile/网格顺序/布局、persistent megakernel、CPU 预处理下一个 token、外部 drafter、SME/AMX、ANE 的指令(§19) | `2ec058fb54bb51df` |

这些文件没有被提交到 git(只提交了 `docs/`)。用户放入的 PDF、图片和笔记保持未跟踪状态。
