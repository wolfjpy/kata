# CHECKPOINT-002:引擎已实现(M0–M6 + MTP + 预填充优化)

日期:2026-10-24 · 自包含的恢复文档。设计阶段的完整记录见 `CHECKPOINT-001.md`;每个里程碑的细节见 `reports/`。

## 1. 现状

项目 `bonsai-m6`:只用 Metal 的 Ternary Bonsai 2 27B(Qwen3.8-27B,`qwen35` 架构,PTQ1_0 Hadamard 折叠三值权重 + MTP 层)推理引擎,C++17 + MSL,目标机器 Mac mini M6 / 16 GB。零拷贝映射 GGUF,没有 Python。

| 指标 | 本引擎 | 基线(Prism llama.cpp,同机器) |
|---|---|---|
| 解码,普通 | 19.8 tok/s | 15.6 tok/s(tg128) |
| 解码,MTP 投机(贪心 / 采样) | **25.0 / 22.7 tok/s** | |
| 解码,MTP,7.3K 上下文 | 21.4–21.7 tok/s | |
| 预填充,精确 fp16,512 token | **245 tok/s** | 208 tok/s(pp512) |
| 预填充,精确 fp16,1024 token chunk | **257 tok/s** | |
| 预填充,精确 fp16,7.3K token 提示 | 213 tok/s | |
| 预填充,int8(`--fast`,不建议) | 270–283 tok/s | |

(GPU 持续负载下会降频,同一配置重复测可差 10–15%;对比 A/B 时必须交错运行。)

质量(都对照精确 fp32 路径):
- MTP 验证路径(fp16 激活 + half2 LUT GEMV):830 个位置,平均 KL 4.3e-6,最大 8.4e-4,top-1 100%。输出分布不变(确定性草稿的精确投机采样),但贪心输出偶尔在近似平局处分叉。
- 精确预填充:平均 KL 1.4e-7。
- int8 预填充:24.5K token 语料 1536 个位置,平均 KL 4.4e-4,**top-1 99.54%,未达 99.7% 门限**,所以默认关闭。

Git:`a6020e4`(M0)→ `4318bf0`(M2)→ `89af761`/`d97f58a`(M3)→ `fbf5ccb`(M5)→ `8d21717`(M6)→ `c435af5`/`f7f3796`(M4 MTP)→ `00264f7`(flash attention)→ `96e048d`(展开)→ `a3581ee`(int8)→ `53b04a8`(GEMM 顺序)→ `9d8fe1d`(flash-decoding)。

## 2. 源码地图

| 文件 | 作用 |
|---|---|
| `config/bonsai_m6.h` | 主机与 MSL 共用的常量(形状、分块);加载时与 GGUF 元数据对照 |
| `src/gguf.*` | GGUF 解析,整个文件 mmap |
| `src/gpu.*` | Metal 设备/队列/管线缓存(metal-cpp;`*_PRIVATE_IMPLEMENTATION` 只在这里) |
| `src/engine.*` | 前向:`step`(1 token 解码)、`prefill`(批量)、`verify2`(2 token 验证)、`mtpRows`(MTP 头)、`feed`(统一入口,维护 MTP 缓存)、`encMtpKv`、`dDecAttn` |
| `src/spec.*` | MTP 投机解码主循环(`specGenerate`),三缓冲状态回滚 |
| `src/tokenizer.*` `src/pretok.mm` | BPE 分词(ICU 正则;唯一的 Objective-C++ 文件),与 `llama-tokenize` 逐 token 一致 |
| `src/chat.*` | Qwen3.8 对话模板与采样器(top-k → top-p → min-p → 温度) |
| `src/cli.*` `src/main.cpp` | 子命令:`run`、`chat`、`tokenize`、`mtp-eval`、`verify-check`、`kl-eval`,以及无子命令的开发入口(逐层对照、KL、`--gen`、`--verify`、`--skip`) |
| `shaders/gemv_fast.metal` | 解码 GEMV(fp32 g/c 形式) |
| `shaders/mtp.metal` | Q8_0 GEMV/展开、MTP 输入、`k_gemv_lut2`(2 token LUT+half2)、`k_gdn_rec_v2`、`k_gdn_conv_v2`、`k_gdn_ab2` |
| `shaders/prefill.metal` | 展开(v2)、fp16/int8 GEMM(MPP `matmul2d`)、批量 GDN/RMSNorm/FWHT 内核 |
| `shaders/flash.metal` | 预填充 flash attention、解码 flash-decoding |
| `shaders/gemv_exp.metal` `layout_probe.metal` | 消融/探针内核(不被引擎使用) |
| `tools/` | `ref_logits`(Prism 参考)、`bench_gemv`/`bench_gemv2`、`alu_probe`、`probe_flash`、`kv_read_probe`、`layout_probe` |
| `docs/MODEL_SPEC.md` | 已验证的公式(含 §8 MTP) |

## 3. 构建与复现

```sh
make                                   # build/bonsai + build/bonsai.metallib
M=../../models/bonsai-2-27b-mtp-ptq1_0.gguf
build/bonsai run --model $M --prompt "..." --temp 0 --max-tokens 200         # MTP 默认开启;--no-spec 关闭
build/bonsai --model $M --ids ... --prefill --repeat 2 [--int8]              # 预填充计时(第一次包含 ~2 s 冷启动)
build/bonsai --model $M --ids ... --prefill --nlogits 72 --cmp-logits golden72.bin   # 对照保存的 fp32 logits
build/bonsai --model $M --ids ... --prefill --verify 20 [--skip a,b]         # 2 token 验证的计时与消融
build/bonsai verify-check --model $M --ids ...                               # 验证路径 vs 精确解码
build/bonsai kl-eval --model $M --file corpus.txt --tokens 24576             # int8 vs 精确预填充
build/bench_gemv2 $M build/bonsai.metallib k_gemv_lut2_nr4_sg4 ...           # GEMV 微基准
```

环境变量(开发用):`BONSAI_SKIP=a,b`(CLI 运行时的算子消融)、`BONSAI_GM`(GEMM 分组大小,默认 32)、`BONSAI_SCALAR_ATTN=1` / `BONSAI_SCALAR_DEC=1`(旧的标量注意力,用于 A/B)、`BONSAI_DEC_CHUNK`(解码 split 的 key 数)、`BONSAI_OLD_EXPAND`、`BONSAI_VERIFY_V1H`、`BONSAI_I8_EXCLUDE=down,wo,…`(int8 路径中走精确 fp16 的矩阵)。
测试用的 `/tmp/ids72.txt`、`/tmp/golden72.bin`、`/tmp/big1024.ids` 由 `build/ref_logits`(需要 Prism 参考运行时,见 `VERSIONS`)重新生成;它们不在仓库里。

## 4. 被实测推翻的假设和经验(每一条都有数据,见对应报告)

1. **解码 GEMV 是 ALU 受限,不是带宽受限**(M4 §3)。T=1 已到 126 GB/s(89%);T=2 用 fp32 g/c 形式只有 80 GB/s。`floor` 几乎免费(与前一条 FMA 融合),`fract` 不免费;half2 FMA 的元素吞吐是 fp32 的 ~1.9×;TG 内存里 2 字节的随机读是 LUT 方案的瓶颈,改成 16 字节向量读后 T=2 快 33%。
2. **持久化 megakernel 不需要**:dispatch 间隙 0.6 µs;全局屏障 0.45–4.7 µs,持久化反而慢 2–11%。
3. **GEMM 网格顺序影响最大**(M5b §2.2):一维线性 id + "M 优先分组"(GM=32)比原来的二维 dispatch 快 39%;一维 N 优先慢 3×。二维 dispatch 被硬件按局部性更好的顺序调度。chunk 超过 1024 行反而变慢(A 矩阵装不进缓存)。
4. **精确 fp16 GEMM 已贴近矩阵单元峰值**(16.6 TFLOPS)。int8 路径因 epilogue 和 7 位激活量化,优势只剩 ~10% 且质量不达标。
5. **展开与 GEMM 重叠会更慢**(展开的 178 MB 写入冲掉 GEMM 的缓存复用)。
6. **解码注意力内核已经在 KV 读取带宽的 ~65%**;K/V 布局(跨步 vs 按头连续)无差别;每个 key 每层 K+V 共 4 KB(每 token 64 KB)。长上下文只能靠压缩 KV。
7. **冷启动 ~2.3 s**(缺页 + 首次 dispatch);Metal 二进制归档和预触摸页都没有净收益。
8. **MPP/协作张量的规则**:张量元素类型不能带 `const`;尾块用真实 extent 时 MPP 对界外补零(避免 `NaN×0`);`cS.load(broadcast)` 结果错误,逐元素用 `get_multidimensional_index`;64×64/4 simdgroup 的目的张量布局固定(每 lane 4 行 × 2 个列块);`matmul2d` 的 int8 要用 `int8_t` 而不是 `char`。
9. **72 个位置的小测试会掩盖误差累积**,量化类改动必须在 ≥20K token 语料上用 `kl-eval` 评估。
10. GPU 持续负载下降频;A/B 对比必须交错、重复。

## 5. 未完成项(按预期收益排序)

| 项目 | 预期收益 | 说明 |
|---|---|---|
| 融合反量化的预填充 GEMM | 预填充 +10–14% | 展开仍占 0.56 s/1024 token;需要每个 M 块重复解码,大改 |
| KV 压缩(fp8/int8) | 32K 上下文解码 +20–30% | 有损,需要质量门限 |
| 验证 GEMV(T=2)从 113 → ~125 GB/s | MTP 周期 −5 ms(+8%) | LUT 读取仍是瓶颈;可试 bank 冲突/条目布局 |
| 草稿词表裁剪(只在高频子集上算 lm_head) | MTP 周期 −2 ms(+3%) | 有轻微降低接受率的风险 |
| 短提示(25–200 token)预填充 | 对话每轮延迟 0.9 s → ~0.3 s | 现在整套展开要 0.9 s;可做 T=8 的 LUT 验证式批处理 |
| GPU 端采样 | 很小 | 现在 CPU 读回 1 MB logits/token |
| 工具调用模板、`<think>` 预算 | 功能 | 未实现 |
| ANE / SME / Metal 4 ICB | 预计无收益 | 设计阶段分析的结论;dispatch 间隙已是 0.5% |

## 6. 需要用户确认的问题

1. **质量门限**:提议 KL ≤ 1e-3 nat、top-1 ≥ 99.7%(≥20K token)。按此 int8 预填充不通过(已默认关闭);MTP 验证路径通过(KL 4.3e-6)。是否接受 MTP 验证路径的 fp16/half 数值(贪心输出会偶尔在近似平局处与普通解码分叉)?
2. 是否要做 KV 压缩(有损)以换取长上下文解码速度?
3. 默认思考模式(现在开,`--no-think` 关)和思考预算。
4. 64K 上下文可以运行,但常驻内存 ~12.7 GB(权重 6.4 + 缓冲 6.3),贴着 16 GB 机器的上限;是否把 `--ctx` 上限限制到 48K?
