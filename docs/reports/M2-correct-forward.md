# 里程碑 M1/M2:正确的完整前向(64 层 + MTP 之外的全部路径)

日期:2026-10-24

## 目标

用最直接的写法实现整个 Qwen3.8-27B(PTQ1_0 + Hadamard)的单 token 前向,并对 Prism llama.cpp 做逐层数值对照,先证明所有公式和格式细节都对,再做性能工作。

## 实现

- `src/gguf.{h,cpp}`:GGUF v3 读取器,整个文件 mmap 后用 `MTLBuffer(bytesNoCopy)` 零拷贝包成一个 buffer;张量按文件偏移绑定;从 GGUF 里读 Hadamard 符号向量。
- `src/gpu.{h,cpp}`:Metal 上下文(metal-cpp,离线编译的 `kata.metallib`)。
- `config/kata_m6.h`:编译期常量,加载时对照 GGUF 元数据校验。
- `shaders/`:
  - `ops.metal`:embedding(反量化 + `s⊙(Hz)`)、`k_rms_fwht`(RMSNorm/SwiGLU/头置换 + 符号 + 1024 点 FWHT 的四种模式)、`k_rmsnorm`、加法、argmax。
  - `gemv.metal`:参考质量的 PTQ1_0 与 bf16 矩阵向量乘(每行一个 simdgroup)。
  - `gdn.metal`:GDN 的 causal conv 和递推步(每个 V 头一个 threadgroup,每个线程持有状态的一行)。
  - `attn.metal`:attention 的 q/k RMSNorm + RoPE + KV 写入、流式 softmax 解码。
- `src/engine.{h,cpp}`:每个 token 一个 command buffer,按层编码;可选地记录每层残差和每层 block 输出用于对照。
- `src/main.cpp`:`kata --model M --ids 1,2,3 [--ref-dir D] [--ref-logits F] [--gen N]`。
- Makefile 对 shader 加了 `-fno-fast-math`,保证数值与参考可比。

## 结果

对照脚本:`tools/ref_logits` 导出参考,`build/kata` 逐层比较。

| 提示 | 位置 | 层间残差相对误差 | top-1 一致 | 最大 logit 差 |
|---|---|---|---|---|
| "The capital of France is"(5 tokens) | 5 | 1e-7 ~ 7e-5 | **5/5** | 0.0005 |
| 一段 72 token 的英文 | 72 | 7e-4 ~ 1.8e-3 | **72/72** | 0.0131 |

- 贪心续写与参考逐 token 完全一致(72-token 提示续写 9 个 token)。
- 长提示的残差误差较大(1e-3),原因是参考实现在批量 prefill 时走 fp16 的 `mul_mm`,而我们全程 fp32;这不是我们的错误。

## 当前速度(未优化,仅作起点)

- 每 token 约 0.45~0.8 秒(约 1.2~2 tok/s)。所有 kernel 都是参考质量:GEMV 每个 lane 逐元素解码、attention 逐位置顺序循环、prefill 是逐 token 的 decode。
- 基线(Prism llama.cpp):tg128 15.6 tok/s,pp512 208 tok/s。

## 下一步

M3:decode 优化 —— 先做一个微基准比较 PTQ1_0 GEMV 的几种写法(Prism 的 float 管线解码、重排布局、activation 侧 LUT),选定最优后替换;然后做每 kernel 的耗时剖析,消除小 kernel 的尾部和 attention/GDN 的顺序瓶颈。
