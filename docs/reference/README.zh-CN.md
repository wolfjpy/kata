# 参考源码

[English](README.md) · **简体中文**

这些文件**不参与构建**。它们是本引擎编写时所对照的上游实现副本,放在这里是为了让 `docs/` 里的数值对照无需联网即可复核。

| 文件 | 来源 | 许可 |
|---|---|---|
| `prism_qwen35_graph.cpp`、`prism_metal_*.metal`、`prism_metal_dequantize_ptq1_0.h`、`prism_ptq1_0_quant_dequant.c` | llama.cpp,Prism 分支(`PrismML-Eng/llama.cpp`,分支 `prism`,commit `b3ebc47`) | MIT,© ggml 作者与 Prism |
| `hf_transformers_modeling_qwen3_5.py` | HuggingFace `transformers`(`modeling_qwen3_5.py`) | Apache-2.0,© HuggingFace |

两种许可都允许再分发,前提是保留版权与许可声明;上面的表格加上各文件原有的头部注释即满足该要求。`src/` 与 `shaders/`
中的引擎是独立实现:这些文件仅作为数值规范被阅读,`build/kata` 不编译其中任何一行代码。
