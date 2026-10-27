# Reference sources

**English** · [简体中文](README.zh-CN.md)

These files are **not** built. They are copies of the upstream implementations that this engine was written against,
kept here so that the numerical comparisons in `docs/` can be checked without fetching anything.

| file | origin | licence |
|---|---|---|
| `prism_qwen35_graph.cpp`, `prism_metal_*.metal`, `prism_metal_dequantize_ptq1_0.h`, `prism_ptq1_0_quant_dequant.c` | llama.cpp, Prism fork (`PrismML-Eng/llama.cpp`, branch `prism`, commit `b3ebc47`) | MIT, © the ggml authors and Prism |
| `hf_transformers_modeling_qwen3_5.py` | HuggingFace `transformers` (`modeling_qwen3_5.py`) | Apache-2.0, © HuggingFace |

Both licences permit redistribution provided the copyright and licence notices are retained; the notices above, plus the
header comments already present in each file, serve that purpose. The engine in `src/` and `shaders/` is an independent
implementation: these files were read as a specification for numerics, and no code from them is compiled into
`build/kata`.
