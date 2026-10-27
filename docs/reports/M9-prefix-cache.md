> 本文以中文写成;下方 `## Summary (EN)` 是英文摘要,双语文档索引见 [docs/README.md](../README.md)。
> *This document is in Chinese. An English summary follows; the bilingual index is at [docs/README.md](../README.md).*

## Summary (EN)

**Goal.** A persistent prefix cache: checkpoint = KV plus the complete continuation state. The idea comes from ninfer (checkpoints of KV + full state) and Strata (checkpoints at message boundaries).

**Result.**

| scenario | without cache | with cache |
|---|---|---|
| 2941-token prompt (2900-token system prompt + short question), then another question | prefill 15-16 s | restore 0.02-0.1 s + prefill only 21 tokens (1.15 s first chat turn, with a background warm-up hiding the cold start) |
| the same 3639-token raw prompt twice | prefill 18.6 s | restore 0.03 s, 0 tokens prefilled |
| extrapolation: 20K-token system prompt | ~80 s | checkpoint ~1.5 GB, restore ~0.1 s (warm page cache) to ~0.5 s (cold SSD read, not measured) + a few hundred tokens |

**Correctness** (`tools/test_cache.sh`, all passing): the restored output is **byte-identical** to a full prefill, for greedy and for seeded sampling; the run that wrote the checkpoint matched the uncached run; a full-prompt hit uses the saved logits; int8 and fp16 never share a checkpoint; a truncated checkpoint is ignored and rewritten.

---

# M9:持久化前缀缓存

日期:2026-10-25 · 对应 CHECKPOINT-003 路线图 M9。灵感来自 ninfer(检查点 = KV + 完整续写状态)和 Strata(消息边界检查点)。

## 1. 结果

| 场景 | 无缓存 | 有缓存 |
|---|---|---|
| 2941 token 提示(2900 token 系统提示 + 短问题),换一个问题再跑 | 预填充 15–16 s | 恢复 0.02–0.1 s + 只预填充 21 个 token(聊天模式首轮 **1.15 s**,后台热身已隐藏冷启动) |
| 3639 token 的原始提示,完全相同再跑一次 | 预填充 18.6 s | 恢复 0.03 s,预填充 0 token |
| 外推:20K token 系统提示 | ~80 s | 检查点 ≈ 1.5 GB,恢复约 0.1 s(页缓存热)到 ~0.5 s(SSD 冷读,未实测) + 尾部几百个 token |

正确性(`tools/test_cache.sh`,全部通过):
- 恢复后的输出与无缓存的完整预填充**逐字节一致**(贪心、固定种子采样都一致);
- 写入检查点的那次运行本身与无缓存一致;
- 整个提示命中(`n == L`,直接使用保存的 logits)与写入它的那次一致;
- int8 路径不会使用 fp16 的检查点(各自有自己的文件);
- 被截断的检查点会被忽略并重写。

## 2. 检查点里有什么

检查点 = 引擎在某个精确 token 前沿 `n` 的**完整续写状态**:
- GDN 状态 151 MB + conv 状态 5.9 MB;
- 16 个注意力层 [0, n) 行的 K 和 V(每 token 64 KB);
- MTP 头的 K/V 行 [0, n)(每 token 4 KB)和最后一个位置的隐状态 `hPrev`(20 KB);
- 最后一个位置的 logits(1 MB,整个提示命中时用来直接采样)和贪心 token;
- 完整的 token 序列(恢复时逐 token 比对,不依赖哈希)。

大小 ≈ 157 MB + 68 KB × n(2.9K token 0.36 GB;20K ≈ 1.5 GB;64K ≈ 4.5 GB)。

文件格式(`src/ckpt.cpp`):4 KiB 头(魔数 `BNSCKPT1`、版本、模型指纹、数值标签、`n`、两个独立的 FNV-1a 64 位前缀哈希、各段偏移)+ token 序列 + 各段数据(4 KiB 对齐)。文件名 = 前缀哈希 + 数值标签/模型哈希 + `n`,所以不同数值模式、不同模型的路径永不冲突。**KV 行直接用 `pread`/`pwrite` 读写 MTLBuffer 的共享内存,没有中间拷贝。**

## 3. 策略

- **在哪里存**(`PrefixCache::plan`):① 最后一条用户消息的起点(最后一个 `<|im_start|>` 后面紧跟 `user` 的位置,要求 ≥ `--cache-min` 个 token);② 提示结尾(与①相距 ≥ 256 token 时)。②对 agent 循环很重要:下一次调用会带上新的工具输出,但前面的历史不用再读。没有可用的消息边界时(例如纯原始文本),每 8192 个 token 存一个,再加结尾。
- **怎么找**:恢复时扫描目录读各文件头,对提示计算每个长度的前缀哈希,取"哈希匹配 + token 序列逐个相同 + 模型/数值标签一致 + 有 MTP 数据(若本次开启投机)"里最长的那个。
- **异步写**:先把会被后续解码修改的小状态(GDN/conv/logits/hPrev,约 157 MB)拷到堆上,KV 行是追加写入的(< n 的行不再变化),所以写线程直接读引擎内存;进程退出和 `/reset` 前会等待写线程。原子写(`.tmp` + `rename`)。
- **预算**:默认 40 GB,最近最少使用(文件 mtime,命中时刷新)先删,刚写入的永不删。
- 只在**位置 0 的第一次喂入**使用(聊天里后面的轮次本来就在进程内续用状态)。

## 4. 命令行

| 选项 | 作用 |
|---|---|
| `--no-cache` | 关闭(基准测试时要用) |
| `--cache-dir D` / `KATA_CACHE_DIR` | 目录,默认 `~/.cache/kata` |
| `--cache-gb N` | 预算,默认 40 |
| `--cache-min N` | 少于 N 个 token 的前缀不缓存,默认 2048 |
| `kata cache [--clear]` | 列出或清空 |

**隐私**:检查点里有提示内容的 KV,可以被还原出大意;敏感提示请用 `--no-cache` 或事后 `kata cache --clear`。

## 5. 已知限制和后续

- 同一个提示的"消息边界"和"结尾"两个检查点各存一份完整 KV(冗余 ~2×)。后续可以做**分段链式存储**:每个检查点只存自上一个检查点以来的 KV 增量 + 一份状态,恢复时读链;LRU 要按链回收。
- 恢复耗时 0.02–0.12 s 是页缓存命中;SSD 冷读未实测。
- 恢复后第一次 GPU 使用仍要付 ~3 s 的冷启动(权重锁页)。聊天模式的后台热身能隐藏;一次性的 `run` 不能。
- 只缓存提示,不缓存生成的 token(agent 框架重新渲染历史时会去掉旧的思考内容,生成的 KV 本来就对不上)。
- 数值标签目前只有 fp16/int8 + 排除集;后面的里程碑(int8 KV、FFN int8 开关)必须把自己的模式加进 `Engine::numericsTag()`,并在改变状态数值时提升 `kVersion`。
