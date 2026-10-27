# M9: persistent prefix cache

**English** · [简体中文](M9-prefix-cache.md)

Date: 2026-10-25 · Corresponds to CHECKPOINT-003 roadmap item M9. Inspired by ninfer (a checkpoint is the KV plus the
complete continuation state) and Strata (checkpoints at message boundaries).

## 1. Results

| scenario | without cache | with cache |
|---|---|---|
| 2941-token prompt (2900-token system prompt + a short question), then a different question | prefill 15-16 s | restore 0.02-0.1 s + prefill only 21 tokens (**1.15 s** for the first chat turn, with a background warm-up hiding the cold start) |
| the same 3639-token raw prompt, run twice | prefill 18.6 s | restore 0.03 s, 0 tokens prefilled |
| extrapolation: 20K-token system prompt | ~80 s | checkpoint ~1.5 GB, restore ~0.1 s (warm page cache) to ~0.5 s (cold SSD read, not measured) + a few hundred trailing tokens |

Correctness (`tools/test_cache.sh`, all passing):
- the restored output is **byte-identical** to an uncached full prefill, for greedy decoding and for fixed-seed sampling;
- the run that wrote the checkpoint matches the uncached run;
- a full-prompt hit (`n == L`, using the stored logits directly) matches the run that wrote it;
- the int8 path never uses an fp16 checkpoint (each has its own file);
- a truncated checkpoint is ignored and rewritten.

## 2. What is in a checkpoint

A checkpoint is the engine's **complete continuation state** at an exact token frontier `n`:
- GDN state 151 MB + conv state 5.9 MB;
- K and V of the 16 attention layers for rows [0, n) (64 KB per token);
- the MTP head's K/V rows for [0, n) (4 KB per token) and the last position's hidden state `hPrev` (20 KB);
- the last position's logits (1 MB, used to sample directly on a full-prompt hit) and the greedy token;
- the full token sequence (compared token by token on restore; no hashing shortcut).

Size is roughly 157 MB + 68 KB x n (0.36 GB at 2.9K tokens; ~1.5 GB at 20K; ~4.5 GB at 64K).

File format (`src/ckpt.cpp`): a 4 KiB header (magic `BNSCKPT1`, version, model fingerprint, numerics tag, `n`, two
independent FNV-1a 64-bit prefix hashes, segment offsets) + the token sequence + the segment data (4 KiB aligned). The
file name is the prefix hash + numerics tag/model hash + `n`, so different numerics modes and different models can never
collide. **KV rows are read and written straight into the MTLBuffer's shared memory with `pread`/`pwrite` — no
intermediate copy.**

## 3. Policy

- **Where to checkpoint** (`PrefixCache::plan`): (1) the start of the last user message (the last `<|im_start|>` followed
  by `user`), provided at least `--cache-min` tokens precede it; (2) the end of the prompt, when it is at least 256 tokens
  away from (1). (2) matters for agent loops: the next call appends new tool output, but the history in front of it does
  not have to be re-read. With no message boundary available (raw text, for example), every 8192 tokens plus the end.
- **How to find one**: on restore, scan the directory, read each header, compute the prefix hash at every candidate
  length, and take the longest of those whose hash matches, whose token sequence is identical token by token, whose
  model and numerics tags agree, and which holds MTP data when speculation is enabled.
- **Asynchronous write**: the small state that later decoding would modify (GDN/conv/logits/hPrev, ~157 MB) is copied to
  the heap first; KV rows are append-only (rows below `n` never change), so the writer thread reads engine memory
  directly. Process exit and `/reset` wait for the writer. Writes are atomic (`.tmp` + `rename`).
- **Budget**: 40 GB by default, least-recently-used (file mtime, refreshed on a hit) evicted first, never the one just
  written.
- Only used for the **first feed at position 0** (later turns in a chat continue the state in-process anyway).

## 4. Command line

| option | effect |
|---|---|
| `--no-cache` | disable (required for benchmarking) |
| `--cache-dir D` / `KATA_CACHE_DIR` | directory, default `~/.cache/kata` |
| `--cache-gb N` | budget, default 40 |
| `--cache-min N` | prefixes shorter than N tokens are not cached, default 2048 |
| `kata cache [--clear]` | list or clear |

**Privacy**: a checkpoint contains the KV of the prompt content, from which the gist can be recovered. For sensitive
prompts use `--no-cache`, or `kata cache --clear` afterwards.

## 5. Known limits and follow-ups

- The "message boundary" and "end of prompt" checkpoints of the same prompt each store a full copy of the KV (~2x
  redundant). A **chained, segmented store** would fix that: each checkpoint keeps only the KV added since the previous
  one plus a state, and restores by walking the chain; LRU would then have to evict whole chains.
- The 0.02-0.12 s restore time is a page-cache hit; a cold SSD read was not measured.
- The first GPU use after a restore still pays ~3 s of cold start (paging the weights in). Chat mode hides it with a
  background warm-up; a one-shot `run` cannot.
- Only the prompt is cached, not the generated tokens (agent frameworks re-render history and drop earlier thinking
  content, so generated KV would not line up anyway).
- The numerics tag currently only covers fp16/int8 and the exclusion set; later milestones (int8 KV, the FFN int8
  switch) must add their mode to `Engine::numericsTag()`, and bump `kVersion` whenever they change the numerics of a
  state.
