# M12: weight layout (16-byte-wide reads) — gate not met, not implemented

**English** · [简体中文](M12-weight-layout.md)

Date: 2026-10-25 · Corresponds to CHECKPOINT-003 roadmap item M12 (originally: a `kata-pack` tool producing a split-plane
file, expected to give +8-10% decode).

## 1. Premise and measurement

The premise came from `tools/weight_read_probe`: **with no compute at all, reads only**, the small reads of the native
28-byte block layout (2/1/1/2 bytes) reach 124-132 GB/s, while the same byte count as 16-byte vector reads reaches
141-142 GB/s. ninfer's split-plane weight layout and Prism CUDA's planar activation layout follow the same idea.

This round first broke it down by access width (contiguous across lanes, normalised to the bytes actually read):

| access width per lane | bandwidth |
|---|---|
| native (2/1/1/2 bytes) | 124-132 GB/s |
| 4 bytes | 127 |
| 8 bytes | 135 |
| 12 bytes (three adjacent 32-bit) | 134-138 |
| 16 bytes | 138-139 |
| whole-row 16-byte vectors | 140-142 |

And it turned out that **no file change is needed**: every row is 16-byte aligned (row lengths 1120/1344/3808 bytes) and
a "4-block group" (112 bytes = 7 vectors) is aligned inside the row. So a simdgroup can cooperatively pull 4 rows x 112
bytes into threadgroup memory with 16-byte reads and let the lanes read the small fields from there — 16x fewer global
loads, no format change at all.

## 2. Prototype result

`docs/probes/gemv_staged_prototype.metal` (`k_gemv_v1s`, software pipelined: issue the next quad's global read, then
compute the current quad):

| shape | `k_gemv_v1` with native reads | `k_gemv_v1s` with staged reads |
|---|---|---|
| ffn_gate/up | 131.4 GB/s | 130.1 |
| ffn_down | 127.4 | 126.0 |
| gdn_qkv | 129.6 | 126.2 |
| lm_head | 134.2 | 134.6 |

**No gain.** Note: this prototype's **results are wrong** (max deviation ~1e6, still wrong after fixing the threadgroup
array alignment, cause not found); but the instruction mix matches the original (same decode, same FMAs, reads turned into
small threadgroup reads), so the timing is still representative.

## 3. Conclusion

Once the decode compute is inside the GEMV, read width is not the limiting factor: 16x fewer global load instructions and
16-byte contiguous reads change nothing. This agrees with the earlier ablations (removing the decode or the activation
reads changed things by only 3-7%): the GEMV sits at a balance point where the decode ALU, the reads and latency do not
fully overlap, so improving any one of them alone buys nothing — the pure-read probe's 141 GB/s does not carry over. At T=1
the GEMV is already at 89-94% of sustained bandwidth (ffn_gate 92%, ffn_down 89%, qkv 92%, lm_head 94%), leaving a ceiling
of <= 8% that would require restructuring the decode/read overlap at once — high cost, high uncertainty.

So **M12 was not implemented** (no `kata-pack` was written, no engine code or weight file was touched). This contradicts
CHECKPOINT-003's earlier judgement that "the 8-10% read gap on the native 28-byte block can only be recovered by
re-layout", which is hereby corrected.

## 4. Incidental changes

- `tools/weight_read_probe` gained a per-access-width measurement (and its byte-count normalisation was fixed).
- `shaders/layout_probe.metal` gained `k_read_w4/8/12/16`.
- If this is ever retried, the more promising direction is a T=1 LUT + half2 (three half2 FMAs plus one 16-byte LUT read
  per byte, replacing today's one conversion + five multiply-floors + five FMAs), which reduces the ALU instruction count
  rather than only touching the reads.
