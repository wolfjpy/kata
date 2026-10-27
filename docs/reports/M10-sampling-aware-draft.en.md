# M10: sampling-aware drafts — offline estimate only, gate not met, implementation deferred

**English** · [简体中文](M10-sampling-aware-draft.md)

Date: 2026-10-25 · Corresponds to CHECKPOINT-003 roadmap item M10.

## 1. The idea

In sampled mode (default: temperature 1.0, top-k 20, top-p 0.95, min-p 0.05) the current speculative scheme is: the MTP
head produces a **deterministic** draft `d = argmax q`, the target distribution `p` draws `s0`, and `s0 == d` accepts, so
the acceptance probability is `p(d)`. ninfer's DFlash2 contract describes another way: **sample** the draft from its own
distribution `q`, accept with `min(1, p(d)/q(d))`, and on rejection resample from the residual `max(p-q, 0)`. Both preserve
the output distribution, but the latter's acceptance probability is `sum_v min(p(v), q(v)) = 1 - TV(p, q)`, which is far
above `p(argmax q)` when q is close to p.

## 2. Offline estimate (`kata mtp-accept`)

The model generates 200 tokens with the real sampler, and at every position both the MTP's `q` and the target's `p` are
computed (each passed through top-k/top-p/min-p/temperature), comparing the two acceptance rates:

| text | deterministic draft | sampled from q | top-1 agreement |
|---|---|---|---|
| prose (no thinking) | 53.6% | 60.5% | 60.5% |
| code (no thinking) | 85.2% | 86.5% | 87.0% |
| reasoning (thinking) | 71.9% | 75.7% | 75.0% |
| explanation (no thinking) | 58.2% | 67.5% | 62.0% |
| **all (800 positions)** | **67.2%** | **72.5%** | 71.1% |
| temperature 0.6, all | 70.1% | 72.6% | 72.4% |

Tokens per cycle 1.67 -> 1.73 (+3.6%); after subtracting the two extra CPU samplings per cycle (about 0.1-0.3 ms with the
single-pass top-K), the net gain is **+2-3%**, and only in sampled mode.

## 3. Conclusion

The gate (a >= 9 point gain in sampled-mode acceptance) **was not met**: only +5 points (prose +7, code +1). The +2-3% is
above the 1% keep line but below the other milestones, so it is **not implemented for now** and is left as an option
(M17).

If it is implemented later it needs: (1) the draft step in `spec.cpp` to obtain `q` via `samplerDistribution` and sample
`d` from it (the MTP logits are already in `eng.logits()`); (2) the verification step to test `min(1, p0(d)/q(d))` and,
on rejection, resample over `supp(p) ∪ supp(q)` from `max(p-q, 0)`; (3) the second token after acceptance sampled from
`p1` as before; (4) a GPU-free unit test of the distribution: with synthetic `(p, q)`, a million repetitions must produce
an empirical output distribution matching `p` (chi-squared test).

## 4. Incidental changes (merged)

- `samplerDistribution` / `sampleFromDist`: the sampler returns a sparse distribution; `sampleToken` is bit-identical
  (fixed-seed output is character-for-character the same as before the refactor, verified for greedy and sampled, with
  speculation on and off).
- Top-K became a single pass (for `K <= 64`), no longer a `partial_sort` over 248K indices: O(n) and about 0.1 ms on a P
  core.
- Observation: the CPU sampling time jumps randomly between 0.1 and 0.9 ms per cycle, most likely because the thread gets
  scheduled onto an efficiency core. M17 can try `pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE)` to see
  whether it can be pinned to a P core.
