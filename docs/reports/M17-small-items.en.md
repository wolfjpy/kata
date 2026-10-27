# M17: small items, measured one by one, none cleared the 1% keep line

**English** · [简体中文](M17-small-items.md)

Date: 2026-10-25 · Corresponds to CHECKPOINT-003 roadmap item M17 (rule: keep only what is worth >= 1%). **Verdict: none
of them were implemented**, and the reasons and data are below.

| item | measurement / estimate | verdict |
|---|---|---|
| Main-thread QoS (`QOS_CLASS_USER_INTERACTIVE`), on the guess that the CPU sampler's 0.1-0.9 ms jitter came from being scheduled onto an efficiency core | four interleaved rounds: 22.5/22.5, 22.6/21.9, 21.6/21.9, 22.3/22.5 tok/s | no effect, reverted |
| A NEON version of top-K sampling (16 logits per comparison) | sampling 0.2-0.8 -> 0.1 ms per cycle, bit-identical output; four interleaved end-to-end rounds: 7.01/6.93, 6.95/6.96, 6.89/6.97, 6.90/6.94 s | inside the noise end to end, reverted (M10's single-pass top-K kept) |
| Draft and verify in one command buffer (the draft token handed to verification by the GPU, so verification's ~0.8 ms of CPU encoding overlaps the draft's GPU execution) | estimated ~1.1 ms saved per cycle (GPU idle: 0.8 ms of encoding + 0.3 ms of submit latency), about 1.7%; implementing it needs async `mtpRows`/`verify2`, a draft-result buffer and a GPU-side token copy | below the keep line on value/complexity, deferred |
| A vocabulary subset for the draft `lm_head` (ninfer's 131072 rows) | saves ~0.9 ms per cycle (1.4%); needs a token-frequency ranking, which the repository does not have | deferred |
| Suffix drafter + online cost policy (Strata) | the T=2 verification window has only one draft slot; MTP already accepts 85%, and suffix only lifts 65% to 95% on long matches (+3-9% on repeated text, ~0 on ordinary conversation); multi-token drafts need a T>=3 verification kernel (T=4 is 60% more expensive than T=2, and M14's wide kernel is actually slower below T=6) | meaningless without wide verification, deferred |

A data point worth remembering: host-side tricks like CPU sampling and QoS all drown in the noise of a 62-66 ms cycle. The
real remaining time is ~51 ms of GPU verification GEMV (113 GB/s) and ~6 ms of drafting (the MTP layer plus 0.73 GB of
lm_head reads).
