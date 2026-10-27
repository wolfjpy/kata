# M13: int8 prefill for the FFN gate+up (`--fast-ffn`, lossy, off by default)

**English** · [简体中文](M13-fast-ffn.md)

Date: 2026-10-25 · Corresponds to CHECKPOINT-003 roadmap item M13, and to one of the decisions the owner confirmed.

## 1. Results

| | exact fp16 (default) | `--fast-ffn` (gate+up int8) | `--fast` (all int8) |
|---|---|---|---|
| prefill, 1024-token chunk | 255-263 tok/s | **274-280 tok/s (+7-8%)** | 281 tok/s (+9-10%) |
| prefill, 7.3K-token prompt | 207 tok/s | **226.5 tok/s (+9.5%)** | 229 tok/s (+10.7%) |
| quality (24.5K-token corpus, against the exact path) | — | **mean KL 7.5e-5, top-1 99.87%** (3068/3072), max KL 0.039 | mean KL 4.4e-4, top-1 99.54% |
| passes the gate (KL <= 1e-3, top-1 >= 99.7%) | — | **yes, with margin** | no |

The FFN `gate` and `up` matrices are ~47% of the per-layer FLOPs, so making only them int8 captures about 80% of the
gain of making everything int8 while making the error an order of magnitude smaller (6x lower mean KL, 9x lower max
KL). The reasoning comes from the Pareto front in CHECKPOINT-003 §3.5: the error is dominated by the input projections
that feed the GDN recurrent state (`qkv`, `z`, `wq`), whereas the `gate`/`up` outputs still go through SwiGLU and the
`down` projection, which makes them insensitive to activation quantisation noise.

## 2. Implementation

- `Engine::setI8(spec)` accepts `"all"` (every matrix), `"ffn"` (gate+up), or a comma-separated list
  (`qkv z ssmout wq wk wv wo gate up down`); an empty string or `none` turns it off. Each `dExpandGemm` call decides
  between int8 and exact fp16 by matrix name; the remaining matrices are unaffected.
- **Activation quantisation only runs for GEMMs that will actually consume int8**: `fusedNorm` / `fwhtB` now take a
  `needQ` flag and only run `k_quant_rows` when one of that activation's consumers is an int8 matrix (with
  `--fast-ffn` that is only the normalisation right after attention).
- The numerics tag carries the matrix set (`int8:gate+up`), so prefix-cache checkpoints from different modes never mix.
- Command line: `--fast-ffn`, `--fast` (everything), `--i8 a,b,c` (development) and `--exact` for `run`/`chat`;
  `--fast-ffn` / `--i8` for `kl-eval`; `--i8` for the development entry point. The previously undocumented
  `KATA_I8_EXCLUDE` environment variable is gone.

## 3. Caveats

- Lossy: prefill multiplies int8 activations (one scale per token per 128-element group) by the ternary weights. The
  error only enters the `gate`/`up` outputs, and from there goes through SwiGLU and the `down` projection into the
  residual stream. The KV and the GDN state are produced under this path, so the effect carries into decoding — which is
  why the evaluation uses a 24.5K-token corpus rather than a few dozen positions.
- Prefill only; the decode path is unchanged.
- On prompts shorter than ~7K tokens the gain is diluted by the fixed cost of expanding the weights (0.56 s per chunk).
