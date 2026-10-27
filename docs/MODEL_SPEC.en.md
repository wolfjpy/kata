# Model specification

**English** · [简体中文](MODEL_SPEC.md)

Verified against Prism llama.cpp and the HuggingFace implementation, written so that kernels can be written directly from
it.

Sources: `docs/reference/prism_qwen35_graph.cpp`, `prism_metal_gated_delta_net.metal`,
`prism_ptq1_0_quant_dequant.c`, and the reference tree `refs/llama.cpp-prism` (`src/llama-graph.cpp`,
`src/llama-model.cpp`, `ggml-cpu/ops.cpp`). Every formula follows the Prism implementation and was checked numerically
against per-layer dumps from `tools/ref_logits`; the verification status is given at the end of each section.

## 1. Overall flow, per layer

```
x0 = embed(token)                                   # §2
for il in 0..63:
    h = rmsnorm(x, attn_norm[il])                   # y = x * rsqrt(mean(x^2)+1e-6) * w   (GGUF stores the effective weight)
    a = GDN(h)  if il % 4 != 3  else  Attention(h) # §4 / §5
    x = x + a
    h = rmsnorm(x, post_attention_norm[il])
    f = ffn_down( silu(ffn_gate(h)) * ffn_up(h) )   # §6
    x = x + f
x = rmsnorm(x, output_norm); logits = lm_head(x)    # only for the positions that need it
```

The input `v` of every ternary matmul `W·v` is Hadamard-transformed first (§3). `ssm_alpha` / `ssm_beta` (bf16) are **not**
rotated: they consume the RMSNorm'd `h` directly.

## 2. Embedding (`token_embd`, PTQ1_0, rotated basis)

`z = dequant(row[token])` (5120 values, 40 blocks out of 6.2); `e = s5120 ⊙ (H z)`, i.e. **Hadamard first, then the
signs** — the opposite order from the linear layers, because the Hadamard transform is its own inverse and `s² = 1`. `H` is
applied independently to each 1024-wide block, five blocks in total.

## 3. Hadamard transform (input side of the linear layers)

For the input vector `v` of a ternary matrix (width W ∈ {5120, 6144, 17408}):

```
v' = H_blockdiag( s_W ⊙ v ) ,   H[r][c] = (-1)^popcount(r & c) / sqrt(1024)     # independent per 1024-wide block
y  = Wternary · v'                                                              # W = scale_group * trits
```

- `s_W` is one of the three segments of `prism.hadamard.sign_values` in the GGUF (5120 / 6144 / 17408), **shared by all
  layers**, concatenated in the order given by `sign_widths = [5120, 6144, 17408]`.
- The RMSNorm `γ` can be folded into `s_5120` as a single vector `γ⊙s` (precomputed offline).
- **The input of `ssm_out` needs a head-order permutation** (`prism.hadamard.gdn_v_grouped = true`): the GDN output uses
  the "tiled" V-head order `[hd=128, nk=16, rep=3]` (V head `j` uses K head `j % 16`), while the folded weights are laid out
  in the "grouped" order `[hd, rep, nk]`. So permute first, then apply the signs, then rotate:
  the value at `f_new = h + 128*(r + 3*k)` is `x_old[h + 128*(k + 16*r)]` for `h∈[0,128)`, `r∈[0,3)`, `k∈[0,16)`.

## 4. Gated DeltaNet layers (48 layers)

Input `h` (post-RMSNorm, 5120):

```
qkv = Wqkv · H(s⊙h)        # [10240] = q(16*128) | k(16*128) | v(48*128)
z   = Wgate · H(s⊙h)       # [6144]
beta_raw = ssm_beta · h    # [48], bf16 weights, not rotated
alpha    = ssm_alpha · h   # [48]
conv:  y[t,c] = sum_{i=0..3} ssm_conv1d[i,c] * xin[t-3+i, c]   (xin = qkv, causal, state keeps the last 3 inputs [3,10240]);   c = silu(y)
q = c[0:2048] (16 heads), k = c[2048:4096] (16 heads), v = c[4096:10240] (48 heads)
q, k: per-head L2 normalisation  x / max(||x||, 1e-6)
for each V head hv (0..47), with K head hk = hv % 16 (tiled):
    gate  g = ssm_a[hv] * softplus(alpha[hv] + ssm_dt.bias[hv])     # softplus(x) = x if x>20 else log(1+exp(x)); ssm_a is negative
    beta  = sigmoid(beta_raw[hv])
    S[hv] is an fp32 state of 128 (key) x 128 (value)
    S *= exp(g)
    kS[v]  = sum_key S[key][v] * k[hk][key]
    d[v]   = (v_in[hv][v] - kS[v]) * beta
    S[key][v] += k[hk][key] * d[v]
    o[v]   = sum_key S[key][v] * q[hk][key]  * (1/sqrt(128))
gated norm (per V head, 128 wide):  n = rmsnorm(o, ssm_norm[128]);   out = silu(z[hv]) * n
out has 6144 dimensions (tiled head order) --permute (§3)--> signs --> Hadamard --> ssm_out · → [5120]
```

State layout: Prism's Metal kernel stores the state transposed so that the value dimension is contiguous; our
implementation chooses its own layout, which is mathematically equivalent.

## 5. Full-attention layers (16 layers, `il % 4 == 3`)

```
Qfull = Wq · H(s⊙h)        # [12288]: per head, 24 interleaved [query(256) | gate(256)]
q = first 256 of each head;  gate = last 256 of each head
q = rmsnorm_per_head(q, q_norm[256]);   k = rmsnorm_per_head(Wk · .., k_norm[256]);   v = Wv · ..   # k,v: [4 heads, 256]
RoPE (IMROPE, equivalent to NEOX for text): rotate only the first 64 dims of each head, in pairs (i, i+32), i=0..31:
    theta_i = pos * 10000000^(-2*i/64);  (x_i, x_{i+32}) -> (x_i cos - x_{i+32} sin,  x_i sin + x_{i+32} cos)
attention: scale = 1/sqrt(256), causal, GQA (24 Q heads share 4 KV heads, 6 Q heads per group)
out = attn * sigmoid(gate)             # [24*256 = 6144]   (the HF config says output_gate_type=swish, but both HF and Prism implement sigmoid)
attn_out = Wo · H(s6144 ⊙ out)
```

## 6. FFN

`gate = Wgate · H(s5120⊙h)`, `up = Wup · H(s5120⊙h)` (both share the same transform); `m = silu(gate) ⊙ up`;
`ffn = Wdown · H(s17408 ⊙ m)` (17408 = 17 blocks of 1024).

## 7. PTQ1_0 weights (ternary + fp16 group scale)

See `docs/reference/prism_ptq1_0_quant_dequant.c` and `CHECKPOINT-001.md` §5.6. A row of K inputs is K/128 blocks of 28
bytes; the dequantised value is `(trit-1) * d`. Element order inside a block: the n-th trit of `qs[m]` maps to element
`n*16+m` (m<16); `qs[16+m]` maps to `80+n*8+m` (m<8); the n-th trit of `qh[h]` maps to `120+n*2+h`.

## 8. MTP layer (`blk.64`, Q8_0) — verified (M4)

The MTP head is **a complete full-attention block**, with Q8_0 weights (34 bytes per block: an fp16 scale plus 32 int8
values), **no Hadamard rotation**, and its own K/V cache (a single layer).

Row `j` of the MTP pairs "the token `x_j` at position `j`" with "the trunk hidden state `h_{j-1}` at position `j-1`" and
predicts the token at position `j+1`:

```
e   = embedding(x_j)                       # same as the trunk: s ⊙ (H z), the inverse of the rotated basis
h   = output_norm(trunk residual at j-1)   # the "flat" hidden state after the last layer and after output_norm (including γ)
cat = [ RMSNorm(e) * nextn.enorm | RMSNorm(h) * nextn.hnorm ]      # 10240
x   = eh_proj · cat                        # Q8_0, 10240 -> 5120
a   = RMSNorm(x) * attn_norm               # plain RMSNorm, no sign / FWHT
q,k,v = Q8 matmuls of a                    # same [query|gate] layout as the trunk attention, per-head q/k norm, partial RoPE (position j), sigmoid gate
y   = x + attn_out(Q8, 6144 -> 5120)
f   = RMSNorm(y) * post_attention_norm
z   = y + ffn_down(silu(ffn_gate f) ⊙ ffn_up f)                    # Q8_0, 5120 -> 17408 -> 5120
logits = lm_head( shared_head_norm(z) )    # shares output.weight with the trunk (rotated basis), so sign + FWHT here
```

Key points:

- Position 0 has no MTP row (there is no `h_{-1}`), so the first valid MTP cache position is 1 (`firstPos = 1`).
- During prefill only K/V is needed for the MTP cache (no FFN, no logits); we compute it in bulk with three GEMMs —
  `eh_proj`, `wk`, `wv` (`Engine::encMtpKv`).
- Depth-1 drafting: `next` (sampled, not yet fed) together with `h_{n-1}` predicts the token at position `n+1`.
- Numerical check: on the model's own greedy continuation the MTP top-1 hit rate is 62-65% (a wrong formula would give
  ~5%).

## 9. Verification status

**M2 verified all of the above numerically** (`build/kata --ref-dir ... --ref-logits ...` against Prism llama.cpp's
per-layer dumps):

- The order in which the Hadamard matrix and the sign vectors are applied; the inverse transform for the embedding,
  `s ⊙ (H z)`.
- GDN: the tap order of the causal convolution (the oldest input goes with `w[0]`), the L2 normalisation of q/k
  (`x / max(||x||, 1e-6)`), the gating and the recurrence, the gated RMSNorm, and the tiled→grouped head permutation on
  the `ssm_out` input.
- Full attention: per-head RMSNorm of q/k, the interleaved `[query|gate]` layout, partial RoPE (first 64 dims, pairs
  `(i, i+32)`, `theta_i = pos * 1e7^(-2i/64)`), the `1/sqrt(256)` scale, and `sigmoid(gate)` (not swish).
- FFN and lm_head.
- A 5-token prompt: per-layer residual relative error 1e-7 to 1e-4, top-1 agreement 5/5, max logit difference 0.0005.
- A 72-token prompt: top-1 agreement 72/72, max logit difference 0.0131, and greedy continuation matching the reference
  token for token. The reference falls back to fp16 `mul_mm` on long batches while we stay in fp32 throughout, so where
  they differ it is our numbers that are more precise.
