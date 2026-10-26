# 模型规格(从 Prism llama.cpp 与 HF 的实现中核实,可直接据此写内核)

来源:`docs/reference/prism_qwen35_graph.cpp`、`prism_metal_gated_delta_net.metal`、`prism_ptq1_0_quant_dequant.c`,以及参考仓库 `refs/llama.cpp-prism`(`src/llama-graph.cpp`、`src/llama-model.cpp`、`ggml-cpu/ops.cpp`)。
所有公式都以 Prism 的实现为准,并用 `tools/ref_logits`(导出每层输出)做数值对照验证(验证状态见各节末尾)。

## 1. 整体流程(每层)

```
x0 = embed(token)                                   # §2
for il in 0..63:
    h = rmsnorm(x, attn_norm[il])                   # y = x * rsqrt(mean(x^2)+1e-6) * w   (GGUF 里存的就是有效权重)
    a = GDN(h)  if il % 4 != 3  else  Attention(h) # §4 / §5
    x = x + a
    h = rmsnorm(x, post_attention_norm[il])
    f = ffn_down( silu(ffn_gate(h)) * ffn_up(h) )   # §6
    x = x + f
x = rmsnorm(x, output_norm); logits = lm_head(x)    # 只对需要的位置
```
每个三值矩阵乘 `W·v` 的输入 `v` 都要先做 Hadamard 变换(§3)。`ssm_alpha`/`ssm_beta`(bf16)**不旋转**,直接吃 RMSNorm 后的 `h`。

## 2. Embedding(`token_embd`,PTQ1_0,旋转基)

`z = dequant(row[token])`(5120 个值,6.2 个块里的 40 个块);`e = s5120 ⊙ (H z)`,即**先 Hadamard 再乘符号**(与线性层相反,Hadamard 是自逆的,`s²=1`)。`H` 对每个 1024 块独立做,5 个块。

## 3. Hadamard 变换(线性层输入侧)

对三值矩阵的输入向量 `v`(宽度 W ∈ {5120, 6144, 17408}):
```
v' = H_blockdiag( s_W ⊙ v ) ,   H[r][c] = (-1)^popcount(r & c) / sqrt(1024)     # 每个 1024 块独立
y  = Wternary · v'                                                              # W = scale_group * trits
```
- `s_W` 是 GGUF 里 `prism.hadamard.sign_values` 的三段(5120 / 6144 / 17408),**所有层共享**;顺序按 `sign_widths = [5120, 6144, 17408]` 拼接。
- RMSNorm 的 `γ` 可以和 `s_5120` 合并成一个向量 `γ⊙s`(离线预计算)。
- **`ssm_out` 的输入有一个头顺序置换**(`prism.hadamard.gdn_v_grouped = true`):GDN 输出是"tiled"V 头顺序 `[hd=128, nk=16, rep=3]`(V 头 j 用 K 头 `j % 16`),折叠权重却按"grouped"顺序 `[hd, rep, nk]` 计算,所以先置换再乘符号再旋转:
  `f_new = h + 128*(r + 3*k)` 处的值 = `x_old[h + 128*(k + 16*r)]`(`h∈[0,128)`, `r∈[0,3)`, `k∈[0,16)`)。

## 4. 门控 DeltaNet 层(GDN,48 层)

输入 `h`(RMSNorm 后,5120):
```
qkv = Wqkv · H(s⊙h)        # [10240] = q(16*128) | k(16*128) | v(48*128)
z   = Wgate · H(s⊙h)       # [6144]
beta_raw = ssm_beta · h    # [48], bf16 权重, 不旋转
alpha    = ssm_alpha · h   # [48]
conv:  y[t,c] = sum_{i=0..3} ssm_conv1d[i,c] * xin[t-3+i, c]   (xin = qkv, 因果, 状态保存最近 3 个输入 [3,10240]);   c = silu(y)
q = c[0:2048] (16头), k = c[2048:4096] (16头), v = c[4096:10240] (48头)
q, k: 每头 L2 归一化  x / max(||x||, 1e-6)
对每个 V 头 hv (0..47), 对应的 K 头 hk = hv % 16 (tiled):
    gate  g = ssm_a[hv] * softplus(alpha[hv] + ssm_dt.bias[hv])     # softplus(x) = x if x>20 else log(1+exp(x)); ssm_a 为负
    beta  = sigmoid(beta_raw[hv])
    S[hv] 是 128(key) x 128(value) 的 fp32 状态
    S *= exp(g)
    kS[v]  = sum_key S[key][v] * k[hk][key]
    d[v]   = (v_in[hv][v] - kS[v]) * beta
    S[key][v] += k[hk][key] * d[v]
    o[v]   = sum_key S[key][v] * q[hk][key]  * (1/sqrt(128))
gated norm (每个 V 头, 128 维):  n = rmsnorm(o, ssm_norm[128]);   out = silu(z[hv]) * n
out 共 6144 维(tiled 头顺序) --置换(§3)--> 符号 --> Hadamard --> ssm_out · → [5120]
```
状态布局(Prism 的 Metal 内核把状态转置存放,使 value 维连续);我们的实现自行决定,数学上等价。

## 5. 全注意力层(16 层,il%4==3)

```
Qfull = Wq · H(s⊙h)        # [12288]: 每个头 24 个 [query(256) | gate(256)] 交错
q = Qfull 每头前 256;  gate = 每头后 256
q = rmsnorm_per_head(q, q_norm[256]);   k = rmsnorm_per_head(Wk · .., k_norm[256]);   v = Wv · ..   # k,v: [4 heads, 256]
RoPE (IMROPE, 文本场景等价于 NEOX 式): 只旋转每头前 64 维, 配对 (i, i+32), i=0..31:
    theta_i = pos * 10000000^(-2*i/64);  (x_i, x_{i+32}) -> (x_i cos - x_{i+32} sin,  x_i sin + x_{i+32} cos)
注意力: scale = 1/sqrt(256), 因果, GQA (24 个 Q 头共享 4 个 KV 头, 每 6 个 Q 头一组)
out = attn * sigmoid(gate)             # [24*256 = 6144]   (HF config 里写 output_gate_type=swish, 但 HF 与 Prism 实现都是 sigmoid)
attn_out = Wo · H(s6144 ⊙ out)
```

## 6. FFN

`gate = Wgate · H(s5120⊙h)`, `up = Wup · H(s5120⊙h)`(两者共享同一次变换);`m = silu(gate) ⊙ up`;`ffn = Wdown · H(s17408 ⊙ m)`(17408 = 17 个 1024 块)。

## 7. PTQ1_0 权重(三值 + fp16 组 scale)

见 `docs/reference/prism_ptq1_0_quant_dequant.c` 和 CHECKPOINT §5.6。一行 K 个输入 = K/128 个 28 字节块;反量化值 `(trit-1) * d`。元素在块内的顺序:`qs[m] 的第 n 个 trit → 元素 n*16+m (m<16)`;`qs[16+m] → 80+n*8+m (m<8)`;`qh[h] 的第 n 个 trit → 120+n*2+h`。

## 8. MTP 层(blk.64,Q8_0)——已核实(M4)

MTP 头是**一个完整的全注意力块**,权重为 Q8_0(34 字节/块:fp16 scale + 32 个 int8),**没有 Hadamard 旋转**,有自己的 K/V 缓存(只有一层)。

第 `j` 行 MTP 计算"位置 `j` 的 token `x_j`"与"主干在位置 `j-1` 的隐状态 `h_{j-1}`"的配对,预测位置 `j+1` 的 token:

```
e   = embedding(x_j)                       # 与主干相同:s ⊙ (H z),旋转基的反变换
h   = output_norm(trunk residual at j-1)   # 主干最后一层之后、output_norm(含 γ)之后的"平坦"隐状态
cat = [ RMSNorm(e) * nextn.enorm | RMSNorm(h) * nextn.hnorm ]      # 10240
x   = eh_proj · cat                        # Q8_0,10240 -> 5120
a   = RMSNorm(x) * attn_norm               # 普通 RMSNorm,没有 sign/FWHT
q,k,v = Q8 matmuls of a                    # 与主干注意力同样的 [query|gate] 布局、逐头 q/k norm、partial RoPE(位置 j)、sigmoid 门
y   = x + attn_out(Q8, 6144 -> 5120)
f   = RMSNorm(y) * post_attention_norm
z   = y + ffn_down(silu(ffn_gate f) ⊙ ffn_up f)                    # Q8_0,5120 -> 17408 -> 5120
logits = lm_head( shared_head_norm(z) )    # 与主干共用 output.weight(旋转基),所以这里要做 sign + FWHT
```

要点:
- 位置 0 没有 MTP 行(没有 `h_{-1}`),所以 MTP 缓存的第一个有效位置是 1(`firstPos = 1`)。
- 预填充时 MTP 缓存只需要 K/V(不需要 FFN、不需要 logits),我们用 `eh_proj`、`wk`、`wv` 三个 GEMM 批量完成(`Engine::encMtpKv`)。
- 草稿的深度 1:用 `next`(已采样、尚未喂入)与 `h_{n-1}` 预测 `n+1` 位置的 token。
- 数值验证:在主模型自己的贪心续写上,MTP 的 top-1 命中率 62–65%(若公式有误只会是 ~5%)。

## 9. 验证状态

**M2 已用数值对照全部验证**(`build/kata --ref-dir ... --ref-logits ...` 对 Prism llama.cpp 的逐层导出):
- Hadamard 矩阵与符号向量的使用顺序;embedding 的反向变换 `s ⊙ (H z)`。
- GDN:causal conv 的抽头顺序(最老的输入对应 `w[0]`)、q/k 的 L2 归一化(`x / max(||x||, 1e-6)`)、门控与递推、gated RMSNorm、`ssm_out` 输入的 tiled→grouped 头置换。
- 全注意力:q/k 的逐头 RMSNorm、`[query|gate]` 交错布局、partial RoPE(前 64 维、配对 `(i, i+32)`、`theta_i = pos * 1e7^(-2i/64)`)、`1/sqrt(256)` 缩放、`sigmoid(gate)`(不是 swish)。
- FFN 与 lm_head。
- 5 个 token 的提示:64 层残差相对误差 1e-7~1e-4,top-1 一致 5/5,最大 logit 差 0.0005。
- 72 个 token 的提示:top-1 一致 72/72,最大 logit 差 0.0131,贪心续写与参考逐 token 一致。参考实现在长批次上走 fp16 的 mul_mm,我们全程 fp32,所以更精确的一方是我们。
