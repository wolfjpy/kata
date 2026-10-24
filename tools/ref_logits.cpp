// Reference tool: runs the Prism llama.cpp build on given token ids (or text) and dumps
//   - logits for every position (float32 [n_tokens][n_vocab])   (--out)
//   - residual stream after every layer, "l_out-<il>" (float32 [n_tokens][n_embd]) (--dump-dir)
//   - greedy continuation (--gen N)
// Build against the reference checkout, see Makefile target `ref_logits`.
#include "llama.h"
#include "ggml-backend.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <map>
#include <sys/stat.h>

struct Dump { std::map<std::string, std::vector<float>> layers; bool enabled = false; };
static bool eval_cb(struct ggml_tensor * t, bool ask, void * ud) {
    Dump * d = (Dump *) ud;
    if (!d->enabled) return false;
    const char * n = ggml_get_name(t);
    bool want = strncmp(n, "l_out-", 6) == 0 || strncmp(n, "attn_norm-", 10) == 0 || strncmp(n, "linear_attn_out-", 16) == 0 || strncmp(n, "attn_output-", 12) == 0 || strcmp(n, "result_norm") == 0 || strncmp(n, "ffn_out-", 8) == 0;
    if (ask) return want;
    if (!want) return true;
    size_t nb = ggml_nbytes(t); std::vector<uint8_t> tmp(nb); ggml_backend_tensor_get(t, tmp.data(), 0, nb);
    int64_t ne0 = t->ne[0], ne1 = t->ne[1]; std::vector<float> f((size_t)(ne0 * ne1));
    if (t->type == GGML_TYPE_F32) { memcpy(f.data(), tmp.data(), f.size() * 4); } else return true;
    auto & v = d->layers[n]; v.insert(v.end(), f.begin(), f.end());
    return true;
}
int main(int argc, char ** argv) {
    std::string model, text, out, dumpdir, idsarg; int gen = 0; int nctx = 4096;
    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        if (a == "--model") model = argv[++i]; else if (a == "--text") text = argv[++i]; else if (a == "--ids") idsarg = argv[++i];
        else if (a == "--out") out = argv[++i]; else if (a == "--dump-dir") dumpdir = argv[++i]; else if (a == "--gen") gen = atoi(argv[++i]); else if (a == "--ctx") nctx = atoi(argv[++i]);
    }
    if (model.empty()) { fprintf(stderr, "usage: ref_logits --model M (--text T | --ids 1,2,3) [--out logits.bin] [--dump-dir D] [--gen N]\n"); return 1; }
    llama_backend_init();
    llama_model_params mp = llama_model_default_params(); mp.n_gpu_layers = 99;
    llama_model * m = llama_model_load_from_file(model.c_str(), mp); if (!m) { fprintf(stderr, "load failed\n"); return 1; }
    const llama_vocab * vocab = llama_model_get_vocab(m); const int nv = llama_vocab_n_tokens(vocab);
    std::vector<llama_token> toks;
    if (!idsarg.empty()) { size_t p = 0; while (p < idsarg.size()) { size_t q = idsarg.find(',', p); if (q == std::string::npos) q = idsarg.size(); toks.push_back(atoi(idsarg.substr(p, q - p).c_str())); p = q + 1; } }
    else { toks.resize(text.size() + 16); int n = llama_tokenize(vocab, text.c_str(), (int)text.size(), toks.data(), (int)toks.size(), false, true); if (n < 0) { toks.resize(-n); n = llama_tokenize(vocab, text.c_str(), (int)text.size(), toks.data(), (int)toks.size(), false, true); } toks.resize(n); }
    printf("n_tokens=%zu n_vocab=%d ids:", toks.size(), nv); for (auto t : toks) printf(" %d", t); printf("\n");
    Dump dump; dump.enabled = !dumpdir.empty();
    llama_context_params cp = llama_context_default_params(); cp.n_ctx = nctx; cp.n_batch = 2048; cp.n_ubatch = 512; cp.cb_eval = eval_cb; cp.cb_eval_user_data = &dump;
    llama_context * ctx = llama_init_from_model(m, cp); if (!ctx) { fprintf(stderr, "ctx failed\n"); return 1; }
    std::vector<float> all((size_t) toks.size() * nv);
    const int CH = 512;
    for (size_t s = 0; s < toks.size(); s += CH) {
        int n = (int) std::min<size_t>(CH, toks.size() - s);
        llama_batch b = llama_batch_init(n, 0, 1);
        for (int i = 0; i < n; i++) { b.token[i] = toks[s + i]; b.pos[i] = (llama_pos)(s + i); b.n_seq_id[i] = 1; b.seq_id[i][0] = 0; b.logits[i] = 1; }
        b.n_tokens = n;
        if (llama_decode(ctx, b) != 0) { fprintf(stderr, "decode failed\n"); return 1; }
        for (int i = 0; i < n; i++) memcpy(all.data() + (s + i) * nv, llama_get_logits_ith(ctx, i), nv * sizeof(float));
        llama_batch_free(b);
    }
    if (!out.empty()) { FILE * f = fopen(out.c_str(), "wb"); fwrite(all.data(), 4, all.size(), f); fclose(f); printf("wrote %s\n", out.c_str()); }
    if (dump.enabled) { mkdir(dumpdir.c_str(), 0755); for (auto & kv : dump.layers) { std::string p = dumpdir + "/" + kv.first + ".f32"; FILE * f = fopen(p.c_str(), "wb"); fwrite(kv.second.data(), 4, kv.second.size(), f); fclose(f); } printf("dumped %zu tensors to %s\n", dump.layers.size(), dumpdir.c_str()); }
    const float * last = all.data() + (toks.size() - 1) * nv;
    std::vector<int> idx(nv); for (int i = 0; i < nv; i++) idx[i] = i; std::partial_sort(idx.begin(), idx.begin() + 5, idx.end(), [&](int a, int b2) { return last[a] > last[b2]; });
    printf("last position top5:"); for (int i = 0; i < 5; i++) printf("  %d(%.4f)", idx[i], last[idx[i]]); printf("\n");
    if (gen > 0) {
        dump.enabled = false; llama_token cur = idx[0]; std::string outs; int pos = (int) toks.size();
        for (int g = 0; g < gen; g++) {
            char buf[256]; int n = llama_token_to_piece(vocab, cur, buf, sizeof buf, 0, true); if (n > 0) outs.append(buf, n); printf("gen[%d]=%d\n", g, cur);
            llama_batch b = llama_batch_init(1, 0, 1); b.token[0] = cur; b.pos[0] = pos++; b.n_seq_id[0] = 1; b.seq_id[0][0] = 0; b.logits[0] = 1; b.n_tokens = 1;
            if (llama_decode(ctx, b) != 0) break; const float * l = llama_get_logits_ith(ctx, 0); int best = 0; for (int i = 1; i < nv; i++) if (l[i] > l[best]) best = i; cur = best; llama_batch_free(b);
        }
        printf("generated text: %s\n", outs.c_str());
    }
    llama_free(ctx); llama_model_free(m); return 0;
}
