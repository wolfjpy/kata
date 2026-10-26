// M28b step: opens the DFlash2 draft GGUF, binds every tensor the draft needs and reports the geometry it found.
#include "draft.h"
#include "engine.h"
#include <cstdio>
int main(int argc, char** argv) {
    if (argc < 3) { fprintf(stderr, "usage: draft_probe model.gguf draft.gguf\n"); return 1; }
    Engine eng; if (!eng.load(argv[1], "build/bonsai.metallib", 4096, 0)) return 1;
    Draft d; if (!d.load(eng, argv[2])) return 1;
    printf("draft ok: %u layers, %u heads / %u kv, head %u, window %u, block %u\n", Draft::kLayers, Draft::kHeads, Draft::kKvHeads, Draft::kHeadDim, Draft::kWindow, Draft::kBlock);
    printf("target taps:");
    for (int i = 0; i < 5; i++) printf(" %d", d.targetLayers[i]);
    printf("\nfc=%llu encNorm=%llu outNorm=%llu | layer0 q=%llu gate=%llu attnCp=%llu ffnCp=%llu\n",
           (unsigned long long)d.fc, (unsigned long long)d.encNorm, (unsigned long long)d.outNorm,
           (unsigned long long)d.L[0].q, (unsigned long long)d.L[0].gate, (unsigned long long)d.L[0].attnCp, (unsigned long long)d.L[0].ffnCp);
    return 0;
}
