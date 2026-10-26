// Dumps the draft layer-0 attention inputs/outputs so an independent implementation can be compared (diagnostic tool).
#include "draft.h"
#include "engine.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
static void dump(const char* path, MTL::Buffer* b, size_t nbytes) { FILE* f = fopen(path, "wb"); fwrite(b->contents(), 1, nbytes, f); fclose(f); }
int main(int argc, char** argv) {
    Engine eng; if (!eng.load(argv[1], "build/kata.metallib", 4096, 0)) return 1;
    Draft d; if (!d.load(eng, argv[2])) return 1; eng.mtpTrack = false;
    MTL::Buffer* taps = eng.gpu.alloc(5 * 8 * 5120 * 2);
    // build a 16-token context through the wide path so the draft has something to attend to
    std::vector<int> X = {32, 417, 351, 357, 417, 351, 357, 417, 351, 357, 417, 351, 357, 417, 351, 357};
    int am[8];
    for (uint32_t p = 0; p < X.size(); p += 8) { eng.verifyWide(X.data() + p, 8, p, am, taps, d.targetLayers, 5); eng.commitWide(8); d.materialise(eng, taps->contents(), 8); }
    int anchor = am[7];
    MTL::Buffer* logits = eng.bufLogitsAllPublic();
    d.dumpLayer0 = true;
    d.forwardBlock(eng, anchor, (uint32_t)X.size(), logits);
    dump("/tmp/d_embed.bin", d.xA, 8 * 5120 * 4);
    dump("/tmp/d_q.bin", d.qOut0, 8 * 32 * 128 * 4);
    dump("/tmp/d_ring_k.bin", d.kbuf, 4096 * 1024 * 2);
    dump("/tmp/d_ring_v.bin", d.vbuf, 4096 * 1024 * 2);
    dump("/tmp/d_attnout.bin", d.attnOut0, 8 * 32 * 128 * 4);
    dump("/tmp/d_convin.bin", d.convIn0, 8 * 5120 * 4);
    dump("/tmp/d_convdyn.bin", d.convDyn0, 8 * 1280 * 4);
    dump("/tmp/d_convout.bin", d.convOut0, 8 * 5120 * 4);
    printf("anchor %d pos %zu dumped\n", anchor, X.size());
    return 0;
}
