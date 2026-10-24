// Persistent-kernel feasibility probe on Apple M6 (Metal 4.1): grid barrier, co-residency, dispatch-chain overhead, streaming bandwidth with phases.
#define NS_PRIVATE_IMPLEMENTATION
#define MTL_PRIVATE_IMPLEMENTATION
#define CA_PRIVATE_IMPLEMENTATION
#include <Foundation/Foundation.hpp>
#include <Metal/Metal.hpp>
#include <cstdio>
#include <cstring>
#include <vector>
#include <string>
static const char* SRC = R"MSL(
#include <metal_stdlib>
using namespace metal;
#define SPIN_MAX 3000u
inline void gbar(device atomic_uint* ctr, device volatile uint* fail, uint target, uint tid) {
  threadgroup_barrier(mem_flags::mem_device);
  if (tid == 0) {
    uint arrived = atomic_fetch_add_explicit(ctr, 1u, memory_order_acq_rel, mem_flags::mem_device) + 1u;
    if (arrived < target) { uint spins = 0; while (atomic_load_explicit(ctr, memory_order_acquire, mem_flags::mem_device) < target) { if (++spins > SPIN_MAX) { fail[0] = 1; break; } } }
  }
  threadgroup_barrier(mem_flags::mem_device);
}
kernel void bar_test(device atomic_uint* ctr [[buffer(0)]], device volatile uint* fail [[buffer(1)]], constant uint& R [[buffer(2)]], constant uint& G [[buffer(3)]],
                     uint tid [[thread_index_in_threadgroup]]) {
  for (uint r = 0; r < R; r++) { gbar(ctr, fail, (r + 1) * G, tid); if (fail[0]) return; }
}
kernel void tiny(device uint* x [[buffer(0)]], uint i [[thread_position_in_grid]]) { if (i == 0) x[0] += 1; }

// streaming: W is uint4 array, P phases of PH uint4 each (PH = 1<<20 = 16 MiB)
constant uint PH = 1u << 20;
inline uint4 ld(device const uint4* W, uint idx) { return W[idx]; }
#define X4(v) ((v).x ^ (v).y ^ (v).z ^ (v).w)
// V0: single dispatch, all phases, no barriers
kernel void v0(device const uint4* W [[buffer(0)]], device uint* out [[buffer(1)]], constant uint& P [[buffer(2)]], constant uint& NT [[buffer(3)]], uint id [[thread_position_in_grid]]) {
  uint acc = 0; uint K = PH / NT;
  for (uint p = 0; p < P; p++) { uint base = p * PH + id; 
    for (uint k = 0; k < K; k += 8) { uint4 a0 = W[base + (k+0)*NT], a1 = W[base + (k+1)*NT], a2 = W[base + (k+2)*NT], a3 = W[base + (k+3)*NT], a4 = W[base + (k+4)*NT], a5 = W[base + (k+5)*NT], a6 = W[base + (k+6)*NT], a7 = W[base + (k+7)*NT];
      acc ^= X4(a0) ^ X4(a1) ^ X4(a2) ^ X4(a3) ^ X4(a4) ^ X4(a5) ^ X4(a6) ^ X4(a7); } }
  out[id] = acc;
}
// V1: one dispatch per phase (phase index passed in)
kernel void v1(device const uint4* W [[buffer(0)]], device uint* out [[buffer(1)]], constant uint& p [[buffer(2)]], constant uint& NT [[buffer(3)]], uint id [[thread_position_in_grid]]) {
  uint acc = 0; uint K = PH / NT; uint base = p * PH + id;
  for (uint k = 0; k < K; k += 8) { uint4 a0 = W[base + (k+0)*NT], a1 = W[base + (k+1)*NT], a2 = W[base + (k+2)*NT], a3 = W[base + (k+3)*NT], a4 = W[base + (k+4)*NT], a5 = W[base + (k+5)*NT], a6 = W[base + (k+6)*NT], a7 = W[base + (k+7)*NT];
    acc ^= X4(a0) ^ X4(a1) ^ X4(a2) ^ X4(a3) ^ X4(a4) ^ X4(a5) ^ X4(a6) ^ X4(a7); }
  out[id] ^= acc;
}
// V2: persistent, naive: barrier then stream the whole phase
kernel void v2(device const uint4* W [[buffer(0)]], device uint* out [[buffer(1)]], constant uint& P [[buffer(2)]], constant uint& NT [[buffer(3)]],
               device atomic_uint* ctr [[buffer(4)]], device volatile uint* fail [[buffer(5)]], uint id [[thread_position_in_grid]], uint tid [[thread_index_in_threadgroup]], uint G [[threadgroups_per_grid]]) {
  uint acc = 0; uint K = PH / NT;
  for (uint p = 0; p < P; p++) { uint base = p * PH + id;
    for (uint k = 0; k < K; k += 8) { uint4 a0 = W[base + (k+0)*NT], a1 = W[base + (k+1)*NT], a2 = W[base + (k+2)*NT], a3 = W[base + (k+3)*NT], a4 = W[base + (k+4)*NT], a5 = W[base + (k+5)*NT], a6 = W[base + (k+6)*NT], a7 = W[base + (k+7)*NT];
      acc ^= X4(a0) ^ X4(a1) ^ X4(a2) ^ X4(a3) ^ X4(a4) ^ X4(a5) ^ X4(a6) ^ X4(a7); }
    gbar(ctr, fail, (p + 1) * G, tid); if (fail[0]) break; }
  out[id] = acc;
}
// V3: persistent with prefetch: before the barrier of phase p, issue PF loads of phase p+1 into registers
#define PF 16
kernel void v3(device const uint4* W [[buffer(0)]], device uint* out [[buffer(1)]], constant uint& P [[buffer(2)]], constant uint& NT [[buffer(3)]],
               device atomic_uint* ctr [[buffer(4)]], device volatile uint* fail [[buffer(5)]], uint id [[thread_position_in_grid]], uint tid [[thread_index_in_threadgroup]], uint G [[threadgroups_per_grid]]) {
  uint acc = 0; uint K = PH / NT; uint4 pf[PF];
  { uint base = id; _Pragma("unroll") for (uint j = 0; j < PF; j++) pf[j] = W[base + j*NT]; }   // phase 0 prefetch
  for (uint p = 0; p < P; p++) { uint base = p * PH + id;
    _Pragma("unroll") for (uint j = 0; j < PF; j++) acc ^= X4(pf[j]);
    for (uint k = PF; k < K; k += 8) { uint4 a0 = W[base + (k+0)*NT], a1 = W[base + (k+1)*NT], a2 = W[base + (k+2)*NT], a3 = W[base + (k+3)*NT], a4 = W[base + (k+4)*NT], a5 = W[base + (k+5)*NT], a6 = W[base + (k+6)*NT], a7 = W[base + (k+7)*NT];
      acc ^= X4(a0) ^ X4(a1) ^ X4(a2) ^ X4(a3) ^ X4(a4) ^ X4(a5) ^ X4(a6) ^ X4(a7); }
    if (p + 1 < P) { uint nb = (p + 1) * PH + id; _Pragma("unroll") for (uint j = 0; j < PF; j++) pf[j] = W[nb + j*NT]; }   // issue next-phase loads BEFORE the barrier
    gbar(ctr, fail, (p + 1) * G, tid); if (fail[0]) break; }
  out[id] = acc;
}
)MSL";

static MTL::Device* dev; static MTL::CommandQueue* q; static MTL::Library* lib;
static MTL::ComputePipelineState* pso(const char* n) { NS::Error* e = nullptr; auto* f = lib->newFunction(NS::String::string(n, NS::UTF8StringEncoding)); auto* p = dev->newComputePipelineState(f, &e); if (!p) { printf("PSO %s fail: %s\n", n, e ? e->localizedDescription()->utf8String() : "?"); exit(1);} return p; }
static double gputime(MTL::CommandBuffer* cb) { return cb->GPUEndTime() - cb->GPUStartTime(); }

int main() {
  setvbuf(stdout, NULL, _IONBF, 0);
  NS::AutoreleasePool* pool = NS::AutoreleasePool::alloc()->init();
  dev = MTL::CreateSystemDefaultDevice(); q = dev->newCommandQueue();
  MTL::CompileOptions* o = MTL::CompileOptions::alloc()->init(); o->setLanguageVersion(MTL::LanguageVersion4_1); NS::Error* e = nullptr;
  lib = dev->newLibrary(NS::String::string(SRC, NS::UTF8StringEncoding), o, &e);
  if (!lib) { printf("compile fail: %s\n", e->localizedDescription()->utf8String()); return 1; }
  MTL::Buffer* ctr = dev->newBuffer(16, MTL::ResourceStorageModeShared); MTL::Buffer* fail = dev->newBuffer(16, MTL::ResourceStorageModeShared);

  // ---- Test A: barrier + residency
  printf("== A: global barrier (R=300 barriers). 'FAIL' = threadgroups not co-resident (bounded spin timed out)\n");
  auto* pA = pso("bar_test");
  for (int T : {256, 1024}) for (int G : {12, 24, 36, 48, 64, 96, 128, 144, 160, 176, 192, 256}) { static int failedT = -1; if (failedT == T) continue;
    uint32_t R = 300, GG = G; memset(ctr->contents(), 0, 16); memset(fail->contents(), 0, 16);
    auto* cb = q->commandBuffer(); auto* en = cb->computeCommandEncoder(); en->setComputePipelineState(pA);
    en->setBuffer(ctr, 0, 0); en->setBuffer(fail, 0, 1); en->setBytes(&R, 4, 2); en->setBytes(&GG, 4, 3);
    en->dispatchThreadgroups(MTL::Size(G, 1, 1), MTL::Size(T, 1, 1)); en->endEncoding(); cb->commit(); cb->waitUntilCompleted();
    bool bad = ((uint32_t*)fail->contents())[0] != 0 || cb->status() != MTL::CommandBufferStatusCompleted;
    printf("  T=%4d G=%3d : %s  %.1f us total, %.2f us / barrier\n", T, G, bad ? "FAIL" : "ok  ", gputime(cb) * 1e6, gputime(cb) * 1e6 / R); if (bad) failedT = T;
  }
  // ---- Test B: dependent dispatch chain
  printf("== B: chain of N dependent dispatches (tiny kernel, 48 TG x 256 thr)\n");
  auto* pB = pso("tiny"); MTL::Buffer* xb = dev->newBuffer(16, MTL::ResourceStorageModeShared);
  for (int N : {100, 400, 1600}) { double best = 1e9; for (int it = 0; it < 5; it++) { auto* cb = q->commandBuffer(); auto* en = cb->computeCommandEncoder(); en->setComputePipelineState(pB); en->setBuffer(xb, 0, 0);
      for (int i = 0; i < N; i++) en->dispatchThreadgroups(MTL::Size(48, 1, 1), MTL::Size(256, 1, 1)); en->endEncoding(); cb->commit(); cb->waitUntilCompleted(); double t = gputime(cb); if (t < best) best = t; }
    printf("  N=%4d : %.1f us total => %.2f us / dispatch\n", N, best * 1e6, best * 1e6 / N); }
  // ---- Test C: streaming
  const uint32_t P = 96; const size_t PHB = 16ull << 20; MTL::Buffer* W = dev->newBuffer(PHB * P, MTL::ResourceStorageModeShared);
  { uint64_t* w = (uint64_t*)W->contents(); size_t n = W->length() / 8; uint64_t s = 88172645463325252ull; for (size_t i = 0; i < n; i++) { s ^= s << 13; s ^= s >> 7; s ^= s << 17; w[i] = s; } }
  MTL::Buffer* out = dev->newBuffer(1 << 20, MTL::ResourceStorageModeShared);
  printf("== C: stream %u phases x 16 MiB = %.2f GB (random data), per phase = 16 MiB\n", P, (double)PHB * P / 1e9);
  struct Cfg { int G, T; }; for (Cfg c : std::vector<Cfg>{{16, 1024}, {32, 256}, {64, 256}, {128, 256}}) {
    uint32_t NT = c.G * c.T, PP = P; if ((1u << 20) % NT) continue;
    double bytes = (double)PHB * P; auto run = [&](const char* name, std::function<void(MTL::ComputeCommandEncoder*)> enc_fn, bool persistent) {
      double best = 1e9; bool bad = false;
      for (int it = 0; it < 6; it++) { memset(ctr->contents(), 0, 16); memset(fail->contents(), 0, 16);
        auto* cb = q->commandBuffer(); auto* en = cb->computeCommandEncoder(); enc_fn(en); en->endEncoding(); cb->commit(); cb->waitUntilCompleted();
        if (persistent && ((uint32_t*)fail->contents())[0]) bad = true; double t = gputime(cb); if (it >= 1 && t < best) best = t; }
      printf("    %-34s : %s %8.1f us  => %6.1f GB/s   (%.2f us / phase)\n", name, bad ? "FAIL" : "ok  ", best * 1e6, bytes / best / 1e9, best * 1e6 / P); };
    printf("  -- G=%d TG x T=%d thr (NT=%u, %u loads/thread/phase)\n", c.G, c.T, NT, (1u << 20) / NT);
    auto* p0 = pso("v0"); auto* p1 = pso("v1"); auto* p2 = pso("v2"); auto* p3 = pso("v3");
    run("V0 single dispatch, no barriers", [&](auto* en) { en->setComputePipelineState(p0); en->setBuffer(W,0,0); en->setBuffer(out,0,1); en->setBytes(&PP,4,2); en->setBytes(&NT,4,3); en->dispatchThreadgroups(MTL::Size(c.G,1,1), MTL::Size(c.T,1,1)); }, false);
    run("V1 one dispatch per phase", [&](auto* en) { en->setComputePipelineState(p1); en->setBuffer(W,0,0); en->setBuffer(out,0,1); en->setBytes(&NT,4,3);
        for (uint32_t p = 0; p < P; p++) { en->setBytes(&p,4,2); en->dispatchThreadgroups(MTL::Size(c.G,1,1), MTL::Size(c.T,1,1)); } }, false);
    run("V2 persistent + grid barrier (naive)", [&](auto* en) { en->setComputePipelineState(p2); en->setBuffer(W,0,0); en->setBuffer(out,0,1); en->setBytes(&PP,4,2); en->setBytes(&NT,4,3); en->setBuffer(ctr,0,4); en->setBuffer(fail,0,5); en->dispatchThreadgroups(MTL::Size(c.G,1,1), MTL::Size(c.T,1,1)); }, true);
    if ((1u << 20) / NT >= 16 + 8) run("V3 persistent + prefetch before barrier", [&](auto* en) { en->setComputePipelineState(p3); en->setBuffer(W,0,0); en->setBuffer(out,0,1); en->setBytes(&PP,4,2); en->setBytes(&NT,4,3); en->setBuffer(ctr,0,4); en->setBuffer(fail,0,5); en->dispatchThreadgroups(MTL::Size(c.G,1,1), MTL::Size(c.T,1,1)); }, true);
  }
  pool->release(); return 0; }
