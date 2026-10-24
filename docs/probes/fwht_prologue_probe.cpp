// (1) cost of fusing RMSNorm + gamma*sign + FWHT-1024 x5 into a GEMV-like kernel's prologue; (2) sustained DRAM streaming bandwidth over ~10 s.
#define NS_PRIVATE_IMPLEMENTATION
#define MTL_PRIVATE_IMPLEMENTATION
#define CA_PRIVATE_IMPLEMENTATION
#include <Foundation/Foundation.hpp>
#include <Metal/Metal.hpp>
#include <cstdio>
#include <cstring>
#include <vector>
#include <cmath>
static const char* SRC = R"MSL(
#include <metal_stdlib>
using namespace metal;
constant uint KD = 5120;
#define X4(v) ((v).x ^ (v).y ^ (v).z ^ (v).w)
template <int MODE>
inline void body(device const uint4* W, uint base, uint NT, uint per, threadgroup half* xh, thread uint4* pf, thread float& acc, uint id) {
  uint k0 = 0;
  if (MODE == 2) { _Pragma("unroll") for (uint j = 0; j < 8; j++) acc += (float)(X4(pf[j]) & 0xffff) * (float)xh[(id + j * NT) % KD]; k0 = 8; }
  for (uint k = k0; k < per; k += 4) {
    uint4 a0 = W[base + (k+0)*NT], a1 = W[base + (k+1)*NT], a2 = W[base + (k+2)*NT], a3 = W[base + (k+3)*NT];
    acc += (float)(X4(a0) & 0xffff) * (float)xh[(id + (k+0)*NT) % KD] + (float)(X4(a1) & 0xffff) * (float)xh[(id + (k+1)*NT) % KD] + (float)(X4(a2) & 0xffff) * (float)xh[(id + (k+2)*NT) % KD] + (float)(X4(a3) & 0xffff) * (float)xh[(id + (k+3)*NT) % KD];
  }
}
// MODE 0: x comes precomputed (fp16 in device memory) -> copy to TG memory. MODE 1: full prologue (rms + gamma*sign + FWHT) then stream. MODE 2: issue first 8 weight loads before the prologue.
template <int MODE>
kernel void gemv(device const uint4* W [[buffer(0)]], device const float* xres [[buffer(1)]], device const float* gv [[buffer(2)]], device const half* xpre [[buffer(3)]], device float* out [[buffer(4)]],
                 constant uint& TOTAL4 [[buffer(5)]], uint tgid [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]], uint G [[threadgroups_per_grid]], uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
  threadgroup float xs[5120]; threadgroup half xh[5120]; threadgroup float part[8];
  const uint T = 256; const uint NT = G * T; const uint id = tgid * T + tid; const uint per = TOTAL4 / NT; const uint base = id;
  uint4 pf[8]; float acc = 0;
  if (MODE == 2) { _Pragma("unroll") for (uint j = 0; j < 8; j++) pf[j] = W[base + j * NT]; }
  if (MODE == 0) { for (uint i = tid; i < KD; i += T) xh[i] = xpre[i]; threadgroup_barrier(mem_flags::mem_threadgroup); }
  else {
    float ss = 0; for (uint i = tid; i < KD; i += T) { float v = xres[i]; xs[i] = v; ss += v * v; }
    ss = simd_sum(ss); if (lane == 0) part[sg] = ss; threadgroup_barrier(mem_flags::mem_threadgroup);
    float tot = 0; for (uint s = 0; s < 8; s++) tot += part[s]; float inv = rsqrt(tot * (1.0f / 5120.0f) + 1e-6f);
    for (uint i = tid; i < KD; i += T) xs[i] = xs[i] * inv * gv[i];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = 0; s < 10; s++) { uint h = 1u << s;
      for (uint j = tid; j < 2560; j += T) { uint blk = j >> 9; uint r = j & 511; uint lo = ((r >> s) << (s + 1)) | (r & (h - 1)); uint i0 = blk * 1024 + lo; uint i1 = i0 + h; float a = xs[i0], b = xs[i1]; xs[i0] = a + b; xs[i1] = a - b; }
      threadgroup_barrier(mem_flags::mem_threadgroup); }
    for (uint i = tid; i < KD; i += T) xh[i] = half(xs[i] * 0.03125f);
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  body<MODE>(W, base, NT, per, xh, pf, acc, id);
  out[id] = acc;
}
template [[host_name("g0")]] kernel void gemv<0>(device const uint4*, device const float*, device const float*, device const half*, device float*, constant uint&, uint, uint, uint, uint, uint);
template [[host_name("g1")]] kernel void gemv<1>(device const uint4*, device const float*, device const float*, device const half*, device float*, constant uint&, uint, uint, uint, uint, uint);
template [[host_name("g2")]] kernel void gemv<2>(device const uint4*, device const float*, device const float*, device const half*, device float*, constant uint&, uint, uint, uint, uint, uint);

// Optimized standalone: 5 TGs x 256 thr. Each TG: sumsq over all 5120 (for rms), gamma*sign on its 1024 block, FWHT-1024 = 2 in-register stages + 5 simd_shuffle_xor stages + TG transpose + 3 in-register stages.
kernel void fwht5(device const float* xres [[buffer(0)]], device const float* gv [[buffer(1)]], device half* xout [[buffer(2)]],
                  uint blk [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]], uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
  threadgroup float part[8]; threadgroup float xs[1024];
  float ss = 0; for (uint i = tid; i < 5120; i += 256) { float v = xres[i]; ss += v * v; }
  ss = simd_sum(ss); if (lane == 0) part[sg] = ss; threadgroup_barrier(mem_flags::mem_threadgroup);
  float tot = 0; for (uint s = 0; s < 8; s++) tot += part[s]; float inv = rsqrt(tot * (1.0f / 5120.0f) + 1e-6f);
  uint i0 = blk * 1024 + tid * 4; float4 v = float4(xres[i0], xres[i0+1], xres[i0+2], xres[i0+3]) * float4(gv[i0], gv[i0+1], gv[i0+2], gv[i0+3]) * inv;
  v = float4(v.x + v.y, v.x - v.y, v.z + v.w, v.z - v.w); v = float4(v.x + v.z, v.y + v.w, v.x - v.z, v.y - v.w);      // bits 0,1 in-register
  for (uint k = 0; k < 5; k++) { uint m = 1u << k; float4 p = float4(simd_shuffle_xor(v.x, m), simd_shuffle_xor(v.y, m), simd_shuffle_xor(v.z, m), simd_shuffle_xor(v.w, m)); v = (lane & m) ? (p - v) : (v + p); } // bits 2..6
  xs[tid * 4 + 0] = v.x; xs[tid * 4 + 1] = v.y; xs[tid * 4 + 2] = v.z; xs[tid * 4 + 3] = v.w; threadgroup_barrier(mem_flags::mem_threadgroup);
  if (tid < 128) { float e[8]; for (uint j = 0; j < 8; j++) e[j] = xs[tid + 128 * j];                                                                              // bits 7..9
    for (uint s = 0; s < 3; s++) { uint h = 1u << s; for (uint j = 0; j < 8; j++) if (!(j & h)) { float a = e[j], b = e[j | h]; e[j] = a + b; e[j | h] = a - b; } }
    for (uint j = 0; j < 8; j++) xout[blk * 1024 + tid + 128 * j] = half(e[j] * 0.03125f); }
}
// sustained streaming
constant uint PH = 1u << 20;
kernel void sv0(device const uint4* W [[buffer(0)]], device uint* out [[buffer(1)]], constant uint& P [[buffer(2)]], constant uint& NT [[buffer(3)]], uint id [[thread_position_in_grid]]) {
  uint acc = 0; uint K = PH / NT;
  for (uint p = 0; p < P; p++) { uint base = p * PH + id;
    for (uint k = 0; k < K; k += 8) { uint4 a0 = W[base + (k+0)*NT], a1 = W[base + (k+1)*NT], a2 = W[base + (k+2)*NT], a3 = W[base + (k+3)*NT], a4 = W[base + (k+4)*NT], a5 = W[base + (k+5)*NT], a6 = W[base + (k+6)*NT], a7 = W[base + (k+7)*NT];
      acc ^= X4(a0) ^ X4(a1) ^ X4(a2) ^ X4(a3) ^ X4(a4) ^ X4(a5) ^ X4(a6) ^ X4(a7); } }
  out[id] ^= acc;
}
)MSL";
int main() {
  setvbuf(stdout, NULL, _IONBF, 0);
  NS::AutoreleasePool* pool = NS::AutoreleasePool::alloc()->init();
  MTL::Device* dev = MTL::CreateSystemDefaultDevice(); MTL::CommandQueue* q = dev->newCommandQueue();
  MTL::CompileOptions* o = MTL::CompileOptions::alloc()->init(); o->setLanguageVersion(MTL::LanguageVersion4_1); NS::Error* e = nullptr;
  MTL::Library* lib = dev->newLibrary(NS::String::string(SRC, NS::UTF8StringEncoding), o, &e); if (!lib) { printf("compile fail: %s\n", e->localizedDescription()->utf8String()); return 1; }
  auto P = [&](const char* n) { auto* p = dev->newComputePipelineState(lib->newFunction(NS::String::string(n, NS::UTF8StringEncoding)), &e); if (!p) { printf("pso %s fail\n", n); exit(1);} return p; };
  const uint32_t TOTAL4 = 2359296;  // 36 MiB of weights (gate+up sized)
  MTL::Buffer* W = dev->newBuffer((size_t)TOTAL4 * 16, MTL::ResourceStorageModeShared); { uint64_t* w = (uint64_t*)W->contents(); uint64_t s = 88172645463325252ull; for (size_t i = 0; i < W->length() / 8; i++) { s ^= s << 13; s ^= s >> 7; s ^= s << 17; w[i] = s; } }
  MTL::Buffer* xres = dev->newBuffer(5120 * 4, MTL::ResourceStorageModeShared); MTL::Buffer* gv = dev->newBuffer(5120 * 4, MTL::ResourceStorageModeShared); MTL::Buffer* xpre = dev->newBuffer(5120 * 2, MTL::ResourceStorageModeShared); MTL::Buffer* out = dev->newBuffer(1 << 20, MTL::ResourceStorageModeShared);
  { float* x = (float*)xres->contents(); float* g = (float*)gv->contents(); _Float16* xp = (_Float16*)xpre->contents(); for (int i = 0; i < 5120; i++) { x[i] = sinf(i * 0.37f) * 3.0f; g[i] = ((i * 7919) & 1 ? 1.f : -1.f) * (0.5f + (i % 13) * 0.05f); xp[i] = (_Float16)(x[i] * 0.1f); } }
  printf("== (1) prologue cost: stream %.1f MiB of weights (gate+up size), K=5120, 256 thr/TG\n", TOTAL4 * 16 / 1048576.0);
  MTL::ComputePipelineState* ps[3] = {P("g0"), P("g1"), P("g2")}; const char* nm[3] = {"M0 no prologue (x precomputed fp16)", "M1 prologue: rms+gamma*sign+FWHT, then stream", "M2 prologue overlapped with first 8 weight loads"};
  for (uint32_t G : {64u, 96u, 128u, 192u, 256u, 384u}) { if (TOTAL4 % (G * 256) || (TOTAL4 / (G * 256)) % 4) continue; double t[3];
    for (int m = 0; m < 3; m++) { double best = 1e9; for (int it = 0; it < 12; it++) { auto* cb = q->commandBuffer(); auto* en = cb->computeCommandEncoder(); en->setComputePipelineState(ps[m]); en->setBuffer(W,0,0); en->setBuffer(xres,0,1); en->setBuffer(gv,0,2); en->setBuffer(xpre,0,3); en->setBuffer(out,0,4); en->setBytes(&TOTAL4,4,5);
          en->dispatchThreadgroups(MTL::Size(G,1,1), MTL::Size(256,1,1)); en->endEncoding(); cb->commit(); cb->waitUntilCompleted(); double d = cb->GPUEndTime() - cb->GPUStartTime(); if (it >= 2 && d < best) best = d; } t[m] = best; }
    printf("  G=%3u TGs:  M0 %6.1f us (%5.1f GB/s) | M1 %6.1f us (+%5.1f us, +%4.1f%%) | M2 %6.1f us (+%5.1f us, +%4.1f%%)\n", G, t[0]*1e6, TOTAL4*16/t[0]/1e9, t[1]*1e6, (t[1]-t[0])*1e6, 100*(t[1]-t[0])/t[0], t[2]*1e6, (t[2]-t[0])*1e6, 100*(t[2]-t[0])/t[0]); (void)nm; }

  printf("== (1b) separate-kernel alternative: [fwht5 (5 TGs) -> gemv M0] as two dependent dispatches vs fused prologue\n");
  auto* pf5 = P("fwht5"); MTL::Buffer* xo = dev->newBuffer(5120 * 2, MTL::ResourceStorageModeShared);
  { double best = 1e9; for (int it = 0; it < 30; it++) { auto* cb = q->commandBuffer(); auto* en = cb->computeCommandEncoder(); en->setComputePipelineState(pf5); en->setBuffer(xres,0,0); en->setBuffer(gv,0,1); en->setBuffer(xo,0,2);
        for (int r = 0; r < 20; r++) en->dispatchThreadgroups(MTL::Size(5,1,1), MTL::Size(256,1,1)); en->endEncoding(); cb->commit(); cb->waitUntilCompleted(); double d = (cb->GPUEndTime() - cb->GPUStartTime()) / 20; if (it >= 3 && d < best) best = d; }
    printf("  fwht5 alone (20 back-to-back, per kernel incl. 0.6us gap): %.2f us\n", best * 1e6); }
  // correctness check of fwht5 vs CPU reference
  { auto* cb = q->commandBuffer(); auto* en = cb->computeCommandEncoder(); en->setComputePipelineState(pf5); en->setBuffer(xres,0,0); en->setBuffer(gv,0,1); en->setBuffer(xo,0,2); en->dispatchThreadgroups(MTL::Size(5,1,1), MTL::Size(256,1,1)); en->endEncoding(); cb->commit(); cb->waitUntilCompleted();
    float* x = (float*)xres->contents(); float* g = (float*)gv->contents(); double ss = 0; for (int i = 0; i < 5120; i++) ss += (double)x[i] * x[i]; float inv = 1.0f / sqrtf((float)(ss / 5120.0) + 1e-6f); std::vector<float> ref(5120); for (int i = 0; i < 5120; i++) ref[i] = x[i] * inv * g[i];
    for (int b = 0; b < 5; b++) for (int h = 1; h < 1024; h <<= 1) for (int i = 0; i < 1024; i++) if (!(i & h)) { float a = ref[b*1024+i], c = ref[b*1024+i+h]; ref[b*1024+i] = a + c; ref[b*1024+i+h] = a - c; }
    double maxerr = 0, maxabs = 0; _Float16* o = (_Float16*)xo->contents(); for (int i = 0; i < 5120; i++) { double r = ref[i] * 0.03125; maxabs = fmax(maxabs, fabs(r)); maxerr = fmax(maxerr, fabs((double)o[i] - r)); }
    printf("  fwht5 correctness vs CPU reference: max abs err %.5f (max |value| %.3f)\n", maxerr, maxabs); }
  for (uint32_t G : {96u, 128u, 192u, 256u}) { if (TOTAL4 % (G * 256) || (TOTAL4 / (G * 256)) % 4) continue; double best = 1e9, bestM0 = 1e9, bestM1 = 1e9;
    for (int it = 0; it < 14; it++) { auto* cb = q->commandBuffer(); auto* en = cb->computeCommandEncoder();
        en->setComputePipelineState(pf5); en->setBuffer(xres,0,0); en->setBuffer(gv,0,1); en->setBuffer(xo,0,2); en->dispatchThreadgroups(MTL::Size(5,1,1), MTL::Size(256,1,1));
        en->setComputePipelineState(ps[0]); en->setBuffer(W,0,0); en->setBuffer(xres,0,1); en->setBuffer(gv,0,2); en->setBuffer(xo,0,3); en->setBuffer(out,0,4); en->setBytes(&TOTAL4,4,5); en->dispatchThreadgroups(MTL::Size(G,1,1), MTL::Size(256,1,1)); en->endEncoding(); cb->commit(); cb->waitUntilCompleted(); double d = cb->GPUEndTime() - cb->GPUStartTime(); if (it >= 2 && d < best) best = d; }
    for (int m = 0; m < 2; m++) { double b2 = 1e9; for (int it = 0; it < 12; it++) { auto* cb = q->commandBuffer(); auto* en = cb->computeCommandEncoder(); en->setComputePipelineState(ps[m]); en->setBuffer(W,0,0); en->setBuffer(xres,0,1); en->setBuffer(gv,0,2); en->setBuffer(xpre,0,3); en->setBuffer(out,0,4); en->setBytes(&TOTAL4,4,5); en->dispatchThreadgroups(MTL::Size(G,1,1), MTL::Size(256,1,1)); en->endEncoding(); cb->commit(); cb->waitUntilCompleted(); double d = cb->GPUEndTime() - cb->GPUStartTime(); if (it >= 2 && d < b2) b2 = d; } (m == 0 ? bestM0 : bestM1) = b2; }
    printf("  G=%3u:  M0 (no FWHT) %6.1f us | fused prologue M1 %6.1f us (+%5.1f) | separate [fwht5 -> M0] %6.1f us (+%5.1f us, +%4.1f%%)\n", G, bestM0*1e6, bestM1*1e6, (bestM1-bestM0)*1e6, best*1e6, (best-bestM0)*1e6, 100*(best-bestM0)/bestM0); }
  return 0;
  printf("== (2) sustained streaming bandwidth (read-only, 1.61 GB buffer re-read), per ~1.1 s window, ~13 s total\n");
  const uint32_t Pn = 96; MTL::Buffer* S = dev->newBuffer((size_t)Pn * (16u << 20), MTL::ResourceStorageModeShared); { uint64_t* w = (uint64_t*)S->contents(); uint64_t s = 99991; for (size_t i = 0; i < S->length() / 8; i++) { s ^= s << 13; s ^= s >> 7; s ^= s << 17; w[i] = s; } }
  auto* sv = P("sv0"); uint32_t NT = 32 * 256; MTL::Buffer* o2 = dev->newBuffer(1 << 20, MTL::ResourceStorageModeShared);
  for (int w = 0; w < 12; w++) { auto* cb = q->commandBuffer(); auto* en = cb->computeCommandEncoder(); en->setComputePipelineState(sv); en->setBuffer(S,0,0); en->setBuffer(o2,0,1); uint32_t PP = Pn; en->setBytes(&PP,4,2); en->setBytes(&NT,4,3);
    for (int i = 0; i < 100; i++) en->dispatchThreadgroups(MTL::Size(32,1,1), MTL::Size(256,1,1)); en->endEncoding(); cb->commit(); cb->waitUntilCompleted(); double d = cb->GPUEndTime() - cb->GPUStartTime();
    printf("  window %2d: %.2f s  => %.1f GB/s\n", w, d, 100.0 * Pn * (16u << 20) / d / 1e9); }
  pool->release(); return 0; }
