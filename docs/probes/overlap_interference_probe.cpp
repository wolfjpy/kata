// Interference diagnosis: which kind of CPU work slows the GPU matmul, and by how much?
#define NS_PRIVATE_IMPLEMENTATION
#define MTL_PRIVATE_IMPLEMENTATION
#define CA_PRIVATE_IMPLEMENTATION
#include <Foundation/Foundation.hpp>
#include <Metal/Metal.hpp>
#include <arm_neon.h>
#include <cstdio>
#include <cstring>
#include <cstdint>
#include <vector>
#include <thread>
#include <atomic>
#include <chrono>
#include <functional>
#include <pthread.h>
#include <pthread/qos.h>
#include <unistd.h>
using Clock = std::chrono::steady_clock;
static double secs(Clock::time_point a, Clock::time_point b) { return std::chrono::duration<double>(b - a).count(); }
enum Mode { FP16 = 0, FP16_NT = 1, INT8 = 2, RESIDENT = 3, SPIN_ALU = 4 };
template <int MODE> static inline void expand_block(const uint8_t* b, void* outp) {
  uint16_t dh; memcpy(&dh, b + 26, 2); __fp16 d; memcpy(&d, &dh, 2); float16x8_t sc = vdupq_n_f16(d);
  uint8x16_t by = vld1q_u8(b); uint8x8_t by8 = vld1_u8(b + 16); int8x16_t one = vdupq_n_s8(1); uint8x16_t three = vdupq_n_u8(3);
  const uint8_t pw[5] = {1, 3, 9, 27, 81};
  __fp16* out = (__fp16*)outp; int8_t* o8 = (int8_t*)outp;
  for (int n = 0; n < 5; n++) {
    uint8x16_t q = vmulq_u8(by, vdupq_n_u8(pw[n]));
    uint8x16_t xi = vcombine_u8(vshrn_n_u16(vmull_u8(vget_low_u8(q), vget_low_u8(three)), 8), vshrn_n_u16(vmull_high_u8(q, three), 8));
    int8x16_t v = vsubq_s8(vreinterpretq_s8_u8(xi), one);
    uint8x8_t q8 = vmul_u8(by8, vdup_n_u8(pw[n]));
    int8x8_t v8 = vsub_s8(vreinterpret_s8_u8(vshrn_n_u16(vmull_u8(q8, vget_low_u8(three)), 8)), vget_low_s8(one));
    if (MODE == INT8) { vst1q_s8(o8 + n * 16, v); vst1_s8(o8 + 80 + n * 8, v8); }
    else {
      float16x8_t f0 = vmulq_f16(vcvtq_f16_s16(vmovl_s8(vget_low_s8(v))), sc), f1 = vmulq_f16(vcvtq_f16_s16(vmovl_high_s8(v)), sc), f2 = vmulq_f16(vcvtq_f16_s16(vmovl_s8(v8)), sc);
      if (MODE == FP16_NT) { __builtin_nontemporal_store(f0, (float16x8_t*)(out + n * 16)); __builtin_nontemporal_store(f1, (float16x8_t*)(out + n * 16 + 8)); __builtin_nontemporal_store(f2, (float16x8_t*)(out + 80 + n * 8)); }
      else { vst1q_f16(out + n * 16, f0); vst1q_f16(out + n * 16 + 8, f1); vst1q_f16(out + 80 + n * 8, f2); }
    }
  }
  for (int h = 0; h < 2; h++) for (int n = 0; n < 4; n++) { uint8_t q = (uint8_t)(b[24 + h] * (n == 0 ? 1 : n == 1 ? 3 : n == 2 ? 9 : 27)); int xi = ((uint16_t)q * 3) >> 8;
    if (MODE == INT8) o8[120 + n * 2 + h] = (int8_t)(xi - 1); else out[120 + n * 2 + h] = (__fp16)((xi - 1) * (float)d); }
}

static inline void expand_block_nt(const uint8_t* b, __fp16* out) {
  uint16_t dh; memcpy(&dh, b + 26, 2); __fp16 d; memcpy(&d, &dh, 2); float16x8_t sc = vdupq_n_f16(d);
  uint8x16_t by = vld1q_u8(b); uint8x8_t by8 = vld1_u8(b + 16); int8x16_t one = vdupq_n_s8(1); uint8x16_t three = vdupq_n_u8(3);
  const uint8_t pw[5] = {1, 3, 9, 27, 81}; float16x8_t f2v[5];
  for (int n = 0; n < 5; n++) {
    uint8x16_t q = vmulq_u8(by, vdupq_n_u8(pw[n]));
    uint8x16_t xi = vcombine_u8(vshrn_n_u16(vmull_u8(vget_low_u8(q), vget_low_u8(three)), 8), vshrn_n_u16(vmull_high_u8(q, three), 8));
    int8x16_t v = vsubq_s8(vreinterpretq_s8_u8(xi), one);
    uint8x8_t q8 = vmul_u8(by8, vdup_n_u8(pw[n]));
    int8x8_t v8 = vsub_s8(vreinterpret_s8_u8(vshrn_n_u16(vmull_u8(q8, vget_low_u8(three)), 8)), vget_low_s8(one));
    float16x8_t f0 = vmulq_f16(vcvtq_f16_s16(vmovl_s8(vget_low_s8(v))), sc), f1 = vmulq_f16(vcvtq_f16_s16(vmovl_high_s8(v)), sc);
    f2v[n] = vmulq_f16(vcvtq_f16_s16(vmovl_s8(v8)), sc);
    __fp16* p = out + n * 16; asm volatile("stnp %q0, %q1, [%2]" :: "w"(f0), "w"(f1), "r"(p) : "memory");
  }
  __fp16 qh[8]; for (int h = 0; h < 2; h++) for (int n = 0; n < 4; n++) { uint8_t q = (uint8_t)(b[24 + h] * (n == 0 ? 1 : n == 1 ? 3 : n == 2 ? 9 : 27)); int xi = ((uint16_t)q * 3) >> 8; qh[n * 2 + h] = (__fp16)((xi - 1) * (float)d); }
  float16x8_t qv = vld1q_f16(qh);
  __fp16* p = out + 80; asm volatile("stnp %q0, %q1, [%2]" :: "w"(f2v[0]), "w"(f2v[1]), "r"(p) : "memory");
  p = out + 96; asm volatile("stnp %q0, %q1, [%2]" :: "w"(f2v[2]), "w"(f2v[3]), "r"(p) : "memory");
  p = out + 112; asm volatile("stnp %q0, %q1, [%2]" :: "w"(f2v[4]), "w"(qv), "r"(p) : "memory");
}
struct Cpu { std::vector<std::thread> th; std::atomic<bool> stop{false}; std::vector<std::atomic<uint64_t>> blocks; Cpu(int n) : blocks(n) { for (auto& b : blocks) b = 0; } };
template <int MODE> static void worker(Cpu& c, int t, int nthreads, const uint8_t* src, uint8_t* dst, size_t nblocks, double target_wps, int qos) {
  pthread_set_qos_class_self_np(qos == 0 ? QOS_CLASS_USER_INTERACTIVE : (qos == 1 ? QOS_CLASS_UTILITY : QOS_CLASS_BACKGROUND), 0);
  const size_t BYTES = (MODE == INT8) ? 128 : 256; size_t per = nblocks / nthreads, b0 = per * t; std::vector<uint8_t> local; const uint8_t* s = src; uint8_t* d = dst;
  if (MODE == RESIDENT) { per = 1024; local.assign(per * BYTES, 0); b0 = 0; d = local.data(); }   // 28 KB src + 256 KB dst per thread: cache resident
  auto t0 = Clock::now(); uint64_t done = 0; const size_t SB = 2048; volatile float sink = 0;
  while (!c.stop) { for (size_t i = 0; i < per && !c.stop; i += SB) { size_t m = std::min(SB, per - i);
      if (MODE == SPIN_ALU) { float x = 1.0f; for (int k = 0; k < 400000; k++) x = x * 1.0000001f + 1e-9f; sink = x; done += 2048 * 8; /* nominal */ }
      else if (MODE == FP16_NT) { for (size_t j = 0; j < m; j++) expand_block_nt(s + (b0 + i + j) * 28, (__fp16*)(d + (b0 + i + j) * BYTES)); }
      else for (size_t j = 0; j < m; j++) expand_block<MODE>(s + (b0 + i + j) * 28, d + (MODE == RESIDENT ? 0 : 0) + (b0 + i + j) * BYTES - (MODE == RESIDENT ? b0 * BYTES : 0));
      if (MODE != SPIN_ALU) done += m; c.blocks[t] = done;
      if (target_wps > 0) { double want = (double)done * 128 / (target_wps / nthreads); double have = secs(t0, Clock::now()); if (want > have) usleep((useconds_t)((want - have) * 1e6)); } } }
}
static void cpu_start(Cpu& c, int mode, int nthreads, const uint8_t* src, uint8_t* dst, size_t nblocks, double target_wps, int qos) {
  c.stop = false;
  for (int t = 0; t < nthreads; t++) c.th.emplace_back([&c, mode, t, nthreads, src, dst, nblocks, target_wps, qos]() {
    switch (mode) { case FP16: worker<FP16>(c, t, nthreads, src, dst, nblocks, target_wps, qos); break; case FP16_NT: worker<FP16_NT>(c, t, nthreads, src, dst, nblocks, target_wps, qos); break;
      case INT8: worker<INT8>(c, t, nthreads, src, dst, nblocks, target_wps, qos); break; case RESIDENT: worker<RESIDENT>(c, t, nthreads, src, dst, nblocks, target_wps, qos); break; case SPIN_ALU: worker<SPIN_ALU>(c, t, nthreads, src, dst, nblocks, target_wps, qos); break; } });
}
static void cpu_stop(Cpu& c) { c.stop = true; for (auto& t : c.th) t.join(); c.th.clear(); }
static uint64_t total(Cpu& c) { uint64_t s = 0; for (auto& b : c.blocks) s += b.load(); return s; }
static const char* SRC = R"MSL(
#include <metal_stdlib>
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace metal; using namespace mpp::tensor_ops;
kernel void mm(device half* A [[buffer(0)]], device half* B [[buffer(1)]], device float* C [[buffer(2)]], constant uint3& dims [[buffer(3)]], uint2 tgid [[threadgroup_position_in_grid]]) {
  int M = dims.x, N = dims.y, K = dims.z; uint nt = tgid.x, mt = tgid.y;
  tensor<device half, dextents<int32_t,2>, tensor_inline> tA(A, dextents<int32_t,2>(K, M));
  tensor<device float, dextents<int32_t,2>, tensor_inline> tC(C, dextents<int32_t,2>(N, M));
  constexpr auto desc = matmul2d_descriptor(32, 64, static_cast<int>(dynamic_extent));
  matmul2d<desc, execution_simdgroups<1>> op;
  auto mA = tA.slice(0, mt*32); auto mC = tC.slice(nt*64, mt*32);
  tensor<device half, dextents<int32_t,2>, tensor_inline> tB(B + (ulong)nt*(ulong)K*64, dextents<int32_t,2>(64, K));
  op.run(mA, tB, mC);
}
)MSL";
int main() {
  setvbuf(stdout, NULL, _IONBF, 0);
  NS::AutoreleasePool* pool = NS::AutoreleasePool::alloc()->init();
  MTL::Device* dev = MTL::CreateSystemDefaultDevice(); MTL::CommandQueue* q = dev->newCommandQueue();
  MTL::CompileOptions* o = MTL::CompileOptions::alloc()->init(); o->setLanguageVersion(MTL::LanguageVersion4_1); NS::Error* e = nullptr;
  MTL::Library* lib = dev->newLibrary(NS::String::string(SRC, NS::UTF8StringEncoding), o, &e);
  auto* pso = dev->newComputePipelineState(lib->newFunction(NS::String::string("mm", NS::UTF8StringEncoding)), &e);
  const int M = 2048, N = 17408, K = 5120;
  MTL::Buffer* A = dev->newBuffer((size_t)M * K * 2, MTL::ResourceStorageModeShared); MTL::Buffer* B = dev->newBuffer((size_t)K * N * 2, MTL::ResourceStorageModeShared); MTL::Buffer* C = dev->newBuffer((size_t)M * N * 4, MTL::ResourceStorageModeShared);
  { uint32_t s = 1; auto rnd = [&]() { s ^= s << 13; s ^= s >> 17; s ^= s << 5; return s; }; _Float16* a = (_Float16*)A->contents(); for (size_t i = 0; i < (size_t)M * K; i++) a[i] = (_Float16)(((int)(rnd() & 0xffff) - 32768) / 65536.0f);
    _Float16* w = (_Float16*)B->contents(); for (size_t i = 0; i < (size_t)K * N; i++) w[i] = (_Float16)(((int)(rnd() % 3) - 1) * 0.01f); }
  const size_t LAYER_W = 382730240ull; const size_t nblocks = LAYER_W / 128;
  std::vector<uint8_t> src(nblocks * 28); { uint32_t s = 7; for (auto& b : src) { s ^= s << 13; s ^= s >> 17; s ^= s << 5; b = (uint8_t)(s % 243); } }
  MTL::Buffer* stage = dev->newBuffer(nblocks * 256, MTL::ResourceStorageModeShared); memset(stage->contents(), 0, stage->length());
  uint32_t dims[4] = {(uint32_t)M, (uint32_t)N, (uint32_t)K, 0};
  auto gpu_run = [&](int iters) { auto* cb = q->commandBuffer(); auto* en = cb->computeCommandEncoder(); en->setComputePipelineState(pso); en->setBuffer(A,0,0); en->setBuffer(B,0,1); en->setBuffer(C,0,2); en->setBytes(dims, sizeof dims, 3);
    for (int i = 0; i < iters; i++) en->dispatchThreadgroups(MTL::Size(N / 64, M / 32, 1), MTL::Size(32, 1, 1)); en->endEncoding(); cb->commit(); cb->waitUntilCompleted(); return cb->GPUEndTime() - cb->GPUStartTime(); };
  double fl1 = 2.0 * M * N * K; gpu_run(5);
  printf("== GPU alone: sustained vs burst (M=%d,N=%d,K=%d fp16)\n", M, N, K);
  for (int it : {30}) { double best = 1e9; for (int r = 0; r < 2; r++) { double t = gpu_run(it); if (t < best) best = t; } printf("  %3d back-to-back matmuls: %.2f ms each => %.2f TFLOPS\n", it, best / it * 1e3, fl1 * it / best / 1e12); }
  const int ITERS = 30; double tg = 1e9; for (int r = 0; r < 3; r++) tg = std::min(tg, gpu_run(ITERS)); double base = fl1 * ITERS / tg / 1e12;
  printf("== GPU baseline for comparisons: %.2f TFLOPS (%d iterations)\n", base, ITERS);
  struct R { int mode; int T; double wps; int qos; const char* label; };
  std::vector<R> rs = {
    {FP16, 1, 6.5e9, 0, "fp16 normal stores,  6.5 G w/s, 1 thr"},
    {FP16_NT, 1, 6.5e9, 0, "fp16 NON-TEMPORAL (stnp), 6.5 G w/s, 1 thr"},
    {FP16, 1, 13e9, 0, "fp16 normal stores, 13 G w/s, 1 thr"},
    {FP16_NT, 1, 13e9, 0, "fp16 NON-TEMPORAL (stnp), 13 G w/s, 1 thr"},
    {FP16_NT, 2, 26e9, 0, "fp16 NON-TEMPORAL (stnp), 26 G w/s, 2 thr"},
    {FP16_NT, 1, 0, 0, "fp16 NON-TEMPORAL (stnp), unthrottled, 1 thr"},
    {FP16, 1, 0, 0, "fp16 normal stores, unthrottled, 1 thr"},
    {INT8, 1, 13e9, 0, "int8 normal stores, 13 G w/s, 1 thr"},
    {FP16_NT, 1, 6.5e9, 0, "fp16 NON-TEMPORAL (stnp), 6.5 G w/s, 1 thr (repeat)"},
  };

  printf("== concurrent runs: GPU TFLOPS (%% of baseline) | CPU rate\n");
  for (auto& r : rs) { Cpu c(r.T); cpu_start(c, r.mode, r.T, src.data(), (uint8_t*)stage->contents(), nblocks, r.wps, r.qos); usleep(300000); uint64_t b0 = total(c); auto t0 = Clock::now();
    double t = gpu_run(ITERS); double dt = secs(t0, Clock::now()); uint64_t b1 = total(c); cpu_stop(c);
    double wps = (double)(b1 - b0) * 128 / dt; double bytes_per_w = r.mode == INT8 ? 1 : 2;
    if (r.mode == SPIN_ALU) printf("  %-58s: GPU %6.2f TFLOPS (%5.1f%%)\n", r.label, fl1 * ITERS / t / 1e12, 100 * (fl1 * ITERS / t / 1e12) / base);
    else printf("  %-58s: GPU %6.2f TFLOPS (%5.1f%%) | CPU %5.2f G w/s (%4.1f GB/s out)\n", r.label, fl1 * ITERS / t / 1e12, 100 * (fl1 * ITERS / t / 1e12) / base, wps / 1e9, wps * bytes_per_w / 1e9); }
  pool->release(); return 0; }
