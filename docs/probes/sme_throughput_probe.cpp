#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <vector>
#include <thread>
#include <atomic>
#include <mach/mach_time.h>
#include <pthread.h>
#include <pthread/qos.h>
static double now() { static mach_timebase_info_data_t tb; if (!tb.denom) mach_timebase_info(&tb); return (double)mach_absolute_time() * tb.numer / tb.denom * 1e-9; }
#define CLOB "memory","cc","x9", \
 "v0","v1","v2","v3","v4","v5","v6","v7","v8","v9","v10","v11","v12","v13","v14","v15","v16","v17","v18","v19","v20","v21","v22","v23","v24","v25","v26","v27","v28","v29","v30","v31", \
 "p0","p1","p2","p3","p4","p5","p6","p7","p8","p9","p10","p11","p12","p13","p14","p15"
// Each loop iteration issues 4 independent outer products (za0..za3). smstart/smstop zero Z/P regs; we init after smstart.
static void k_probe() { asm volatile("smstart\n smstop\n" ::: CLOB); }
static void k_f32(uint64_t n) { asm volatile("smstart\n ptrue p0.s\n fmov z0.s, #1.0\n fmov z1.s, #1.0\n"
  "1:\n fmopa za0.s, p0/m, p0/m, z0.s, z1.s\n fmopa za1.s, p0/m, p0/m, z0.s, z1.s\n fmopa za2.s, p0/m, p0/m, z0.s, z1.s\n fmopa za3.s, p0/m, p0/m, z0.s, z1.s\n subs %0, %0, #1\n b.ne 1b\n smstop\n" : "+r"(n) :: CLOB); }
static void k_f16(uint64_t n) { asm volatile("smstart\n ptrue p0.h\n fmov z0.h, #1.0\n fmov z1.h, #1.0\n"
  "1:\n fmopa za0.s, p0/m, p0/m, z0.h, z1.h\n fmopa za1.s, p0/m, p0/m, z0.h, z1.h\n fmopa za2.s, p0/m, p0/m, z0.h, z1.h\n fmopa za3.s, p0/m, p0/m, z0.h, z1.h\n subs %0, %0, #1\n b.ne 1b\n smstop\n" : "+r"(n) :: CLOB); }
static void k_bf16(uint64_t n) { asm volatile("smstart\n ptrue p0.h\n mov w9, #0x3f80\n dup z0.h, w9\n dup z1.h, w9\n"
  "1:\n bfmopa za0.s, p0/m, p0/m, z0.h, z1.h\n bfmopa za1.s, p0/m, p0/m, z0.h, z1.h\n bfmopa za2.s, p0/m, p0/m, z0.h, z1.h\n bfmopa za3.s, p0/m, p0/m, z0.h, z1.h\n subs %0, %0, #1\n b.ne 1b\n smstop\n" : "+r"(n) :: CLOB); }
static void k_s8(uint64_t n) { asm volatile("smstart\n ptrue p0.b\n mov z0.b, #1\n mov z1.b, #1\n"
  "1:\n smopa za0.s, p0/m, p0/m, z0.b, z1.b\n smopa za1.s, p0/m, p0/m, z0.b, z1.b\n smopa za2.s, p0/m, p0/m, z0.b, z1.b\n smopa za3.s, p0/m, p0/m, z0.b, z1.b\n subs %0, %0, #1\n b.ne 1b\n smstop\n" : "+r"(n) :: CLOB); }
struct K { const char* name; void (*fn)(uint64_t); double macs; };   // MACs per MOPA at SVL=512: f32 256; f16/bf16 512; s8 1024
static void run(const K& k, int nt, uint64_t iters, int qos) {
  std::vector<std::thread> th; std::atomic<int> ready{0}; std::atomic<bool> go{false}; std::vector<double> dt(nt);
  for (int t = 0; t < nt; t++) th.emplace_back([&, t]() {
    pthread_set_qos_class_self_np(qos == 0 ? QOS_CLASS_USER_INTERACTIVE : QOS_CLASS_BACKGROUND, 0);
    k.fn(1000); ready++; while (!go) {} double t0 = now(); k.fn(iters); dt[t] = now() - t0; });
  while (ready < nt) {} go = true; for (auto& x : th) x.join();
  double worst = 0, best = 1e9; for (double d : dt) { worst = d > worst ? d : worst; best = d < best ? d : best; }
  double ops = 2.0 * k.macs * 4.0 * (double)iters;
  printf("  %-5s %-10s threads=%2d : per-thread %6.2f (best %6.2f) T(FL)OPS | aggregate %6.2f\n", k.name, qos == 0 ? "UI-QoS" : "BG-QoS(E?)", nt, ops / worst / 1e12, ops / best / 1e12, nt * ops / worst / 1e12);
}
int main(int argc, char** argv) {
  setvbuf(stdout, NULL, _IONBF, 0);
  printf("probe smstart/smstop...\n"); k_probe(); printf("smstart OK\n");
  K ks[] = { {"f32", k_f32, 256}, {"f16", k_f16, 512}, {"bf16", k_bf16, 512}, {"s8", k_s8, 1024} };
  for (auto& k : ks) { for (int n : {1, 2, 3, 4, 6}) run(k, n, 30000000ull, 0); }
  K& f16 = ks[1]; K& s8 = ks[3];
  printf("-- efficiency cores (background QoS)\n"); run(f16, 1, 30000000ull, 1); run(s8, 1, 30000000ull, 1); run(s8, 6, 30000000ull, 1);
  return 0;
}
