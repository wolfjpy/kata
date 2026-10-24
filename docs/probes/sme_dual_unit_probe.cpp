#include <cstdio>
#include <cstdint>
#include <thread>
#include <atomic>
#include <vector>
#include <mach/mach_time.h>
#include <pthread.h>
#include <pthread/qos.h>
static double now() { static mach_timebase_info_data_t tb; if (!tb.denom) mach_timebase_info(&tb); return (double)mach_absolute_time() * tb.numer / tb.denom * 1e-9; }
#define CLOB "memory","cc","x9","v0","v1","v2","v3","v4","v5","v6","v7","v8","v9","v10","v11","v12","v13","v14","v15","v16","v17","v18","v19","v20","v21","v22","v23","v24","v25","v26","v27","v28","v29","v30","v31","p0","p1","p2","p3","p4","p5","p6","p7","p8","p9","p10","p11","p12","p13","p14","p15"
static void k_s8(uint64_t n) { asm volatile("smstart\n ptrue p0.b\n mov z0.b, #1\n mov z1.b, #1\n"
  "1:\n smopa za0.s, p0/m, p0/m, z0.b, z1.b\n smopa za1.s, p0/m, p0/m, z0.b, z1.b\n smopa za2.s, p0/m, p0/m, z0.b, z1.b\n smopa za3.s, p0/m, p0/m, z0.b, z1.b\n subs %0, %0, #1\n b.ne 1b\n smstop\n" : "+r"(n) :: CLOB); }
int main() { setvbuf(stdout, NULL, _IONBF, 0);
  const uint64_t iters = 30000000ull; const double opsPer = 2.0 * 1024 * 4 * iters;
  auto trial = [&](const char* label, std::vector<qos_class_t> qs) {
    std::vector<std::thread> th; std::atomic<int> ready{0}; std::atomic<bool> go{false}; std::vector<double> dt(qs.size());
    for (size_t t = 0; t < qs.size(); t++) th.emplace_back([&, t]() { pthread_set_qos_class_self_np(qs[t], 0); k_s8(1000); ready++; while (!go) {} double t0 = now(); k_s8(iters); dt[t] = now() - t0; });
    while (ready < (int)qs.size()) {} go = true; for (auto& x : th) x.join();
    double agg = 0; printf("%-34s", label); for (size_t t = 0; t < qs.size(); t++) { double r = opsPer / dt[t] / 1e12; agg += r; printf("  [%s %.2f]", qs[t] == QOS_CLASS_USER_INTERACTIVE ? "UI" : "BG", r); } printf("  => aggregate %.2f TOPS\n", agg); };
  trial("1 UI thread", {QOS_CLASS_USER_INTERACTIVE});
  trial("1 BG thread", {QOS_CLASS_BACKGROUND});
  trial("1 UI + 1 BG concurrently", {QOS_CLASS_USER_INTERACTIVE, QOS_CLASS_BACKGROUND});
  trial("2 UI + 2 BG concurrently", {QOS_CLASS_USER_INTERACTIVE, QOS_CLASS_USER_INTERACTIVE, QOS_CLASS_BACKGROUND, QOS_CLASS_BACKGROUND});
  trial("2 UI (Super-sized)", {QOS_CLASS_USER_INTERACTIVE, QOS_CLASS_USER_INTERACTIVE});
  return 0; }
