// Latency micro-benchmark for the EKF step and PINN forward pass.
//
// Reports: warm-up count, total iterations, median/p99 nanoseconds per op,
// throughput (ops/sec). High-resolution clock from <chrono>; no allocations
// in the hot loop.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "vd/ekf.hpp"
#include "vd/pinn.hpp"

namespace {

using clk = std::chrono::steady_clock;

struct Stats {
    long long median_ns;
    long long p99_ns;
    long long min_ns;
    long long max_ns;
};

Stats summarize(std::vector<long long>& samples) {
    std::sort(samples.begin(), samples.end());
    const auto n = samples.size();
    Stats s{};
    s.median_ns = samples[n / 2];
    s.p99_ns    = samples[std::min<size_t>(n - 1, static_cast<size_t>(0.99 * n))];
    s.min_ns    = samples.front();
    s.max_ns    = samples.back();
    return s;
}

void bench_ekf(int n_iters, int batch) {
    vd::Ekf ekf(30.0, 0.5);
    std::vector<long long> samples;
    samples.reserve(n_iters);
    volatile double sink = 0.0;

    // warm up
    for (int i = 0; i < 1000; ++i) {
        ekf.predict(0.01);
        ekf.update(29.5);
        sink += ekf.mu();
    }

    for (int i = 0; i < n_iters; ++i) {
        const auto t0 = clk::now();
        for (int b = 0; b < batch; ++b) {
            ekf.predict(0.01);
            ekf.update(29.5 + (b % 7) * 0.01);
            sink += ekf.mu();
        }
        const auto t1 = clk::now();
        samples.push_back(std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count() / batch);
    }
    auto s = summarize(samples);
    double mops = (s.median_ns > 0) ? (1e3 / s.median_ns) : 0.0;
    std::printf("EKF step    : median=%lld ns  p99=%lld ns  min=%lld ns  (%.2f Mops/s median)\n",
                s.median_ns, s.p99_ns, s.min_ns, mops);
    (void)sink;
}

void bench_pinn(int n_iters, int batch) {
    std::vector<long long> samples;
    samples.reserve(n_iters);

    volatile double sink = 0.0;
    for (int i = 0; i < 1000; ++i) sink += vd::Pinn::forward(0.1);

    for (int i = 0; i < n_iters; ++i) {
        const auto t0 = clk::now();
        for (int b = 0; b < batch; ++b) {
            const double s = 0.001 + (b % 300) / 1000.0;
            sink += vd::Pinn::forward(s);
        }
        const auto t1 = clk::now();
        samples.push_back(std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count() / batch);
    }
    auto s = summarize(samples);
    std::printf("PINN forward: median=%lld ns  p99=%lld ns  min=%lld ns  (%.2f Mops/s median)\n",
                s.median_ns, s.p99_ns, s.min_ns, 1e3 / s.median_ns);
    (void)sink;
}

}  // namespace

int main(int argc, char** argv) {
    int n_iters = 100000;
    int batch   = 1;
    if (argc >= 2) n_iters = std::atoi(argv[1]);
    if (argc >= 3) batch   = std::atoi(argv[2]);
    std::printf("# bench: %d iters, batch %d (per-op timings)\n", n_iters, batch);
    bench_ekf(n_iters, batch);
    bench_pinn(n_iters, batch);
    return 0;
}
