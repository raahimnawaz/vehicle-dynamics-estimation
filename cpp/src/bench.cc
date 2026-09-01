// Latency micro-benchmark for the EKF step and PINN forward pass.
//
// Reports: warm-up count, total iterations, median/p99 nanoseconds per op,
// throughput (ops/sec). High-resolution clock from <chrono>; no allocations
// in the hot loop.
//
// Usage:
//   bench [n_iters] [batch] [--json PATH]
//
// `batch` defaults to 200 and MUST stay >> 1: steady_clock's resolution on
// most hosts is coarser than a single 30-flop EKF step, so at batch=1 the
// timer -- not the filter -- is what gets measured. The symptom is a min of
// 0 ns and a median several times the batched figure. The Makefile's
// `run-bench` target passes the default explicitly for the same reason.
//
// `--json` writes the same numbers to a file in the schema `benchmarks/`
// uses, so the C++ column of the README has an archived artifact rather than
// a figure typed in by hand. See `make bench-json`.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <vector>

#include "vd/ekf.hpp"
#include "vd/pinn.hpp"

namespace {

using clk = std::chrono::steady_clock;

// Build identification, resolved at compile time so the JSON records the
// binary that actually produced the numbers.
#if defined(__clang__)
constexpr const char* kCompiler = "clang " __clang_version__;
#elif defined(__GNUC__)
constexpr const char* kCompiler = "gcc " __VERSION__;
#else
constexpr const char* kCompiler = "unknown";
#endif

#if defined(__aarch64__) || defined(_M_ARM64)
constexpr const char* kMachine = "arm64";
#elif defined(__x86_64__) || defined(_M_X64)
constexpr const char* kMachine = "x86_64";
#else
constexpr const char* kMachine = "unknown";
#endif

#if defined(__APPLE__)
constexpr const char* kOs = "Darwin";
#elif defined(_WIN32)
constexpr const char* kOs = "Windows";
#elif defined(__linux__)
constexpr const char* kOs = "Linux";
#else
constexpr const char* kOs = "unknown";
#endif

// Passed in by the Makefile so the recorded flags cannot drift from the ones
// the binary was built with.
#ifndef VD_BENCH_FLAGS
#define VD_BENCH_FLAGS "unknown"
#endif

struct Stats {
    long long median_ns;
    long long p99_ns;
    long long min_ns;
    long long max_ns;

    double mops() const { return median_ns > 0 ? 1e3 / static_cast<double>(median_ns) : 0.0; }
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

Stats bench_ekf(int n_iters, int batch) {
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
    (void)sink;
    return summarize(samples);
}

Stats bench_pinn(int n_iters, int batch) {
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
    (void)sink;
    return summarize(samples);
}

void print_stats(const char* label, const Stats& s) {
    std::printf("%s: median=%lld ns  p99=%lld ns  min=%lld ns  (%.2f Mops/s median)\n",
                label, s.median_ns, s.p99_ns, s.min_ns, s.mops());
}

void write_result(std::FILE* f, const char* key, const Stats& s, bool trailing_comma) {
    std::fprintf(f,
                 "    \"%s\": {\n"
                 "      \"median\": %lld,\n"
                 "      \"p99\": %lld,\n"
                 "      \"min\": %lld,\n"
                 "      \"max\": %lld,\n"
                 "      \"throughput_mops\": %.6f\n"
                 "    }%s\n",
                 key, s.median_ns, s.p99_ns, s.min_ns, s.max_ns, s.mops(),
                 trailing_comma ? "," : "");
}

// Mirrors the schema of benchmarks/*.json written by tools/bench_python.py,
// so the two sides of the speedup table are directly comparable. Hand-rolled
// rather than pulled from a JSON library to keep the port dependency-free.
bool write_json(const char* path, int n_iters, int batch,
                const Stats& ekf, const Stats& pinn) {
    std::FILE* f = std::fopen(path, "w");
    if (f == nullptr) return false;

    char today[32] = "unknown";
    const std::time_t now = std::time(nullptr);
    std::tm tm_buf{};
#if defined(_WIN32)
    if (localtime_s(&tm_buf, &now) == 0)
#else
    if (localtime_r(&now, &tm_buf) != nullptr)
#endif
        std::strftime(today, sizeof(today), "%Y-%m-%d", &tm_buf);

    std::fprintf(f, "{\n");
    std::fprintf(f,
                 "  \"host\": {\n"
                 "    \"os\": \"%s\",\n"
                 "    \"machine\": \"%s\"\n"
                 "  },\n",
                 kOs, kMachine);
    std::fprintf(f, "  \"compiler\": \"%s\",\n", kCompiler);
    std::fprintf(f, "  \"flags\": \"%s\",\n", VD_BENCH_FLAGS);
    std::fprintf(f, "  \"iterations\": %d,\n", n_iters);
    std::fprintf(f, "  \"batch\": %d,\n", batch);
    std::fprintf(f, "  \"results\": {\n");
    write_result(f, "ekf_step_ns", ekf, true);
    write_result(f, "pinn_forward_ns", pinn, false);
    std::fprintf(f, "  },\n");
    std::fprintf(f, "  \"date\": \"%s\",\n", today);
    std::fprintf(f,
                 "  \"notes\": \"Per-op timings; batch amortises steady_clock "
                 "resolution, which is coarser than one EKF step -- at batch=1 "
                 "the timer dominates and the median inflates several-fold. "
                 "Absolute figures are host- and thermal-state-dependent; the "
                 "speedup ratios against the matching *-python.json on the same "
                 "host are the portable quantity.\"\n");
    std::fprintf(f, "}\n");
    std::fclose(f);
    return true;
}

}  // namespace

int main(int argc, char** argv) {
    int n_iters = 100000;
    // Deliberately 200, not 1 -- see the header comment. A single EKF step is
    // shorter than the clock's tick.
    int batch   = 200;
    const char* json_path = nullptr;

    int positional = 0;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--json") == 0 && i + 1 < argc) {
            json_path = argv[++i];
        } else if (positional == 0) {
            n_iters = std::atoi(argv[i]);
            ++positional;
        } else if (positional == 1) {
            batch = std::atoi(argv[i]);
            ++positional;
        }
    }

    if (n_iters < 1) n_iters = 1;
    if (batch < 1) batch = 1;
    if (batch < 50) {
        std::printf("# WARNING: batch=%d is too small to amortise the clock; "
                    "the reported medians measure timer resolution, not the "
                    "code under test. Use batch >= 200.\n", batch);
    }

    std::printf("# bench: %d iters, batch %d (per-op timings)\n", n_iters, batch);
    std::printf("# build: %s | %s/%s\n", kCompiler, kOs, kMachine);

    const Stats ekf  = bench_ekf(n_iters, batch);
    const Stats pinn = bench_pinn(n_iters, batch);

    print_stats("EKF step    ", ekf);
    print_stats("PINN forward", pinn);

    if (json_path != nullptr) {
        if (!write_json(json_path, n_iters, batch, ekf, pinn)) {
            std::fprintf(stderr, "bench: could not write %s\n", json_path);
            return 1;
        }
        std::printf("wrote %s\n", json_path);
    }
    return 0;
}
