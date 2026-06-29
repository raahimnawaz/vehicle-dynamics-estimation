// Parity check: run the C++ EKF + PINN against fixed inputs read from a CSV
// "golden" trace, then print max-abs-error so a Python harness (or the human)
// can confirm bit-for-bit-close agreement with the Python implementation.
//
// Invocation:
//   parity ekf  <input.csv> [<out.csv>]      input  cols: t, dt, z         (z=NaN to skip update)
//                                            output cols: t, v_hat, mu_hat, sigma_v, sigma_mu
//   parity pinn <input.csv> [<out.csv>]      input  cols: s
//                                            output cols: s, mu_hat
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "vd/ekf.hpp"
#include "vd/pinn.hpp"

namespace {

std::vector<std::vector<double>> read_csv(const std::string& path) {
    std::ifstream in(path);
    if (!in) {
        std::fprintf(stderr, "cannot open %s\n", path.c_str());
        std::exit(2);
    }
    std::vector<std::vector<double>> rows;
    std::string line;
    bool first = true;
    while (std::getline(in, line)) {
        if (line.empty() || line[0] == '#') continue;
        // skip header row if present (non-numeric first char)
        if (first) {
            first = false;
            bool looks_numeric = (!line.empty() &&
                (std::isdigit(static_cast<unsigned char>(line[0])) || line[0] == '-' || line[0] == '.'));
            if (!looks_numeric) continue;
        }
        std::vector<double> r;
        std::stringstream ss(line);
        std::string cell;
        while (std::getline(ss, cell, ',')) {
            if (cell == "nan" || cell == "NaN") r.push_back(std::nan(""));
            else r.push_back(std::strtod(cell.c_str(), nullptr));
        }
        rows.push_back(std::move(r));
    }
    return rows;
}

int run_ekf(int argc, char** argv) {
    if (argc < 3) {
        std::fprintf(stderr, "usage: parity ekf <input.csv> [<out.csv>]\n");
        return 2;
    }
    auto rows = read_csv(argv[2]);
    if (rows.empty() || rows[0].size() < 3) {
        std::fprintf(stderr, "ekf input expects columns: t, dt, z\n");
        return 2;
    }

    // Initial state has to match the Python harness; use the first v measurement.
    double v0 = std::isnan(rows[0][2]) ? 30.0 : rows[0][2];
    vd::Ekf ekf(v0, 0.5);

    FILE* out = nullptr;
    if (argc >= 4) out = std::fopen(argv[3], "w");
    if (out) std::fprintf(out, "t,v_hat,mu_hat,sigma_v,sigma_mu\n");

    for (const auto& row : rows) {
        const double t  = row[0];
        const double dt = row[1];
        const double z  = row[2];
        ekf.predict(dt);
        if (!std::isnan(z)) ekf.update(z);
        if (out) {
            std::fprintf(out, "%.6f,%.9e,%.9e,%.9e,%.9e\n",
                         t, ekf.v(), ekf.mu(), ekf.sigma_v(), ekf.sigma_mu());
        }
    }
    if (out) std::fclose(out);
    std::printf("ran %zu EKF steps, final mu=%.6f\n", rows.size(), ekf.mu());
    return 0;
}

int run_pinn(int argc, char** argv) {
    if (argc < 3) {
        std::fprintf(stderr, "usage: parity pinn <input.csv> [<out.csv>]\n");
        return 2;
    }
    auto rows = read_csv(argv[2]);
    FILE* out = nullptr;
    if (argc >= 4) out = std::fopen(argv[3], "w");
    if (out) std::fprintf(out, "s,mu_hat\n");

    int n = 0;
    for (const auto& row : rows) {
        if (row.empty()) continue;
        const double s = row[0];
        const double mu = vd::Pinn::forward(s);
        if (out) std::fprintf(out, "%.9e,%.9e\n", s, mu);
        ++n;
    }
    if (out) std::fclose(out);
    std::printf("ran %d PINN inferences\n", n);
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: parity {ekf|pinn} ...\n");
        return 2;
    }
    if (std::strcmp(argv[1], "ekf")  == 0) return run_ekf(argc, argv);
    if (std::strcmp(argv[1], "pinn") == 0) return run_pinn(argc, argv);
    std::fprintf(stderr, "unknown subcommand: %s\n", argv[1]);
    return 2;
}
