// PINN forward pass: s -> mu.
//
// Architecture mirrors src/ml/pinn.py::MuNet exactly:
//   Linear(1, H) -> tanh -> Linear(H, H) -> tanh -> Linear(H, 1) -> 1.2 * sigmoid
//
// Weights are baked into the binary by tools/export_weights.py — there is no
// run-time file I/O, malloc, or external dependency. The forward pass uses a
// fixed-size stack scratchpad so it fits comfortably on a microcontroller
// (Cortex-M4 with the FPU enabled would hold a single inference's working set
// in <300 bytes of stack).
#pragma once

#include <cmath>
#include "pinn_weights.h"

namespace vd {

class Pinn {
public:
    // Returns mu_theta(s).
    static double forward(double s) noexcept {
        constexpr int H = vd::pinn_weights::kHidden;
        double h1[H];
        double h2[H];

        // Layer 1: h1 = tanh(W1 * s + b1)
        for (int i = 0; i < H; ++i) {
            h1[i] = std::tanh(vd::pinn_weights::W1[i] * s + vd::pinn_weights::b1[i]);
        }

        // Layer 2: h2 = tanh(W2 * h1 + b2)
        for (int i = 0; i < H; ++i) {
            double acc = vd::pinn_weights::b2[i];
            const double* row = &vd::pinn_weights::W2[i * H];
            for (int j = 0; j < H; ++j) acc += row[j] * h1[j];
            h2[i] = std::tanh(acc);
        }

        // Layer 3: out = W3 * h2 + b3
        double out = vd::pinn_weights::b3[0];
        for (int j = 0; j < H; ++j) out += vd::pinn_weights::W3[j] * h2[j];

        // 1.2 * sigmoid(out)
        const double sigmoid = 1.0 / (1.0 + std::exp(-out));
        return 1.2 * sigmoid;
    }
};

}  // namespace vd
