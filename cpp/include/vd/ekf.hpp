// Extended Kalman Filter for joint (v, mu) estimation.
//
// State: x = [v, mu]^T
// Process model (per-step):
//   v_{k+1}  = v_k + dt * ( -mu_k * g  -  (k_drag / m) * v_k^2 )
//   mu_{k+1} = mu_k
// Measurement: z = v + noise   ->   H = [1, 0]
//
// All matrices are 2x2 / 2x1 fixed-size; the entire filter step touches no
// heap and uses no exceptions. Designed to drop into a Cortex-M build target
// unchanged (Jetson today, MCU later).
#pragma once

#include <cstddef>
#include "matrix.hpp"

namespace vd {

struct EkfParams {
    double g       = 9.81;
    // The forward-model convention here matches src/physics/model.py exactly:
    //   dv/dt = -mu * g - k * v^2          (k has units of 1/m)
    // The Python EKF parameterises drag as rho_cd_a / (2m) instead; the
    // parity_check.py harness reconciles the two so this port is a true
    // numerical mirror, not a re-derivation.
    double k       = 0.02;
    double q_v     = 1e-1;  // process noise on v
    double q_mu    = 1e-2;  // process noise on mu (tune per scenario)
    double r       = 1.0;   // measurement noise on v
};

class Ekf {
public:
    Vec2 x{};
    Mat2 P{};
    EkfParams params{};

    Ekf() {
        x(0, 0) = 30.0;
        x(1, 0) = 0.5;
        P(0, 0) = 1.0;
        P(1, 1) = 1.0;
    }

    explicit Ekf(double v_init, double mu_init, EkfParams p = {}) : params(p) {
        x(0, 0) = v_init;
        x(1, 0) = mu_init;
        P(0, 0) = 1.0;
        P(1, 1) = 1.0;
    }

    void predict(double dt) noexcept {
        const double v  = x(0, 0);
        const double mu = x(1, 0);

        const double a = -mu * params.g - params.k * v * v;
        const double v_next = v + a * dt;

        x(0, 0) = v_next;
        x(1, 0) = mu;

        // Jacobian of f wrt (v, mu)
        const double df_dv  = 1.0 - dt * 2.0 * params.k * v;
        const double df_dmu = -params.g * dt;

        Mat2 F;
        F(0, 0) = df_dv;
        F(0, 1) = df_dmu;
        F(1, 0) = 0.0;
        F(1, 1) = 1.0;

        Mat2 Q;
        Q(0, 0) = params.q_v;
        Q(1, 1) = params.q_mu;

        // P = F P F^T + Q
        P = add(matmul(matmul(F, P), transpose(F)), Q);
    }

    void update(double z) noexcept {
        // H = [1, 0]; S = H P H^T + r = P(0,0) + r
        const double S = P(0, 0) + params.r;
        // K = P H^T / S = [P(0,0); P(1,0)] / S
        const double K0 = P(0, 0) / S;
        const double K1 = P(1, 0) / S;

        const double innov = z - x(0, 0);
        x(0, 0) += K0 * innov;
        x(1, 0) += K1 * innov;

        // P = (I - K H) P
        const double p00 = P(0, 0), p01 = P(0, 1), p10 = P(1, 0), p11 = P(1, 1);
        P(0, 0) = (1.0 - K0) * p00;
        P(0, 1) = (1.0 - K0) * p01;
        P(1, 0) = -K1 * p00 + p10;
        P(1, 1) = -K1 * p01 + p11;
    }

    double v()       const noexcept { return x(0, 0); }
    double mu()      const noexcept { return x(1, 0); }
    double sigma_v() const noexcept { return std::sqrt(P(0, 0)); }
    double sigma_mu() const noexcept { return std::sqrt(P(1, 1)); }
};

}  // namespace vd
