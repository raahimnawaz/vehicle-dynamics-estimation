// Minimal translation unit that exercises the shipped headers and nothing else.
//
// This exists to make the footprint claim in the README measurable against the
// thing it is actually about. `bench` and `parity` both link machinery that
// never ships -- std::vector, <chrono>, <fstream>, a JSON writer -- so their
// binary sizes have always overstated what an embedded target would flash, and
// they drift whenever the harness gains a feature. (They did: adding JSON
// output to the benchmark grew it from 35 KB to 52 KB without a single byte
// changing in ekf.hpp or pinn.hpp.)
//
// What this links is the whole deployable surface: the EKF step, the PINN
// forward pass, and the baked-in weights. Build and measure with
// `make -C cpp footprint`.
//
// The result is printed so the optimiser cannot discard the computation.
#include <cstdio>

#include "vd/ekf.hpp"
#include "vd/pinn.hpp"

int main() {
    vd::Ekf ekf(30.0, 0.5);
    for (int i = 0; i < 100; ++i) {
        ekf.predict(0.01);
        ekf.update(29.5);
    }
    const double mu_net = vd::Pinn::forward(0.12);
    std::printf("%.9f %.9f\n", ekf.mu(), mu_net);
    return 0;
}
