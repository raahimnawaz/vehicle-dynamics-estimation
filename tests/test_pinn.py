"""Tests for the slip-aware physics module and the PINN recovery pipeline."""

import numpy as np
import pytest

from src.ml.pinn import evaluate_curve, generate_dataset, train_pinn
from src.physics.wheel import (
    PACEJKA_DRY,
    mu_exponential,
    mu_pacejka,
    pacejka_peak,
    ramp_slip,
    simulate,
)


def test_smooth_does_not_bias_the_endpoints():
    """Edge-padded smoothing must keep the endpoints on the signal (D-02).

    The old zero-padded `convolve(mode="same")` dragged v_s[0] to ~0.55*v0 and
    spiked its dv/dt to ~330 m/s^2; those samples were only kept out of training
    incidentally by the |dv/dt| < 12 mask.
    """
    from src.ml.pinn import _smooth

    assert np.allclose(_smooth(np.full(100, 30.0), 9), 30.0, atol=1e-9)
    ramp = np.linspace(30.0, 10.0, 100)
    rs = _smooth(ramp, 9)
    assert abs(rs[0] - 30.0) < 0.5 and abs(rs[-1] - 10.0) < 0.5


def test_mu_exponential_saturates():
    s = np.array([0.0, 0.05, 0.15, 0.3])
    mu = mu_exponential(s, mu_max=0.9, C=20.0)
    assert mu[0] == 0.0
    assert np.all(np.diff(mu) > 0)              # monotone
    assert 0.85 < mu[-1] < 0.9                  # saturates near mu_max


def test_mu_pacejka_has_peak_and_falloff():
    """Real Pacejka curves rise, peak, then fall -- a key qualitative property
    that distinguishes them from the exponential ablation."""
    s = np.linspace(0.0, 0.4, 4001)
    mu = mu_pacejka(s, **PACEJKA_DRY)
    assert mu[0] == 0.0
    s_peak, mu_peak = pacejka_peak(**PACEJKA_DRY)
    # Peak is in the physically reasonable region.
    assert 0.06 < s_peak < 0.20
    # The curve falls after the peak (not just saturates).
    assert mu[-1] < mu_peak - 0.02
    # Peak magnitude matches D (=mu_max) within a few percent.
    assert abs(mu_peak - PACEJKA_DRY["D"]) < 0.05


def test_simulate_decelerates_under_slip():
    t = np.arange(0, 3.0, 0.01)
    sched = ramp_slip(s_peak=0.15, ramp=0.3)
    v = simulate(30.0, t, sched, lambda s: mu_pacejka(s, **PACEJKA_DRY))
    assert v[-1] < v[0]
    assert np.all(np.diff(v) <= 1e-9)           # monotone non-increasing


@pytest.mark.slow
def test_munet_recovers_pacejka_shape():
    """Free-form MuNet (no shape prior) must recover the Pacejka shape."""
    ds, meta = generate_dataset(n_runs=12, t_final=4.0, seed=0)
    net, _ = train_pinn(ds, epochs=3000, seed=0)
    s, mu_hat = evaluate_curve(net, n=200, s_max=0.3)
    mu_true = mu_pacejka(s, **PACEJKA_DRY)
    lo, hi = meta["s_range"]
    mask = (s >= lo + 1e-3) & (s <= hi - 1e-3)
    err = np.abs(mu_hat[mask] - mu_true[mask])
    # Pinned near the README's headline 0.013 (measured 0.0126 at this config),
    # not the old 0.10 that let the claim silently degrade 7x (D-05).
    assert err.mean() < 0.03, f"mean err {err.mean():.3f} too large"
    # The recovered curve has a peak somewhere in the data range.
    i_peak = int(np.argmax(mu_hat))
    assert 0.05 < s[i_peak] < 0.3
    # Post-peak fall is present (not just saturating).
    assert mu_hat[-1] < mu_hat[i_peak] - 0.02


@pytest.mark.slow
def test_pacejkanet_recovers_parameters():
    """Grey-box parametric fit must recover (B, C, D, E) close to truth."""
    from src.ml.pinn import evaluate_pacejka_curve, train_pacejka
    ds, _ = generate_dataset(n_runs=12, t_final=4.0, seed=0)
    net, _ = train_pacejka(ds, epochs=2000, seed=0)
    s, mu_hat = evaluate_pacejka_curve(net, s_max=0.3)
    mu_true = mu_pacejka(s, **PACEJKA_DRY)
    err = float(np.mean(np.abs(mu_hat - mu_true)))
    assert err < 0.03, f"grey-box mean err {err:.3f} too large"
    p = net.params_dict()
    assert abs(p["D"] - PACEJKA_DRY["D"]) < 0.05


def test_friction_ellipse_endpoints_and_monotonicity():
    """The ellipse is exact physics at both ends and never rewards cornering."""
    from src.physics.wheel import friction_ellipse
    assert friction_ellipse(0.0) == pytest.approx(1.0), "full budget when straight"
    assert friction_ellipse(1.0) == pytest.approx(0.0), "no budget left at the limit"
    assert friction_ellipse(0.6) == pytest.approx(0.8), "sqrt(1 - 0.36)"
    n = np.linspace(0.0, 1.0, 200)
    assert np.all(np.diff(friction_ellipse(n)) <= 1e-12), "must be non-increasing"
    # Over-demanded corners clip rather than returning NaN.
    assert friction_ellipse(1.5) == pytest.approx(0.0)
    assert np.isfinite(friction_ellipse(np.array([2.0, -1.0]))).all()


def test_combined_dataset_meets_the_identifiability_conditions():
    """The two data conditions the mu(s) * ellipse(n) split actually needs.

    Both were established by ablation rather than assumed, and only these two
    matter -- the plane-coverage story that looks like the obvious answer is
    not one of them. Ramp-up-only data is 0.70-correlated and recovers the
    split fine (0.004); it is *exact* collinearity that destroys it (0.124).

      1. n must not be an exact function of s, or no (shape, scale) split is
         distinguishable from any other.
      2. The data must reach n ~ 0, or the ellipse(0) = 1 anchor is pinning a
         point the samples never visit: starving it degrades mean |d_mu| from
         0.007 to 0.208 while every loss term still looks healthy.

    A regression in either is silent -- the ODE residual fits either way -- so
    it is asserted on the dataset rather than left to the training run.
    """
    from src.ml.pinn import generate_dataset_combined
    ds, meta = generate_dataset_combined(n_runs=16, seed=0)
    assert abs(float(np.corrcoef(ds.s, ds.n)[0, 1])) < 0.90, (
        "s and n are near-collinear; the factorisation is not identifiable"
    )
    assert meta["n_range"][0] < 0.05, (
        f"data never gets closer to straight-line than n = {meta['n_range'][0]:.3f}, "
        "so the ellipse(0) = 1 anchor has no samples to bind against"
    )


def test_pacejka_defaults_match_the_ground_truth_set():
    """Calling the Pacejka helpers bare must give the curve the truth uses.

    These carried `E = 0.97` in their signatures while PACEJKA_DRY -- the set
    every dataset and every published figure is generated from -- specifies
    0.5. Nothing shipped was wrong, because every call site passes
    `**PACEJKA_DRY` explicitly, but a caller who omitted the keyword got a
    curve peaking at s = 0.180 against the true 0.127: a 42% error in the
    quantity an ABS controller exists to track, and up to 0.133 in mu, ten
    times the headline recovery error of 0.013.

    This is the same defect class as the 75x drag inconsistency (C1) -- one
    physical constant with two values in the codebase -- so it gets the same
    treatment: a single source, and a test that fails if a second copy appears.
    """
    from src.physics.wheel import PACEJKA_DRY, mu_combined, mu_pacejka, pacejka_peak

    s = np.linspace(0.0, 0.35, 701)
    assert np.array_equal(mu_pacejka(s), mu_pacejka(s, **PACEJKA_DRY)), (
        "mu_pacejka's keyword defaults have drifted from PACEJKA_DRY"
    )
    assert pacejka_peak() == pacejka_peak(**PACEJKA_DRY), (
        "pacejka_peak's keyword defaults have drifted from PACEJKA_DRY"
    )
    n = np.linspace(0.0, 0.8, 701)
    assert np.array_equal(mu_combined(s, n), mu_combined(s, n, **PACEJKA_DRY)), (
        "mu_combined's keyword defaults have drifted from PACEJKA_DRY"
    )

    # And the peak really is where the README says it is, from the bare call.
    s_peak, mu_peak = pacejka_peak()
    assert abs(s_peak - 0.127) < 1e-3, f"peak slip moved to {s_peak}"
    assert abs(mu_peak - 0.900) < 1e-3, f"peak mu moved to {mu_peak}"
