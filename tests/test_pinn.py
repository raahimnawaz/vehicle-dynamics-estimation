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
    """Free-form MuNet (concavity prior) must recover the Pacejka shape."""
    ds, meta = generate_dataset(n_runs=12, t_final=4.0, seed=0)
    net, _ = train_pinn(ds, epochs=3000, seed=0)
    s, mu_hat = evaluate_curve(net, n=200, s_max=0.3)
    mu_true = mu_pacejka(s, **PACEJKA_DRY)
    lo, hi = meta["s_range"]
    mask = (s >= lo + 1e-3) & (s <= hi - 1e-3)
    err = np.abs(mu_hat[mask] - mu_true[mask])
    assert err.mean() < 0.10, f"mean err {err.mean():.3f} too large"
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
