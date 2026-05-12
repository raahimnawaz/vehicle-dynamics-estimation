"""Tests for the slip-aware physics module and the PINN recovery pipeline."""

import numpy as np
import pytest

from src.ml.pinn import evaluate_curve, generate_dataset, train_pinn
from src.physics.wheel import mu_pacejka, ramp_slip, simulate


def test_mu_pacejka_shape():
    s = np.array([0.0, 0.05, 0.15, 0.3])
    mu = mu_pacejka(s, mu_max=0.9, C=20.0)
    assert mu[0] == 0.0
    assert np.all(np.diff(mu) > 0)              # monotonic
    assert mu[-1] < 0.9 and mu[-1] > 0.85       # saturates near mu_max


def test_simulate_decelerates_under_slip():
    t = np.arange(0, 3.0, 0.01)
    sched = ramp_slip(s_peak=0.15, ramp=0.3)
    v = simulate(30.0, t, sched, lambda s: mu_pacejka(s))
    assert v[-1] < v[0]
    assert np.all(np.diff(v) <= 1e-9)           # monotone non-increasing


@pytest.mark.slow
def test_pinn_recovers_saturating_curve():
    ds, meta = generate_dataset(n_runs=8, t_final=4.0, seed=0)
    net, _ = train_pinn(ds, epochs=2000, seed=0)
    s, mu_hat = evaluate_curve(net, n=100, s_max=0.3)
    mu_true = mu_pacejka(s, meta["mu_max"], meta["C"])
    # only score in the slip range actually covered by the data
    lo, hi = meta["s_range"]
    mask = (s >= lo + 1e-3) & (s <= hi - 1e-3)
    err = np.abs(mu_hat[mask] - mu_true[mask])
    assert err.max() < 0.15, f"max err {err.max():.3f} too large"
    assert err.mean() < 0.05, f"mean err {err.mean():.3f} too large"
