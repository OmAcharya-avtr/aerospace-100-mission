"""The clock polynomial, its noise processes and its snapshot/restore."""

from __future__ import annotations

import numpy as np
import pytest

from timetransfer.clockmodel import ClockModel, ClockSpec
from timetransfer.errors import ConfigurationError, TimebaseRegressionError


def test_deterministic_offset_known_answer() -> None:
    # Hand calculation at t = 100 s, t0 = 0:
    #   theta0 = 1e-3, a = 2e-6 s/s, D = 1e-11 1/s
    #   theta = 1e-3 + 2e-6*100 + 0.5*1e-11*100^2
    #         = 1e-3 + 2e-4 + 5e-8 = 1.20005e-3 s
    spec = ClockSpec(offset0_s=1.0e-3, skew_s_per_s=2.0e-6, drift_per_s=1.0e-11)
    assert spec.deterministic_offset_s(100.0) == pytest.approx(1.20005e-3, rel=1e-12)


def test_deterministic_skew_known_answer() -> None:
    # a(t) = a0 + D t = 2e-6 + 1e-11 * 100 = 2e-6 + 1e-9 = 2.001e-6 s/s
    spec = ClockSpec(skew_s_per_s=2.0e-6, drift_per_s=1.0e-11)
    assert spec.deterministic_skew_s_per_s(100.0) == pytest.approx(2.001e-6, rel=1e-12)


def test_deterministic_offset_vectorises() -> None:
    spec = ClockSpec(offset0_s=1.0, skew_s_per_s=0.5)
    out = spec.deterministic_offset_s(np.array([0.0, 1.0, 2.0]))
    assert isinstance(out, np.ndarray)
    assert out == pytest.approx(np.array([1.0, 1.5, 2.0]))


def test_allan_variance_known_answer() -> None:
    # sigma_y^2(tau) = q1/tau + q2*tau/3 (Zucca and Tavella 2005).
    # q1 = 1e-20, q2 = 3e-26, tau = 10 -> 1e-21 + 1e-25 = 1.001e-21
    spec = ClockSpec(q1_s2_per_s=1.0e-20, q2_per_s=3.0e-26)
    assert spec.allan_variance(10.0) == pytest.approx(1.001e-21, rel=1e-12)


def test_allan_variance_rejects_non_positive_tau() -> None:
    with pytest.raises(ConfigurationError, match="tau_s"):
        ClockSpec(q1_s2_per_s=1e-20).allan_variance(0.0)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"q1_s2_per_s": -1.0}, "q1_s2_per_s"),
        ({"q2_per_s": -1.0}, "q2_per_s"),
        ({"offset0_s": float("nan")}, "offset0_s"),
        ({"skew_s_per_s": 1.0}, "skew_s_per_s"),
        ({"skew_s_per_s": -2.0}, "skew_s_per_s"),
    ],
)
def test_clock_spec_validates(kwargs: dict[str, float], match: str) -> None:
    with pytest.raises(ConfigurationError, match=match):
        ClockSpec(**kwargs)


def test_noiseless_model_returns_the_polynomial_exactly() -> None:
    spec = ClockSpec(offset0_s=1.0e-3, skew_s_per_s=2.0e-6)
    model = ClockModel(spec, seed=1)
    assert model.offset_at(50.0) == pytest.approx(spec.deterministic_offset_s(50.0), rel=1e-15)
    assert model.noise_s() == 0.0


def test_zero_length_advance_consumes_no_randomness() -> None:
    model = ClockModel(ClockSpec(q1_s2_per_s=1e-18), seed=3)
    model.advance_to(10.0)
    before = model.snapshot()["bit_generator_state"]
    model.advance_to(10.0)
    assert model.snapshot()["bit_generator_state"] == before


def test_model_refuses_to_go_backwards() -> None:
    model = ClockModel(ClockSpec(), seed=0)
    model.advance_to(5.0)
    with pytest.raises(TimebaseRegressionError, match="backwards"):
        model.advance_to(4.999)


def test_snapshot_restore_reproduces_the_realisation() -> None:
    model = ClockModel(ClockSpec(q1_s2_per_s=1e-18, q2_per_s=1e-24), seed=7)
    model.advance_to(10.0)
    snap = model.snapshot()
    first = [model.offset_at(t) for t in (20.0, 30.0, 40.0)]
    model.restore(snap)
    second = [model.offset_at(t) for t in (20.0, 30.0, 40.0)]
    assert first == second


def test_white_fm_variance_grows_linearly_in_time() -> None:
    # Var[n(t)] = q1 * t for a Wiener process of intensity q1. Measured over
    # 400 seeded realisations and compared with a generous tolerance, because
    # the sampling error of a variance from 400 samples is about 7 per cent.
    q1 = 1.0e-18
    horizon = 100.0
    values = []
    for seed in range(400):
        model = ClockModel(ClockSpec(q1_s2_per_s=q1), seed=seed)
        model.advance_to(horizon)
        values.append(model.noise_s())
    measured = float(np.var(np.asarray(values), ddof=1))
    assert measured == pytest.approx(q1 * horizon, rel=0.20)
