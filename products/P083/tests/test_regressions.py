"""Frozen values and pinned counterexamples.

Two kinds of test live here.

**Frozen values.** Model constants and a whole declared run's verdict, pinned
to twelve significant figures, so that a silent change in a library default, a
random-number stream or a quantile rule is caught by a test rather than
discovered in a validation-output diff. None of these numbers is a reference
value from outside this repository; each is what the code produced on the
declared input, and a test that fails here means the behaviour changed, which
may be a defect or may be an intended change that has to be re-validated.

**Pinned counterexamples.** ``.hypothesis/`` is not committed, so a falsifying
example found by one property-test run is lost on a fresh clone. Every
counterexample Hypothesis found during this product's build is written down
here with its literal inputs.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from hilverdict.certificate import (
    CertificateModel,
    certified_overrun_s,
    certify_run,
    exponential_integral,
)
from hilverdict.plant import LoopModel
from hilverdict.resimulate import resimulate_run
from hilverdict.synthetic import generate_run
from hilverdict.timeline import overrun_counts
from hilverdict.verdict import NOT_TIMING_FAITHFUL, run_verdict

REGRESSION_SEED = 902
REGRESSION_PROFILE = "loaded"
REGRESSION_AMPLITUDE_DEG = 8.0
REGRESSION_STEPS = 200


@pytest.fixture(scope="module")
def frozen_model() -> CertificateModel:
    return CertificateModel.build(LoopModel(), REGRESSION_STEPS)


@pytest.fixture(scope="module")
def frozen_run():
    return generate_run(
        seed=REGRESSION_SEED,
        profile=REGRESSION_PROFILE,
        amplitude_deg=REGRESSION_AMPLITUDE_DEG,
        steps=REGRESSION_STEPS,
    )


def test_frozen_log_norm(frozen_model):
    assert frozen_model.mu_s == pytest.approx(98.5777000448, rel=1e-12, abs=0.0)


def test_frozen_input_gain(frozen_model):
    assert frozen_model.input_gain == pytest.approx(
        1208.65579292, rel=1e-11, abs=0.0
    )


def test_frozen_free_factor(frozen_model):
    assert frozen_model.free_factor == pytest.approx(
        2.67989335400, rel=1e-11, abs=0.0
    )


def test_frozen_horizon_gain(frozen_model):
    assert frozen_model.gamma_sup(REGRESSION_STEPS) == pytest.approx(
        3.20389916623, rel=1e-11, abs=0.0
    )


def test_frozen_horizon_gain_peaks_at_step_eight(frozen_model):
    assert int(np.argmax(frozen_model.gains)) == 8


def test_frozen_spectral_radius():
    assert LoopModel().spectral_radius() == pytest.approx(
        0.984651506648, rel=1e-11, abs=0.0
    )


def test_frozen_monodromy():
    expected = np.array(
        [
            [0.987051969817993, 0.005714201454389082],
            [-0.1323628203471301, 0.6695673887310939],
        ]
    )
    assert LoopModel().monodromy() == pytest.approx(expected, rel=1e-12, abs=1e-18)


def test_frozen_certified_overrun_at_one_degree(frozen_model):
    assert certified_overrun_s(
        frozen_model, math.radians(1.0), REGRESSION_STEPS, 1.0
    ) == pytest.approx(0.00440811166937, rel=1e-11, abs=0.0)


def test_frozen_miss_count(frozen_run):
    assert len(certify_run(frozen_run, 1.0).certificates) == 22


def test_frozen_dismissal_count(frozen_run):
    certificate = certify_run(frozen_run, 1.0)
    assert len(certificate.dismissed) == 19
    assert len(certificate.undecided) == 3


def test_frozen_undecided_indices(frozen_run):
    certificate = certify_run(frozen_run, 1.0)
    assert [item.event.index for item in certificate.undecided] == [110, 111, 174]


def test_frozen_aggregate_bound(frozen_run):
    assert certify_run(frozen_run, 1.0).aggregate_bound == pytest.approx(
        378.177349876, rel=1e-11, abs=0.0
    )


def test_frozen_run_certified_overrun(frozen_run):
    assert certify_run(frozen_run, 1.0).certified_overrun_s == pytest.approx(
        0.000200900177596, rel=1e-11, abs=0.0
    )


def test_frozen_overrun_counts(frozen_run):
    counts = overrun_counts(frozen_run)
    assert counts.direct == 22
    assert counts.cascade == 55


def test_frozen_consequential_indices(frozen_run):
    results = resimulate_run(frozen_run, 1.0)
    assert [r.event.index for r in results if r.consequential] == [111, 174]


def test_frozen_peak_divergences(frozen_run):
    results = {r.event.index: r for r in resimulate_run(frozen_run, 1.0)}
    assert results[111].peak_divergence == pytest.approx(
        22.9675253837, rel=1e-9, abs=0.0
    )
    assert results[174].peak_divergence == pytest.approx(
        1.77629756260, rel=1e-9, abs=0.0
    )


def test_frozen_peak_index(frozen_run):
    results = {r.event.index: r for r in resimulate_run(frozen_run, 1.0)}
    assert results[111].peak_index == 120


def test_frozen_plant_call_count(frozen_run):
    results = resimulate_run(frozen_run, 1.0)
    assert sum(r.plant_calls for r in results) == 32576


def test_frozen_run_verdict(frozen_run):
    assert run_verdict(frozen_run, 1.0).verdict == NOT_TIMING_FAITHFUL


def test_frozen_zero_bound_misses_have_a_zero_input_step(frozen_run):
    certificate = certify_run(frozen_run, 1.0)
    zero = [item for item in certificate.certificates if item.bound == 0.0]
    assert len(zero) == 3
    assert all(item.event.input_step_rad == 0.0 for item in zero)


def test_pinned_counterexample_subnormal_negative_mu_one_ulp():
    """Hypothesis, build of 2026-11-02.

    ``expm1(mu t)/mu`` at ``mu = -1.1125369292536007e-308`` and
    ``t = 0.015045340851328677`` returned 0.015045340851328692, one unit in
    the last place ABOVE ``t``, which breaks the sign statement of equation
    (9) for negative ``mu``. The series branch fixes it.
    """
    mu = -1.1125369292536007e-308
    duration = 0.015045340851328677
    assert exponential_integral(mu, duration) <= duration


def test_pinned_counterexample_smallest_subnormal_mu_underflow():
    """Hypothesis, build of 2026-11-02.

    At ``mu = 5e-324`` and ``t = 0.03125``, ``expm1(mu t)`` underflows to
    exactly zero and the quotient returned 0.0 instead of ``t``: a total loss
    rather than a rounding error. The series branch returns ``t`` exactly.
    """
    assert exponential_integral(5e-324, 0.03125) == pytest.approx(
        0.03125, rel=1e-15, abs=0.0
    )


def test_pinned_counterexample_series_branch_boundary():
    """The series branch and the expm1 branch must agree across the switch.

    The switch is at ``|mu t| = 1e-8``. Both branches are evaluated either
    side of it and must agree to within one unit in the last place, or the
    function has a discontinuity where it changes method.
    """
    duration = 0.01
    below = exponential_integral(1e-6 / duration * 0.999, duration)
    above = exponential_integral(1e-6 / duration * 1.001, duration)
    assert below == pytest.approx(above, rel=1e-8, abs=0.0)
    boundary_low = exponential_integral(0.999e-8 / duration, duration)
    boundary_high = exponential_integral(1.001e-8 / duration, duration)
    assert boundary_low == pytest.approx(boundary_high, rel=1e-10, abs=0.0)


def test_exponential_integral_series_branch_matches_expm1_in_the_ordinary_range():
    for product in (1e-7, 1e-5, 1e-3, 1e-1, 1.0):
        duration = 0.01
        mu = product / duration
        assert exponential_integral(mu, duration) == pytest.approx(
            float(np.expm1(product) / mu), rel=1e-14, abs=0.0
        )
