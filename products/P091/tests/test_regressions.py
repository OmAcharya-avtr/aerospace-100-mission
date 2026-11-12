"""Defects found while building this product, pinned with their literal inputs.

Each test below is a thing that was actually wrong, with the inputs that showed
it. They are pinned here because the Hypothesis example database is disabled and
because a defect found once and not pinned is a defect that returns.
"""

from __future__ import annotations

import math
import warnings

import numpy as np
import pytest

from thermalbudget import schema
from thermalbudget.baselines import burst_baseline, overstatement, sustained_truth
from thermalbudget.equilibrium import equilibrium_test
from thermalbudget.provenance import Origin
from thermalbudget.rc_model import fit_exponential_relaxation, tau_from_log_differences
from thermalbudget.trace import load_trace

from fixtures import synthetic_trace as st


def test_regression_log_difference_tau_returned_451_s_for_a_true_90_s():
    """Measured defect. Selecting difference pairs individually on their own
    magnitude keeps the upward half of the noise beyond about three time
    constants and flattens the fitted log slope. On this literal curve -- true
    tau 90 s, asymptote 68 degC, initial 45 degC, 720 s observed, Gaussian noise
    of 0.035 degC at seed 31 -- the unguarded route returned 450.97 s. The
    guarded route confines the fit to a contiguous signal window.
    """
    t = np.arange(1.0, 721.0)
    truth = 68.0 + (45.0 - 68.0) * np.exp(-t / 90.0)
    noisy = truth + np.random.default_rng(31).normal(0.0, 0.035, size=t.size)
    guarded = tau_from_log_differences(t, noisy)
    assert guarded.tau_s < 150.0
    assert guarded.tau_s == pytest.approx(90.0, rel=0.25, abs=0.0)
    # The unguarded behaviour, reproduced to show the guard is what fixes it.
    differences = np.diff(noisy)
    midpoints = 0.5 * (t[1:] + t[:-1])
    positive = differences > 0.0
    logs = np.log(differences[positive])
    times = midpoints[positive]
    centred = times - times.mean()
    slope = float(centred @ (logs - logs.mean()) / float(centred @ centred))
    assert -1.0 / slope > 300.0


def test_regression_an_all_cores_missing_row_warned_from_nanmean(tmp_path):
    """Measured defect. ``numpy.nanmean`` over a row in which every core clock is
    the ``-1`` sentinel emits ``RuntimeWarning: Mean of empty slice``. The schema
    allows that row, so a warning on it trains a reader to ignore warnings. The
    mean is now an explicit sum and count.
    """
    header = ",".join(schema.expected_columns())
    good = ("1794457189.964,1.001,0,19447.34,0.7512,0.38,20.10,45875,46000,45812,45125,"
            "44531,45875,1728000,1728000,1728000,1728000,1728000,1728000,306000000")
    missing = ("1794457191.0,2.051,32760,31200.88,1.1942,0.40,-1,-1,-1,-1,-1,-1,-1,"
               "-1,-1,-1,-1,-1,-1,-1")
    path = tmp_path / "soak_1.00_r1.csv"
    path.write_text(f"{header}\n{good}\n{missing}\n")
    trace = load_trace(path, origin=Origin.MEASURED)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        clocks = trace.mean_cpu_khz
    assert math.isnan(clocks[1])
    assert clocks[0] == pytest.approx(1728000.0, rel=0.0, abs=1e-9)


def test_regression_a_cooldown_starting_at_the_idle_floor_had_no_decay(tmp_path):
    """Measured defect in the fixture generator. Each run started from the idle
    floor, so every ``cool_*`` trace was flat and the independent cooldown tau
    had nothing to estimate from -- the log-difference route refused on all eight
    of them. The generator now threads the die temperature across runs.
    """
    campaign = st.load_synthetic_campaign(tmp_path, regime="no_throttle")
    fit = fit_exponential_relaxation(
        campaign["cool_r1"].t_rel, campaign["cool_r1"].die_temperature_c
    )
    assert fit.amplitude_c > 10.0
    assert fit.tau_s > 50.0


def test_regression_a_twenty_second_burst_must_not_report_equilibrated(tmp_path):
    """A 20 s run has 20 one-second samples, below the declared 30-sample
    minimum, and is nowhere near equilibrium. The verdict is
    ``INSUFFICIENT_SAMPLES``, which is explicitly NOT a pass; an earlier draft
    that returned a slope over 20 points would have called it equilibrated.
    """
    path = st.write_run(tmp_path, "burst_1.00_a", 1.0, 20, regime="no_throttle")
    verdict = equilibrium_test(load_trace(path, origin=Origin.SYNTHETIC))
    assert verdict.status == "INSUFFICIENT_SAMPLES"
    assert not verdict.equilibrated


def test_regression_the_two_headline_forms_differ_by_percentile_interpolation(tmp_path):
    """Measured non-identity, not a rounding artefact. ``numpy.percentile``
    interpolates linearly between order statistics, and a monotone nonlinear
    transform does not commute with linear interpolation, so the sustained
    fraction's interval and the overstatement's interval agree to about 1.6e-8
    relative rather than exactly. Asserted with a tolerance and with its reason.
    """
    campaign = st.load_synthetic_campaign(tmp_path, regime="throttled")
    base1 = burst_baseline(campaign.bursts(), n_resamples=2000)
    base3 = sustained_truth(campaign.soaks(1.0), n_resamples=2000)
    head = overstatement(base1.throughput_groups, base3.throughput_groups, n_resamples=2000)
    transformed = 1.0 / head.sustained_fraction.ci_high - 1.0
    assert transformed == pytest.approx(head.overstatement.ci_low, rel=1e-6, abs=0.0)
    assert transformed != head.overstatement.ci_low


def test_regression_the_missing_sentinel_must_not_become_minus_one_milli_degree(tmp_path):
    """The sentinel is ``-1`` in milli-degrees. Dividing it by 1000 gives
    ``-0.001 degC``, a plausible-looking number that would drag a mean towards
    zero and bias a slope. It must become ``nan`` BEFORE the unit conversion.
    """
    header = ",".join(schema.expected_columns())
    rows = [
        "1794457189.964,1.001,0,19447.34,0.7512,0.38,20.10,45875,46000,45812,45125,44531,"
        "-1,1728000,1728000,1728000,1728000,1728000,1728000,306000000",
        "1794457191.0,2.051,32760,31200.88,1.1942,0.40,20.10,46343,46400,46312,45625,45031,"
        "46343,1728000,1728000,1728000,1728000,1728000,1728000,306000000",
    ]
    path = tmp_path / "soak_1.00_r1.csv"
    path.write_text(header + "\n" + "\n".join(rows) + "\n")
    trace = load_trace(path, origin=Origin.MEASURED)
    assert math.isnan(trace.die_temperature_c[0])
    assert trace.die_temperature_c[0] != pytest.approx(-0.001, rel=0.0, abs=1e-12)
    assert float(trace.finite(schema.DIE_COLUMN).mean()) == pytest.approx(
        46.343, rel=0.0, abs=1e-12
    )
