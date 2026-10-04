"""Regression suite: pinned outputs for fixed seeds.

Every value below was produced by the 0.1.0 build session and is pinned here
so that a change in the plant model, the loop, the feature code or the trace
generator shows up as a test failure rather than as a silently different
number in the README. The pins are **not** independent checks of correctness —
they are change detectors. The correctness checks live in the other test files
and in ``validation/``.

If a pin fails after a deliberate change, update it in the same commit as the
change and say so in ``CHANGELOG.md``.
"""

from __future__ import annotations

import numpy as np
import pytest

from hilforge.backends import make_backend_pair
from hilforge.loop import HilLoop, LoopConfig
from hilforge.plant import AttitudePlant, PlantConfig
from hilforge.predict import TraceConfig, generate_trace
from hilforge.timing import PeriodSpec, overrun_report

PERIOD = 0.010
SEED = 20261004
N = 500

PINNED_DATA_DIGEST = "c955385f9f0522551f857b4527f3353b41cc001b574ea36351a8a17e1dad716d"
PINNED_FULL_DIGEST = "6233d8afe161c13bacca2647cf7556ba678d908bdb00e320aaed3e407499c750"
PINNED_DIRECT = 167
PINNED_CASCADE = 167
PINNED_MAXRUN = 167
PINNED_THETA_HAT_LAST = -0.0032182389227649764
PINNED_CMD_LAST = -0.0038191376287159198
PINNED_SAMPLE0 = (0.07995176313764274, -0.0019565700959251068)
PINNED_SIG_SUM = 18.659889514420314
PINNED_PLANT_STATE = (0.0025907643443245317, -0.011771693226175889)
PINNED_TRACE = {
    "jittery": {
        "mean": 0.0070087972905031647,
        "direct": 764,
        "cascade": 1109,
        "maxrun": 10,
        "util": 0.70087972905031648,
    },
    "bursty": {
        "mean": 0.0077282903634267033,
        "direct": 684,
        "cascade": 1854,
        "maxrun": 413,
        "util": 0.77282903634267031,
    },
}


def _config() -> LoopConfig:
    return LoopConfig(
        period=PeriodSpec(period_s=PERIOD),
        n_iterations=N,
        injected_durations_s=tuple(np.linspace(0.0030, 0.0135, N)),
    )


def _run(which: str):
    sim, dev = make_backend_pair(seed=SEED, sample_dt_s=PERIOD)
    return HilLoop(sim if which == "simulated" else dev, _config()).run()


@pytest.mark.parametrize("which", ["simulated", "device"])
def test_pinned_digests(which):
    record = _run(which)
    assert record.data_digest() == PINNED_DATA_DIGEST
    assert record.full_digest() == PINNED_FULL_DIGEST


@pytest.mark.parametrize("which", ["simulated", "device"])
def test_pinned_overrun_counts(which):
    record = _run(which)
    assert record.overruns.direct_count == PINNED_DIRECT
    assert record.overruns.cascade_count == PINNED_CASCADE
    assert record.overruns.max_consecutive_cascade == PINNED_MAXRUN


@pytest.mark.parametrize("which", ["simulated", "device"])
def test_pinned_signal_values(which):
    record = _run(which)
    assert record.iterations[0].sample[0] == pytest.approx(PINNED_SAMPLE0[0], rel=1e-15)
    assert record.iterations[0].sample[1] == pytest.approx(PINNED_SAMPLE0[1], rel=1e-15)
    assert record.iterations[-1].theta_hat_rad == pytest.approx(
        PINNED_THETA_HAT_LAST, rel=1e-12
    )
    assert float(record.iterations[-1].command[0]) == pytest.approx(
        PINNED_CMD_LAST, rel=1e-12
    )
    assert float(record.signal_matrix().sum()) == pytest.approx(PINNED_SIG_SUM, rel=1e-12)


def test_pinned_plant_trajectory():
    plant = AttitudePlant(PlantConfig(), seed=777)
    for _ in range(300):
        sample = plant.measure(0.01)
        plant.apply_torque(-12.0 * sample[0] - 16.8 * sample[1])
        plant.step(0.01)
    assert plant.state[0] == pytest.approx(PINNED_PLANT_STATE[0], rel=1e-12)
    assert plant.state[1] == pytest.approx(PINNED_PLANT_STATE[1], rel=1e-12)


@pytest.mark.parametrize("preset", ["jittery", "bursty"])
def test_pinned_trace_statistics(preset):
    pins = PINNED_TRACE[preset]
    cfg = TraceConfig.preset(preset, n_iterations=5000)
    trace = generate_trace(cfg, seed=99)
    account = overrun_report(trace.totals_s, cfg.period_s)
    assert float(trace.totals_s.mean()) == pytest.approx(pins["mean"], rel=1e-14)
    assert account.direct_count == pins["direct"]
    assert account.cascade_count == pins["cascade"]
    assert account.max_consecutive_cascade == pins["maxrun"]
    assert trace.utilisation == pytest.approx(pins["util"], rel=1e-14)


def test_the_two_backends_agree_on_every_pinned_value():
    """Simulation/device parity, restated as a regression invariant."""
    a = _run("simulated")
    b = _run("device")
    assert a.signal_matrix().tobytes() == b.signal_matrix().tobytes()
    assert a.full_digest() == b.full_digest()
    assert a.overruns.as_dict() == b.overruns.as_dict()
