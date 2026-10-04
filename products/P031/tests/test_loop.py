"""Loop behaviour: stages, accounting, digests, injected timing, policies."""

from __future__ import annotations

import numpy as np
import pytest

from hilforge.backends import make_backend_pair
from hilforge.errors import ConfigurationError
from hilforge.loop import STAGES, HilLoop, LoopConfig
from hilforge.timing import PeriodSpec

PERIOD = 0.010
SEED = 20261004


def _cfg(n=100, **kwargs):
    kwargs.setdefault("injected_durations_s", tuple(np.full(n, 0.004)))
    return LoopConfig(period=PeriodSpec(period_s=PERIOD), n_iterations=n, **kwargs)


def _sim():
    sim, _ = make_backend_pair(seed=SEED, sample_dt_s=PERIOD)
    return sim


def test_run_completes_and_records_every_stage():
    record = HilLoop(_sim(), _cfg()).run()
    assert record.n_completed == 100
    assert record.aborted is False
    assert set(record.stage_histograms) == set(STAGES) | {"total"}
    for stage in STAGES:
        assert len(record.stage_histograms[stage]) == 100
    assert record.writes_issued == 100
    assert record.rehearsed_writes == 0


def test_injected_durations_are_split_by_the_stage_fractions():
    # total 0.004 s, default fractions (0.30, 0.20, 0.15, 0.35)
    record = HilLoop(_sim(), _cfg(n=5)).run()
    first = record.iterations[0]
    assert first.stage_s["sense"] == pytest.approx(0.004 * 0.30)
    assert first.stage_s["estimate"] == pytest.approx(0.004 * 0.20)
    assert first.stage_s["control"] == pytest.approx(0.004 * 0.15)
    assert first.stage_s["actuate"] == pytest.approx(0.004 * 0.35)
    assert first.total_s == pytest.approx(0.004)


def test_injected_run_is_deterministic_timing_and_measured_run_is_not():
    injected = HilLoop(_sim(), _cfg(n=20)).run()
    assert injected.deterministic_timing is True
    measured = HilLoop(
        _sim(),
        LoopConfig(period=PeriodSpec(period_s=PERIOD), n_iterations=20),
    ).run()
    assert measured.deterministic_timing is False
    assert measured.durations_s().min() >= 0.0


def test_overrun_accounting_matches_a_direct_computation():
    n = 60
    rng = np.random.default_rng(3)
    durations = tuple(rng.uniform(0.002, 0.016, size=n))
    record = HilLoop(_sim(), _cfg(n=n, injected_durations_s=durations)).run()
    expected_direct = int(np.sum(np.asarray(durations) > PERIOD))
    assert record.overruns is not None
    assert record.overruns.direct_count == expected_direct
    assert sum(1 for it in record.iterations if it.direct_overrun) == expected_direct


def test_digests_are_stable_and_timing_free():
    a = HilLoop(_sim(), _cfg(n=50)).run()
    b = HilLoop(_sim(), _cfg(n=50)).run()
    assert a.data_digest() == b.data_digest()
    # Different injected timing, same data path -> same data digest, different full.
    c = HilLoop(_sim(), _cfg(n=50, injected_durations_s=tuple(np.full(50, 0.006)))).run()
    assert c.data_digest() == a.data_digest()
    assert c.full_digest() != a.full_digest()


def test_signal_matrix_shape_and_content():
    record = HilLoop(_sim(), _cfg(n=30)).run()
    matrix = record.signal_matrix()
    assert matrix.shape == (30, 5)
    assert np.all(np.isfinite(matrix))


def test_record_signals_off_disables_the_signal_matrix():
    record = HilLoop(_sim(), _cfg(n=10, record_signals=False)).run()
    assert record.iterations[0].sample is None
    with pytest.raises(ValueError, match="record_signals"):
        record.signal_matrix()


def test_stage_durations_accessor_and_unknown_stage():
    record = HilLoop(_sim(), _cfg(n=10)).run()
    assert record.stage_durations_s("sense").size == 10
    with pytest.raises(KeyError):
        record.stage_durations_s("transmit")


def test_summary_contains_the_accounting_keys():
    record = HilLoop(_sim(), _cfg(n=10)).run()
    summary = record.summary()
    for key in (
        "backend",
        "is_hardware",
        "deterministic_timing",
        "direct_overruns",
        "cascade_overruns",
        "writes_issued",
    ):
        assert key in summary
    assert summary["is_hardware"] is False


def test_complementary_filter_tracks_the_true_attitude():
    sim = _sim()
    record = HilLoop(sim, _cfg(n=400)).run()
    # The plant starts at 0.08 rad and the loop drives it toward zero, so the
    # estimate must end up small; the bias leaves a small offset.
    assert abs(record.iterations[-1].theta_hat_rad) < 0.02


def test_backend_left_closed_if_it_was_closed():
    sim = _sim()
    assert sim.is_open is False
    HilLoop(sim, _cfg(n=5)).run()
    assert sim.is_open is False
    sim.open()
    HilLoop(sim, _cfg(n=5)).run()
    assert sim.is_open is True
    sim.close()


def test_loop_config_validation():
    spec = PeriodSpec(period_s=PERIOD)
    with pytest.raises(ConfigurationError):
        LoopConfig(period=spec, n_iterations=0)
    with pytest.raises(ConfigurationError):
        LoopConfig(period=spec, n_iterations=5, filter_gyro_weight=1.0)
    with pytest.raises(ConfigurationError):
        LoopConfig(period=spec, n_iterations=5, on_dropped_sample="ignore")
    with pytest.raises(ConfigurationError):
        LoopConfig(period=spec, n_iterations=5, on_write_rejected="ignore")
    with pytest.raises(ConfigurationError):
        LoopConfig(period=spec, n_iterations=5, on_cascade="shrug")
    with pytest.raises(ConfigurationError):
        LoopConfig(period=spec, n_iterations=5, injected_durations_s=(0.001, 0.002))
    with pytest.raises(ConfigurationError):
        LoopConfig(period=spec, n_iterations=2, injected_durations_s=(0.001, -0.002))
    with pytest.raises(ConfigurationError):
        LoopConfig(period=spec, n_iterations=5, injected_stage_fractions=(0.5, 0.5))
    with pytest.raises(ConfigurationError):
        LoopConfig(
            period=spec, n_iterations=5, injected_stage_fractions=(0.3, 0.3, 0.3, 0.3)
        )
    with pytest.raises(ConfigurationError):
        LoopConfig(period=spec, n_iterations=5, shed_factor=0.0)
    with pytest.raises(ConfigurationError):
        LoopConfig(period=spec, n_iterations=5, flagger_history=0)


def test_flagger_is_consulted_and_sheds_the_estimate_stage():
    calls = {"n": 0}

    def flagger(history: np.ndarray) -> bool:
        calls["n"] += 1
        return bool(history[-1] > 0.008)

    n = 40
    durations = tuple(np.where(np.arange(n) % 4 == 0, 0.012, 0.004))
    cfg = _cfg(
        n=n,
        injected_durations_s=durations,
        overrun_flagger=flagger,
        shed_factor=0.25,
    )
    record = HilLoop(_sim(), cfg).run()
    assert calls["n"] == n - 1  # not called on the first iteration
    assert record.flagged_iterations > 0
    assert record.shed_iterations == record.flagged_iterations
    shed = [it for it in record.iterations if it.shed]
    assert shed
    # Shedding reduces the estimate stage to a quarter of its injected share.
    it = shed[0]
    assert it.stage_s["estimate"] == pytest.approx(
        durations[it.index] * 0.20 * 0.25
    )
    assert it.total_s < durations[it.index]


def test_shed_factor_one_records_flags_without_shedding():
    n = 20
    cfg = _cfg(n=n, overrun_flagger=lambda h: True, shed_factor=1.0)
    record = HilLoop(_sim(), cfg).run()
    assert record.flagged_iterations == n - 1
    assert record.shed_iterations == 0


def test_deadline_shorter_than_period_tightens_the_count():
    n = 40
    durations = tuple(np.full(n, 0.007))
    loose = HilLoop(_sim(), _cfg(n=n, injected_durations_s=durations)).run()
    tight = HilLoop(
        _sim(),
        LoopConfig(
            period=PeriodSpec(period_s=PERIOD, deadline_s=0.005),
            n_iterations=n,
            injected_durations_s=durations,
        ),
    ).run()
    assert loose.overruns.direct_count == 0
    assert tight.overruns.direct_count == n


def test_every_aborting_exception_carries_the_partial_record():
    """An aborted run stays inspectable through the exception's ``record``."""
    from hilforge.errors import OverrunCascadeError

    n = 20
    cfg = LoopConfig(
        period=PeriodSpec(period_s=PERIOD, cascade_limit=2),
        n_iterations=n,
        injected_durations_s=tuple(np.full(n, 1.5 * PERIOD)),
        on_cascade="abort",
    )
    with pytest.raises(OverrunCascadeError) as exc:
        HilLoop(_sim(), cfg).run()
    record = exc.value.record
    assert record.aborted is True
    assert record.n_completed == 2
    assert record.overruns is not None


def test_on_disconnect_is_validated():
    with pytest.raises(ConfigurationError):
        LoopConfig(
            period=PeriodSpec(period_s=PERIOD), n_iterations=5, on_disconnect="ignore"
        )
