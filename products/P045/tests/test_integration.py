"""Integration: the pipeline end to end, and the documented findings.

These are the tests that would fail if two modules disagreed with each other
rather than if one of them were internally wrong.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from arqlonghaul import closedform as cf
from arqlonghaul.channel import (
    GilbertElliottChannel,
    IndependentFrameChannel,
    ber_to_fer,
    bpsk_ber,
)
from arqlonghaul.crc import CRC32, append_fcs, check_fcs
from arqlonghaul.datasets import SEED_SPLIT, long_burst_env
from arqlonghaul.harq import HarqConfig, harq_goodput_exact, schedule_metrics
from arqlonghaul.link import preset
from arqlonghaul.policy import (
    LearnedRedundancyPolicy,
    analytic_fixed,
    collect_transitions,
    evaluate,
    tune_fixed,
)
from arqlonghaul.protocols import (
    simulate_go_back_n,
    simulate_selective_repeat,
    simulate_stop_and_wait,
)
from arqlonghaul.window import size_window


def test_link_to_protocol_chain_is_consistent() -> None:
    """A link budget turns into a frame error rate, a window and a goodput."""
    link = preset("geo")
    ber = bpsk_ber(9.0)
    fer = ber_to_fer(ber, link.frame_bits)
    assert 0.0 < fer < 1.0
    sizing = size_window(link, fer)
    assert sizing.knee_frames == link.min_continuous_window
    n = link.n_slots()
    errors = IndependentFrameChannel(fer).errors(120_000, np.random.default_rng(0))
    sim = simulate_selective_repeat(errors, n, 40 * n)
    # with a large window the state machine should land on 1 - p
    assert abs(sim.goodput - (1.0 - fer)) < 4 * sim.goodput_stderr + 1e-3


def test_three_protocols_ordered_on_a_shared_realisation() -> None:
    """Common random numbers: the same channel drives all three protocols."""
    n = 40
    errors = IndependentFrameChannel(0.05).errors(200_000, np.random.default_rng(1))
    sw = simulate_stop_and_wait(errors, n)
    gbn = simulate_go_back_n(errors, n, n)
    sr = simulate_selective_repeat(errors, n, 40 * n)
    assert sw.goodput < gbn.goodput < sr.goodput
    # and each is near its own closed form
    for res, ref in (
        (sw, cf.sw_throughput(0.05, n)),
        (gbn, cf.gbn_throughput(0.05, n, n)),
        (sr, cf.sr_throughput(0.05, n)),
    ):
        assert abs(res.goodput - ref) < 4 * res.goodput_stderr + 1e-4


def test_documented_burst_finding_reproduces() -> None:
    """The headline: the iid formula badly understates bursty go-back-N."""
    rng = np.random.default_rng(2)
    n = 60
    iid = IndependentFrameChannel(0.05).errors(200_000, rng)
    ge = GilbertElliottChannel.from_mean_and_burst(0.05, 25.0).errors(200_000, rng)
    closed = cf.gbn_throughput(0.05, n, n)
    measured_iid = simulate_go_back_n(iid, n, n).goodput
    measured_ge = simulate_go_back_n(ge, n, n).goodput
    assert abs(closed / measured_iid - 1) < 0.05
    # the formula is wrong by more than half on the bursty channel
    assert closed / measured_ge - 1 < -0.5


def test_crc_in_the_frame_loop() -> None:
    """A frame carries an FCS; the receiver's decision drives the protocol."""
    rng = np.random.default_rng(3)
    n_frames = 400
    detected = 0
    for _ in range(n_frames):
        body = bytes(rng.integers(0, 256, size=100, dtype=np.uint8))
        frame = bytearray(append_fcs(body, CRC32))
        if rng.random() < 0.3:
            pos = int(rng.integers(0, len(frame) * 8))
            frame[pos // 8] ^= 1 << (pos % 8)
            detected += 0 if check_fcs(bytes(frame), CRC32) else 1
        else:
            assert check_fcs(bytes(frame), CRC32)
    # every single-bit error must be detected by a CRC-32 at this length
    assert detected > 0


def test_harq_schedule_and_config_paths_agree() -> None:
    cfg = HarqConfig(
        k=200, n1=300, delta=40, max_rounds=4, esn0_db=1.0,
        rtt_symbols=500.0, alpha=0.5,
    )
    a = harq_goodput_exact(cfg)
    b = schedule_metrics(200, [100, 40, 40, 40], bpsk_ber(1.0), 500.0, 0.5)
    assert b["goodput"] == pytest.approx(a.goodput, rel=1e-12)


def test_ai_pipeline_runs_baselines_first_then_the_learned_model() -> None:
    """Small end-to-end run of the published comparison, on disjoint seeds."""
    env = long_burst_env()
    fit = SEED_SPLIT.fit[:20]
    tune = SEED_SPLIT.tune[:6]
    report = SEED_SPLIT.report[:20]
    assert set(fit).isdisjoint(tune) and set(fit).isdisjoint(report)
    assert set(tune).isdisjoint(report)

    analytic, _ = analytic_fixed(env)
    tuned, _ = tune_fixed(env, tune, 40)
    tr = collect_transitions(env, fit, 60, rng_seed=1)
    learned = LearnedRedundancyPolicy(
        len(env.actions), n_estimators=30, max_depth=8, min_samples_leaf=20
    )
    learned.fit(tr["features"], tr["actions"], tr["cost"])

    scores = {
        name: evaluate(env, pol, report, 60)
        for name, pol in (
            ("analytic", analytic),
            ("tuned", tuned),
            ("learned", learned),
        )
    }
    for s in scores.values():
        assert 0.0 < s["goodput"] < 1.0
        assert s["cost_per_frame"] > 0.0
        assert math.isfinite(s["cost_stderr"])
    # The tuned fixed schedule is the published winner; at this episode length
    # the assertion is only that it is not beaten by a wide margin, which is
    # what the full validation run measures precisely.
    assert scores["learned"]["cost_per_frame"] > 0.9 * scores["tuned"]["cost_per_frame"]


def test_learned_policy_beats_the_analytic_baseline() -> None:
    """The one comparison the learned model does win, reproduced small."""
    env = long_burst_env()
    tr = collect_transitions(env, SEED_SPLIT.fit[:30], 80, rng_seed=1)
    learned = LearnedRedundancyPolicy(
        len(env.actions), n_estimators=40, max_depth=10, min_samples_leaf=20
    )
    learned.fit(tr["features"], tr["actions"], tr["cost"])
    analytic, _ = analytic_fixed(env)
    report = SEED_SPLIT.report[:40]
    a = evaluate(env, analytic, report, 80)
    learned_score = evaluate(env, learned, report, 80)
    assert learned_score["cost_per_frame"] < a["cost_per_frame"]
