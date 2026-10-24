"""Simulation/device parity, asserted by test, with no hardware present.

The requirement is that the same estimator code paths run against a simulated
clock and a real one, and that parity is asserted by test. The device side is
exercised through the loopback stub
(:class:`~timetransfer.backends.stubs.LoopbackDriver`), so this test runs in a
container with no board attached -- which is the only way it could run at all,
and is also the reason it says nothing about timing on a board.

What is compared
----------------
* The four timestamp arrays, **byte for byte**. They are ``int64``
  nanoseconds, so there is no rounding to excuse a difference.
* The per-exchange offset and delay estimates, by maximum absolute difference
  *and* by bytes. The byte comparison is legitimate here, and not a violation
  of the rule against bit-exact floating-point assertions, for a specific
  reason: the two paths execute the *identical* sequence of operations on the
  *identical* inputs. Nothing is being claimed about floating-point
  associativity. The tolerance comparison is asserted first and is the primary
  claim; the byte comparison is an additional, stronger statement that the
  code paths have not diverged.
* The least-squares parameters and the Kalman final state, to tolerance.
* The loss accounting and the completion counts, exactly (integers).
"""

from __future__ import annotations

import numpy as np
import pytest

from timetransfer.acquire import acquire
from timetransfer.backends import make_backend_pair
from timetransfer.skew import KalmanConfig, fit_least_squares, kalman_skew
from timetransfer.twoway import two_way_estimate

from conftest import make_spec

CASES = (
    ("symmetric, no jitter, no loss", {"asymmetry_s": 0.0, "jitter_scale_s": 0.0}),
    ("asymmetric with jitter", {"asymmetry_s": 8.0e-4, "jitter_scale_s": 2.0e-4}),
    (
        "lossy with clock noise and drift",
        {
            "asymmetry_s": -4.0e-4,
            "loss_probability": 0.12,
            "q1": 1.0e-20,
            "q2": 1.0e-26,
            "drift_per_s": 2.0e-12,
            "timestamp_noise_s": 2.0e-8,
        },
    ),
    ("coarse 1 us timestamps", {"resolution_ns": 1000, "timestamp_noise_s": 5.0e-7}),
)

N = 400


@pytest.mark.parametrize(("label", "kwargs"), CASES, ids=[c[0] for c in CASES])
def test_timestamps_are_byte_identical(label: str, kwargs: dict[str, object]) -> None:
    spec = make_spec(seed=20261024, **kwargs)
    sim, dev = make_backend_pair(spec)
    sim.open()
    dev.open()
    a = acquire(sim, N, min_exchanges=8).exchanges
    b = acquire(dev, N, min_exchanges=8).exchanges
    sim.close()
    dev.close()
    assert a.n == b.n, label
    assert a.n_lost == b.n_lost, label
    assert a.seq.tolist() == b.seq.tolist(), label
    assert a.timestamp_bytes() == b.timestamp_bytes(), label


@pytest.mark.parametrize(("label", "kwargs"), CASES, ids=[c[0] for c in CASES])
def test_estimator_outputs_agree(label: str, kwargs: dict[str, object]) -> None:
    spec = make_spec(seed=20261024, **kwargs)
    sim, dev = make_backend_pair(spec)
    sim.open()
    dev.open()
    a = acquire(sim, N, min_exchanges=8).exchanges
    b = acquire(dev, N, min_exchanges=8).exchanges
    sim.close()
    dev.close()

    est_a = two_way_estimate(a, assumed_skew_s_per_s=2.0e-6)
    est_b = two_way_estimate(b, assumed_skew_s_per_s=2.0e-6)
    # Primary claim: agreement to tolerance, on both sides.
    assert est_a.offset_s == pytest.approx(est_b.offset_s, abs=1e-18, rel=1e-15), label
    assert est_a.delay_s == pytest.approx(est_b.delay_s, abs=1e-18, rel=1e-15), label
    # Stronger additional claim: identical operations on identical inputs.
    assert est_a.offset_s.tobytes() == est_b.offset_s.tobytes(), label
    assert float(np.max(np.abs(est_a.offset_s - est_b.offset_s))) == 0.0, label

    fit_a = fit_least_squares(est_a.midpoint_s, est_a.offset_s, order=2)
    fit_b = fit_least_squares(est_b.midpoint_s, est_b.offset_s, order=2)
    assert fit_a.params == pytest.approx(fit_b.params, abs=1e-24, rel=1e-12), label
    assert fit_a.sigma_s == pytest.approx(fit_b.sigma_s, rel=1e-12), label

    cfg = KalmanConfig(q1_s2_per_s=1e-20, q2_per_s=1e-26)
    kal_a = kalman_skew(est_a.midpoint_s, est_a.offset_s, config=cfg)
    kal_b = kalman_skew(est_b.midpoint_s, est_b.offset_s, config=cfg)
    assert kal_a.final_offset_s == pytest.approx(kal_b.final_offset_s, rel=1e-12), label
    assert kal_a.final_skew_s_per_s == pytest.approx(
        kal_b.final_skew_s_per_s, abs=1e-20, rel=1e-12
    ), label


def test_parity_holds_for_several_seeds() -> None:
    for seed in (1, 7, 101, 20261024):
        spec = make_spec(seed=seed, asymmetry_s=5.0e-4, loss_probability=0.08, q1=1e-20)
        sim, dev = make_backend_pair(spec)
        sim.open()
        dev.open()
        a = acquire(sim, 150, min_exchanges=8).exchanges
        b = acquire(dev, 150, min_exchanges=8).exchanges
        sim.close()
        dev.close()
        assert a.timestamp_bytes() == b.timestamp_bytes(), f"seed {seed}"


def test_parity_is_a_falsifiable_claim() -> None:
    # A different seed must break parity. A test that cannot fail is not a test,
    # and the only thing that makes the byte comparison above meaningful is that
    # it distinguishes.
    sim, _ = make_backend_pair(make_spec(seed=1))
    _, dev = make_backend_pair(make_spec(seed=2))
    sim.open()
    dev.open()
    a = acquire(sim, 60, min_exchanges=8).exchanges
    b = acquire(dev, 60, min_exchanges=8).exchanges
    sim.close()
    dev.close()
    assert a.timestamp_bytes() != b.timestamp_bytes()


def test_the_device_side_declares_itself_a_stub() -> None:
    sim, dev = make_backend_pair(make_spec())
    dev.open()
    assert dev.info.kind == "device"
    assert dev.info.driver == "loopback"
    assert dev.info.is_hardware is False
    assert dev.info.detail["is_hardware"] == "false"
    assert "no device is present" in dev.info.detail["note"]
    assert sim.info.kind == "simulated"
    dev.close()
