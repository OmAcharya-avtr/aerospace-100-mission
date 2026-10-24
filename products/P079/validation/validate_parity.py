"""Check 5 -- HAL simulation/device parity, asserted with no hardware present.

Requirement (BATCH_08_SPEC, P079, L4 scaffolding item 1): "Hardware abstraction
layer over the timebase with simulation/device parity -- the same estimator
code paths run against a simulated clock and a real one, and parity is itself
asserted by test."

What is compared
----------------
A :class:`~timetransfer.backends.simulated.SimulatedBackend` and a
:class:`~timetransfer.backends.device.DeviceBackend` over
:class:`~timetransfer.backends.stubs.LoopbackDriver`, both from the same
:class:`~timetransfer.backends.simulated.SimulationSpec` and both on a virtual
timebase:

* the four timestamp arrays, **byte for byte** -- they are ``int64``
  nanoseconds, so there is no rounding to excuse a difference;
* the per-exchange offset and delay estimates, by maximum absolute difference;
* the least-squares parameters and the Kalman final state;
* the loss accounting and the completion counts.

What this does NOT establish
----------------------------
The device backend's only driver is a loopback stub over the simulation engine.
It is **not hardware**, ``is_hardware`` is ``False`` on both sides, and this
check says nothing about timing on a board. It establishes that the HAL path,
the acquisition, the estimators and the accounting behave identically whichever
backend they are handed -- which is the thing a board run would otherwise
discover the hard way.

A falsifying control is included: two backends built from different seeds must
differ. A comparison that cannot fail is not evidence.

Run: ``python validation/validate_parity.py``
"""

from __future__ import annotations

import sys

import numpy as np

from _bootstrap import Tee, bootstrap, environment_lines, scaled

ROOT = bootstrap()

from timetransfer.acquire import acquire  # noqa: E402
from timetransfer.backends import SimulationSpec, make_backend_pair  # noqa: E402
from timetransfer.clockmodel import ClockSpec  # noqa: E402
from timetransfer.linkmodel import LinkSpec  # noqa: E402
from timetransfer.skew import KalmanConfig, fit_least_squares, kalman_skew  # noqa: E402
from timetransfer.twoway import two_way_estimate  # noqa: E402

N = scaled(500, minimum=30)
SEED = 20261024

CASES: tuple[tuple[str, SimulationSpec], ...] = (
    (
        "symmetric, no jitter, no loss",
        SimulationSpec(
            clock=ClockSpec(offset0_s=1.0e-3, skew_s_per_s=2.0e-6),
            link=LinkSpec(jitter_scale_ab_s=0.0, jitter_scale_ba_s=0.0),
            period_s=1.0,
            seed=SEED,
        ),
    ),
    (
        "asymmetric with jitter",
        SimulationSpec(
            clock=ClockSpec(offset0_s=1.0e-3, skew_s_per_s=2.0e-6, q1_s2_per_s=1.0e-20),
            link=LinkSpec(floor_ab_s=5.4e-3, floor_ba_s=4.6e-3, jitter_shape_ba=1.6),
            period_s=1.0,
            timestamp_noise_s=2.0e-8,
            seed=SEED,
        ),
    ),
    (
        "lossy with an outage and drift",
        SimulationSpec(
            clock=ClockSpec(
                offset0_s=-3.0e-3,
                skew_s_per_s=-1.1e-6,
                drift_per_s=2.0e-12,
                q1_s2_per_s=1.0e-20,
                q2_per_s=1.0e-26,
            ),
            link=LinkSpec(
                floor_ab_s=4.6e-3,
                floor_ba_s=5.4e-3,
                loss_probability=0.1,
                outage_windows=((200.0, 220.0),),
            ),
            period_s=1.0,
            timestamp_noise_s=2.0e-8,
            seed=SEED,
        ),
    ),
    (
        "coarse 1 us timestamps",
        SimulationSpec(
            clock=ClockSpec(offset0_s=1.0e-3, skew_s_per_s=2.0e-6),
            link=LinkSpec(floor_ab_s=5.0e-3, floor_ba_s=5.0e-3),
            period_s=0.1,
            timestamp_noise_s=5.0e-7,
            resolution_ns=1000,
            seed=SEED,
        ),
    ),
)


def main() -> int:
    out = Tee("validate_parity_output.txt")
    out("timetransfer validation 5 -- simulation/device parity through the HAL")
    out("=" * 78)
    for line in environment_lines():
        out(line)
    out("")
    out(f"exchanges attempted per case : {N}")
    out("device backend driver        : loopback stub over the simulation engine")
    out("                               (NOT hardware; is_hardware is False)")
    failures = 0

    for label, spec in CASES:
        sim, dev = make_backend_pair(spec)
        sim.open()
        dev.open()
        a = acquire(sim, N, min_exchanges=8).exchanges
        b = acquire(dev, N, min_exchanges=8).exchanges
        sim.close()
        dev.close()

        est_a = two_way_estimate(a, assumed_skew_s_per_s=spec.clock.skew_s_per_s)
        est_b = two_way_estimate(b, assumed_skew_s_per_s=spec.clock.skew_s_per_s)
        fit_a = fit_least_squares(est_a.midpoint_s, est_a.offset_s, order=2)
        fit_b = fit_least_squares(est_b.midpoint_s, est_b.offset_s, order=2)
        cfg = KalmanConfig(q1_s2_per_s=1.0e-20, q2_per_s=1.0e-26)
        kal_a = kalman_skew(est_a.midpoint_s, est_a.offset_s, config=cfg)
        kal_b = kalman_skew(est_b.midpoint_s, est_b.offset_s, config=cfg)

        bytes_equal = a.timestamp_bytes() == b.timestamp_bytes()
        offset_max = float(np.max(np.abs(est_a.offset_s - est_b.offset_s)))
        delay_max = float(np.max(np.abs(est_a.delay_s - est_b.delay_s)))
        params_max = float(np.max(np.abs(fit_a.params - fit_b.params)))
        kf_offset = abs(kal_a.final_offset_s - kal_b.final_offset_s)
        kf_skew = abs(kal_a.final_skew_s_per_s - kal_b.final_skew_s_per_s)
        counts_equal = (a.n, a.n_lost) == (b.n, b.n_lost)
        ok = bytes_equal and counts_equal and offset_max == 0.0 and delay_max == 0.0

        out.rule(f"case: {label}")
        out(f"  simulated backend     : {a.backend_kind}/{a.backend_driver or 'n/a'}  "
            f"is_hardware={a.is_hardware}")
        out(f"  device backend        : {b.backend_kind}/{b.backend_driver}  "
            f"is_hardware={b.is_hardware}")
        out(f"  completed / lost      : {a.n} / {a.n_lost}  vs  {b.n} / {b.n_lost}")
        seq_equal = a.seq.tolist() == b.seq.tolist()
        out(f"  sequence numbers      : {'identical' if seq_equal else 'DIFFER'}")
        out(
            f"  timestamp bytes       : {'identical' if bytes_equal else 'DIFFER'} "
            f"({len(a.timestamp_bytes())} bytes)"
        )
        out(f"  offset max |diff|     : {offset_max:.3e} s")
        out(f"  delay  max |diff|     : {delay_max:.3e} s")
        out(f"  ls params max |diff|  : {params_max:.3e}")
        out(f"  kf offset |diff|      : {kf_offset:.3e} s")
        out(f"  kf skew   |diff|      : {kf_skew:.3e} s/s")
        out(f"  verdict               : {'PASS' if ok else 'FAIL'}")
        if not ok:
            failures += 1

    out.rule("falsifying control -- two different seeds must differ")
    spec_a = CASES[1][1]
    spec_b = SimulationSpec(
        clock=spec_a.clock,
        link=spec_a.link,
        period_s=spec_a.period_s,
        timestamp_noise_s=spec_a.timestamp_noise_s,
        resolution_ns=spec_a.resolution_ns,
        seed=SEED + 1,
    )
    sim, _ = make_backend_pair(spec_a)
    _, dev = make_backend_pair(spec_b)
    sim.open()
    dev.open()
    a = acquire(sim, scaled(200, minimum=20), min_exchanges=8).exchanges
    b = acquire(dev, scaled(200, minimum=20), min_exchanges=8).exchanges
    sim.close()
    dev.close()
    differ = a.timestamp_bytes() != b.timestamp_bytes()
    out(f"seed {SEED} vs seed {SEED + 1}")
    out(f"  timestamp bytes differ : {differ}")
    control_max = float(
        np.max(np.abs(two_way_estimate(a).offset_s - two_way_estimate(b).offset_s))
    )
    out(f"  offset max |diff|      : {control_max:.3e} s")
    out(f"  verdict                : {'PASS' if differ else 'FAIL'}")
    if not differ:
        failures += 1

    out.rule("what this does and does not establish")
    out("Establishes: the HAL path, acquisition, both baselines and the accounting")
    out("behave identically whichever backend they are given, byte for byte on the")
    out("timestamps and exactly on the estimates.")
    out("Does NOT establish: anything about a board. The device driver is a loopback")
    out("stub, both backends report is_hardware=False, and the Level 4 benchmark rows")
    out("stay empty. This product is labelled Level 3, hardware-pending.")

    out.rule("summary")
    out(f"hard failures : {failures}")
    out(f"saved to      : {out.save()}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
