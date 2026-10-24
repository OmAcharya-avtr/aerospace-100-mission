"""Check 1 -- the closed-form asymmetry bias, reproduced against a declared asymmetry.

Requirement (BATCH_08_SPEC, P079): "the standard two-way offset estimator
assuming symmetric delay -- exact under its assumption, and the
asymmetry-induced bias is derived in closed form and verified".

What is checked
---------------
1. **Exactness under symmetry.** With ``d_ab == d_ba`` constructed exactly in
   integer nanoseconds, the estimator returns ``theta``. The claim is agreement
   to within the single float division the implementation performs, so the
   tolerance is stated and tight rather than absent.
2. **The bias equals ``A/2``.** Over a declared static asymmetry sweep, the
   mean estimation error against simulation truth is compared with
   ``asymmetry/2``, and the slope of error against asymmetry is fitted.
   Equation (5) says the slope is exactly one half.
3. **The bias does not depend on the round trip, the turnaround or the
   offset.** Three sweeps at a fixed asymmetry, varying each of the three, with
   the bias required to stay put.
4. **Averaging does not remove it.** The mean error is reported at 20, 200 and
   2000 exchanges: the statistical term falls as the square root of the count
   and the bias does not move, which is the identifiability result made
   numerical.

Run: ``python validation/validate_symmetric_bias.py``
"""

from __future__ import annotations

import sys

import numpy as np

from _bootstrap import Tee, bootstrap, environment_lines, scaled

ROOT = bootstrap()

from timetransfer.acquire import acquire  # noqa: E402
from timetransfer.backends import SimulatedBackend, SimulationSpec  # noqa: E402
from timetransfer.clockmodel import ClockSpec  # noqa: E402
from timetransfer.exchange import ExchangeSet  # noqa: E402
from timetransfer.linkmodel import LinkSpec  # noqa: E402
from timetransfer.twoway import asymmetry_bias_s, symmetric_offsets, two_way_estimate  # noqa: E402

ASYMMETRIES_S = (-1.6e-3, -8.0e-4, -2.0e-4, 0.0, 2.0e-4, 8.0e-4, 1.6e-3)
SEED = 20261024
N = scaled(400, minimum=20)


def _spec(
    asymmetry_s: float,
    *,
    floor_s: float = 5.0e-3,
    turnaround_s: float = 1.0e-4,
    offset0_s: float = 1.0e-3,
    jitter_scale_s: float = 0.0,
    seed: int = SEED,
) -> SimulationSpec:
    return SimulationSpec(
        clock=ClockSpec(offset0_s=offset0_s, skew_s_per_s=0.0),
        link=LinkSpec(
            floor_ab_s=floor_s + 0.5 * asymmetry_s,
            floor_ba_s=floor_s - 0.5 * asymmetry_s,
            jitter_scale_ab_s=jitter_scale_s,
            jitter_scale_ba_s=jitter_scale_s,
            turnaround_s=turnaround_s,
        ),
        period_s=1.0,
        timestamp_noise_s=0.0,
        resolution_ns=1,
        seed=seed,
    )


def _mean_error(spec: SimulationSpec, n: int = N) -> float:
    backend = SimulatedBackend(spec)
    backend.open()
    exchanges = acquire(backend, n, min_exchanges=8).exchanges
    backend.close()
    estimate = two_way_estimate(exchanges)
    truth = exchanges.true_offset_s
    assert truth is not None
    return float(np.mean(estimate.offset_s - truth))


def main() -> int:
    out = Tee("validate_symmetric_bias_output.txt")
    out("timetransfer validation 1 -- symmetric-delay estimator and its closed-form bias")
    out("=" * 78)
    for line in environment_lines():
        out(line)
    failures = 0

    out.rule("1a. exactness under symmetry, constructed in integer nanoseconds")
    out(f"{'theta [ns]':>14} {'d [ns]':>12} {'turn [ns]':>12} {'estimate [s]':>18} "
        f"{'error [s]':>14}")
    worst_exact = 0.0
    for theta_ns in (-1_000_000_000, -1_000_000, 0, 1_000_000, 999_999_999):
        for d_ns in (1, 1_000, 5_000_000, 250_000_000):
            for turn_ns in (0, 1, 100_000):
                t1 = 0
                t2 = t1 + d_ns + theta_ns
                t3 = t2 + turn_ns
                t4 = t1 + d_ns + turn_ns + d_ns
                one = ExchangeSet(
                    t1_ns=np.asarray([t1], dtype=np.int64),
                    t2_ns=np.asarray([t2], dtype=np.int64),
                    t3_ns=np.asarray([t3], dtype=np.int64),
                    t4_ns=np.asarray([t4], dtype=np.int64),
                    seq=np.asarray([0], dtype=np.int64),
                )
                offset, _ = symmetric_offsets(one)
                error = float(offset[0] - theta_ns * 1e-9)
                worst_exact = max(worst_exact, abs(error))
                if turn_ns == 100_000 and d_ns in (1, 250_000_000):
                    out(
                        f"{theta_ns:>14d} {d_ns:>12d} {turn_ns:>12d} "
                        f"{float(offset[0]):>18.12e} {error:>14.3e}"
                    )
    out("")
    out(f"worst |error| over 60 constructed exchanges : {worst_exact:.3e} s")
    out("tolerance (one float division of an int64 ns): 1.000e-15 s")
    if worst_exact > 1.0e-15:
        failures += 1
        out("FAIL: the estimator is not exact under symmetry to within rounding")
    else:
        out("PASS")

    out.rule("1b. bias against a declared static asymmetry")
    out(f"{'asymmetry [s]':>16} {'A/2 closed form [s]':>22} {'measured error [s]':>22} "
        f"{'difference [s]':>16}")
    measured = []
    worst_bias = 0.0
    for asym in ASYMMETRIES_S:
        error = _mean_error(_spec(asym))
        closed = float(asymmetry_bias_s(asym))
        measured.append(error)
        worst_bias = max(worst_bias, abs(error - closed))
        out(f"{asym:>16.6e} {closed:>22.12e} {error:>22.12e} {error - closed:>16.3e}")
    out("")
    out(f"worst |measured - A/2| : {worst_bias:.3e} s")
    out("tolerance              : 2.000e-09 s (one timestamp quantisation step is 1 ns,")
    out("                         and the estimator differences four of them)")
    if worst_bias > 2.0e-9:
        failures += 1
        out("FAIL: the measured bias does not match the closed form")
    else:
        out("PASS")

    out.rule("1c. fitted slope of error against asymmetry")
    coefficients = np.polyfit(np.asarray(ASYMMETRIES_S), np.asarray(measured), 1)
    slope = float(coefficients[0])
    intercept = float(coefficients[1])
    out(f"fitted slope     : {slope:.12f}")
    out("closed form      : 0.500000000000  (equation (5), exactly one half)")
    out(f"slope error      : {slope - 0.5:.3e}")
    out(f"fitted intercept : {intercept:.3e} s")
    if abs(slope - 0.5) > 1.0e-6:
        failures += 1
        out("FAIL: the slope is not one half")
    else:
        out("PASS")

    out.rule("1d. the bias is independent of delay, turnaround and offset")
    asym = 8.0e-4
    closed = float(asymmetry_bias_s(asym))
    out(f"declared asymmetry {asym:.6e} s -> closed-form bias {closed:.9e} s")
    out("")
    out(f"{'swept quantity':>22} {'value':>14} {'measured bias [s]':>22} "
        f"{'difference [s]':>16}")
    worst_independence = 0.0
    for floor_s in (1.0e-3, 5.0e-3, 5.0e-2):
        error = _mean_error(_spec(asym, floor_s=floor_s))
        worst_independence = max(worst_independence, abs(error - closed))
        out(f"{'one-way floor [s]':>22} {floor_s:>14.3e} {error:>22.12e} "
            f"{error - closed:>16.3e}")
    for turnaround_s in (0.0, 1.0e-4, 1.0e-1):
        error = _mean_error(_spec(asym, turnaround_s=turnaround_s))
        worst_independence = max(worst_independence, abs(error - closed))
        out(f"{'turnaround [s]':>22} {turnaround_s:>14.3e} {error:>22.12e} "
            f"{error - closed:>16.3e}")
    for offset0_s in (-1.0e-1, 0.0, 1.0e-3, 1.0e-1):
        error = _mean_error(_spec(asym, offset0_s=offset0_s))
        worst_independence = max(worst_independence, abs(error - closed))
        out(f"{'true offset [s]':>22} {offset0_s:>14.3e} {error:>22.12e} "
            f"{error - closed:>16.3e}")
    out("")
    out(f"worst departure : {worst_independence:.3e} s")
    if worst_independence > 2.0e-9:
        failures += 1
        out("FAIL: the bias depends on something equation (5) says it does not")
    else:
        out("PASS")

    out.rule("1e. averaging reduces the statistical term and not the bias")
    out("jitter is switched on here, so there is a statistical term to reduce.")
    out("")
    out(f"{'n exchanges':>14} {'mean error [s]':>20} {'minus A/2 [s]':>18} "
        f"{'sd of per-exchange error [s]':>30}")
    bias_by_n = []
    for n in (20, scaled(200, minimum=20), scaled(2000, minimum=30)):
        spec = _spec(asym, jitter_scale_s=2.0e-4)
        backend = SimulatedBackend(spec)
        backend.open()
        exchanges = acquire(backend, n, min_exchanges=8).exchanges
        backend.close()
        estimate = two_way_estimate(exchanges)
        truth = exchanges.true_offset_s
        assert truth is not None
        err = estimate.offset_s - truth
        bias_by_n.append(float(np.mean(err)))
        out(
            f"{n:>14d} {float(np.mean(err)):>20.12e} {float(np.mean(err)) - closed:>18.3e} "
            f"{float(np.std(err, ddof=1)):>30.6e}"
        )
    out("")
    out(f"spread of the mean error across n : {float(np.ptp(np.asarray(bias_by_n))):.3e} s")
    out("The mean error does not approach zero as n grows. That is the point: the")
    out("static asymmetry enters the observable difference identically to the offset,")
    out("so no amount of data separates them.")

    out.rule("summary")
    out(f"hard failures : {failures}")
    out(f"saved to      : {out.save()}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
