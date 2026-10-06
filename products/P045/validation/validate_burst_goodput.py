"""Validation: how much the independent-error assumption misstates goodput.

This is the central measurement of the repository.  For each protocol, a
correlated (Gilbert-Elliott) frame-error channel and a memoryless channel are
constructed with the *same marginal frame error rate*, and three numbers are
compared:

    closed form   the textbook expression evaluated at that marginal rate,
    iid measured  the state machine simulated on the memoryless channel,
    GE measured   the state machine simulated on the correlated channel.

``closed form`` versus ``iid measured`` is the agreement already established in
``validate_closed_form.py``; it is reprinted here as a control.  The number the
repository exists for is ``closed form`` versus ``GE measured``: the error a
practitioner makes by sizing a long link with the independent-error formula
when the errors arrive in bursts.

What the measurement found, which is not what the author expected:

    go-back-N        bursts *help*, by a lot.  One burst of b consecutive frame
                     errors costs one go-back of N slots; b independent errors
                     spread through the window cost up to b of them.  The
                     independent-error formula therefore *understates* bursty
                     go-back-N goodput, by up to 74 per cent at a mean burst of
                     100 frames.
    selective repeat bursts help here too, at a finite window, for the same
                     reason.  The author expected the opposite -- that
                     head-of-line blocking would make bursts worse -- and the
                     measurement says no: head-of-line blocking costs a fixed
                     pipeline drain per *blocking event*, and correlation
                     reduces the number of events at a fixed marginal rate.
                     With a window large enough never to stall, selective
                     repeat is insensitive to correlation.
    stop-and-wait    bursts are almost irrelevant.  One attempt occupies
                     exactly N slots and the long-run fraction of attempts that
                     succeed is the stationary marginal, so goodput depends on
                     the marginal rate and not on its correlation.  This one
                     was predicted before the measurement, and it is the
                     cheapest check in the file that the simulator is behaving.

The unifying statement, written after the numbers rather than before them: a
protocol whose cost per error event contains a fixed round-trip term is helped
by correlation, because correlation bundles a fixed amount of error mass into
fewer events.  A protocol whose cost is linear in the marginal error rate is
indifferent to it.  Nothing here is helped by assuming independence.

Runtime: about 150 s on one core.
"""

from __future__ import annotations

import math
import os
import sys

import numpy as np

sys.path.insert(
    0, os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, "src")
)

from arqlonghaul import closedform as cf  # noqa: E402
from arqlonghaul.channel import GilbertElliottChannel, IndependentFrameChannel  # noqa: E402
from arqlonghaul.protocols import (  # noqa: E402
    simulate_go_back_n,
    simulate_selective_repeat,
    simulate_stop_and_wait,
)

SLOTS = 800_000
BURSTS = (2.0, 5.0, 10.0, 25.0, 50.0, 100.0)
FER = 0.05
N_GRID = (20, 60)


def _measure(errors: np.ndarray, n: int) -> dict[str, float]:
    """Goodput of the four protocol configurations on one error realisation."""
    w_min = math.ceil(n)
    w_big = 40 * n
    return {
        "sw": simulate_stop_and_wait(errors, n).goodput,
        "gbn": simulate_go_back_n(errors, n, w_min).goodput,
        "sr_wn": simulate_selective_repeat(errors, n, w_min).goodput,
        "sr_big": simulate_selective_repeat(errors, n, w_big).goodput,
        "realised_fer": float(errors.mean()),
    }


def main() -> None:
    rng = np.random.default_rng(45_045_2)
    print(f"slots per run: {SLOTS}   marginal frame error rate held at {FER}")
    print("channels differ only in correlation; the marginal rate is identical")
    print()
    for n in N_GRID:
        w_min = math.ceil(n)
        w_big = 40 * n
        closed = {
            "sw": cf.sw_throughput(FER, n),
            "gbn": cf.gbn_throughput(FER, n, w_min),
            "sr_wn": cf.sr_throughput(FER, n, w_min),
            "sr_big": cf.sr_throughput(FER, n, w_big),
        }
        iid = _measure(IndependentFrameChannel(FER).errors(SLOTS, rng), n)
        print("=" * 92)
        print(f"N = {n} slots   window(min) = {w_min}   window(big) = {w_big}")
        print("=" * 92)
        print("control: memoryless channel")
        print(f"{'protocol':<28}{'closed form':>13}{'iid measured':>14}"
              f"{'closed/iid - 1 %':>18}")
        for key, label in (
            ("sw", "stop-and-wait"),
            ("gbn", f"go-back-N W={w_min}"),
            ("sr_wn", f"selective repeat W={w_min}"),
            ("sr_big", f"selective repeat W={w_big}"),
        ):
            print(
                f"{label:<28}{closed[key]:>13.6f}{iid[key]:>14.6f}"
                f"{100 * (closed[key] / iid[key] - 1):>18.2f}"
            )
        print(f"  realised marginal frame error rate: {iid['realised_fer']:.6f}")
        print()
        print("correlated channel, same marginal rate:")
        print(f"{'burst':>7}{'protocol':<28}{'closed form':>13}{'GE measured':>13}"
              f"{'closed/GE - 1 %':>17}{'GE/iid - 1 %':>14}")
        for burst in BURSTS:
            ch = GilbertElliottChannel.from_mean_and_burst(FER, burst)
            ge = _measure(ch.errors(SLOTS, rng), n)
            for key, label in (
                ("sw", "stop-and-wait"),
                ("gbn", f"go-back-N W={w_min}"),
                ("sr_wn", f"selective repeat W={w_min}"),
                ("sr_big", f"selective repeat W={w_big}"),
            ):
                print(
                    f"{burst:>7g}{label:<28}{closed[key]:>13.6f}{ge[key]:>13.6f}"
                    f"{100 * (closed[key] / ge[key] - 1):>17.2f}"
                    f"{100 * (ge[key] / iid[key] - 1):>14.2f}"
                )
            print(f"       realised marginal: {ge['realised_fer']:.6f}")
            print("-" * 92)
        print()

    print("=" * 92)
    print("SUMMARY: the misstatement of the independent-error closed form on a")
    print("bursty channel, as a percentage of the measured bursty goodput")
    print("=" * 92)
    print(f"marginal frame error rate {FER}, N = 60 slots, window = 60 frames")
    print(f"{'burst':>7}{'stop-and-wait':>16}{'go-back-N':>13}"
          f"{'sel. repeat W=N':>18}{'sel. repeat W=40N':>20}")
    n = 60
    w_min = n
    closed = {
        "sw": cf.sw_throughput(FER, n),
        "gbn": cf.gbn_throughput(FER, n, w_min),
        "sr_wn": cf.sr_throughput(FER, n, w_min),
        "sr_big": cf.sr_throughput(FER, n, 40 * n),
    }
    rows: list[tuple[float, dict[str, float]]] = []
    for burst in (1.0, *BURSTS):
        if burst <= 1.0:
            errors = IndependentFrameChannel(FER).errors(SLOTS, rng)
        else:
            errors = GilbertElliottChannel.from_mean_and_burst(FER, burst).errors(
                SLOTS, rng
            )
        m = _measure(errors, n)
        rows.append((burst, m))
        print(
            f"{burst:>7g}"
            f"{100 * (closed['sw'] / m['sw'] - 1):>16.2f}"
            f"{100 * (closed['gbn'] / m['gbn'] - 1):>13.2f}"
            f"{100 * (closed['sr_wn'] / m['sr_wn'] - 1):>18.2f}"
            f"{100 * (closed['sr_big'] / m['sr_big'] - 1):>20.2f}"
        )
    print()
    print("burst = 1 is the memoryless channel and is the control row.")
    print("A positive entry means the closed form predicts MORE goodput than the")
    print("protocol achieves; a negative entry means it predicts less.")
    print()
    worst = {}
    for key in ("sw", "gbn", "sr_wn", "sr_big"):
        vals = [(b, 100 * (closed[key] / m[key] - 1)) for b, m in rows if b > 1.0]
        best = max(vals, key=lambda x: abs(x[1]))
        worst[key] = best
        print(
            f"{key:<8} largest misstatement {best[1]:+8.2f} % at burst = "
            f"{best[0]:g} slots"
        )
    print()
    print("The same comparison against the MEASURED memoryless goodput, which")
    print("isolates the effect of correlation from the error in the formula:")
    print(f"{'burst':>7}{'stop-and-wait':>16}{'go-back-N':>13}"
          f"{'sel. repeat W=N':>18}{'sel. repeat W=40N':>20}")
    base = rows[0][1]
    for burst, m in rows:
        print(
            f"{burst:>7g}"
            f"{100 * (m['sw'] / base['sw'] - 1):>16.2f}"
            f"{100 * (m['gbn'] / base['gbn'] - 1):>13.2f}"
            f"{100 * (m['sr_wn'] / base['sr_wn'] - 1):>18.2f}"
            f"{100 * (m['sr_big'] / base['sr_big'] - 1):>20.2f}"
        )
    print("(per cent change in measured goodput relative to the memoryless")
    print(" channel at the same marginal frame error rate)")
    print()
    print("Headline, stated as a practitioner would need it:")
    print("  at a 5 per cent frame error rate on an N = 60 link, the")
    print(f"  independent-error go-back-N formula is wrong by "
          f"{worst['gbn'][1]:+.1f} per cent")
    print(f"  once errors arrive in bursts of {worst['gbn'][0]:g} frames, and the")
    print("  selective-repeat formula at the textbook window is wrong by")
    print(f"  {worst['sr_wn'][1]:+.1f} per cent at bursts of {worst['sr_wn'][0]:g} frames.")
    print(f"  Stop-and-wait is wrong by at most {worst['sw'][1]:+.2f} per cent,")
    print("  as predicted, because its throughput is linear in the marginal rate.")


if __name__ == "__main__":
    main()
