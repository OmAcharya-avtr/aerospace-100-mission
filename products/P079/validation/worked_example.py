"""The worked example quoted in README.md, run so the output cannot drift.

This is the example in the README's "A worked example" section, executed here
so that the text under it is what the code actually printed. Every command,
number and output fragment quoted in README.md, validation/VALIDATION.md and
MODEL_CARD.md is re-executed after the final code change and replaced with what
the run produced.

Run: ``python validation/worked_example.py``
"""

from __future__ import annotations

import sys

import numpy as np

from _bootstrap import Tee, bootstrap, environment_lines, scaled

ROOT = bootstrap()

from timetransfer import (  # noqa: E402
    AsymmetrySource,
    ClockSpec,
    KalmanConfig,
    LinkSpec,
    SimulatedBackend,
    SimulationSpec,
    acquire,
    allan_table,
    fit_least_squares,
    kalman_skew,
    offset_uncertainty,
    two_way_estimate,
)


def main() -> int:
    out = Tee("worked_example_output.txt")
    out("timetransfer -- the worked example from README.md")
    out("=" * 78)
    for line in environment_lines():
        out(line)

    out.rule("the example")
    out("A link whose forward path is 800 us longer than its reverse path, a remote")
    out("clock 1.25 ms fast and running 1.8 ppm slow, 600 exchanges at 1 Hz.")
    out("")

    spec = SimulationSpec(
        clock=ClockSpec(
            offset0_s=1.250e-3,
            skew_s_per_s=1.800e-6,
            drift_per_s=2.000e-12,
            q1_s2_per_s=1.0e-20,
            q2_per_s=1.0e-26,
        ),
        link=LinkSpec(
            floor_ab_s=5.400e-3,
            floor_ba_s=4.600e-3,
            jitter_scale_ab_s=2.400e-4,
            jitter_scale_ba_s=1.800e-4,
            loss_probability=0.03,
        ),
        period_s=1.0,
        timestamp_noise_s=2.000e-8,
        resolution_ns=1,
        seed=20261024,
    )
    backend = SimulatedBackend(spec)
    backend.open()
    report = acquire(backend, scaled(600, minimum=40), min_exchanges=8)
    backend.close()

    estimate = two_way_estimate(report.exchanges, assumed_skew_s_per_s=1.8e-6)
    fit = fit_least_squares(estimate.midpoint_s, estimate.offset_s, order=2)
    kalman = kalman_skew(
        estimate.midpoint_s,
        estimate.offset_s,
        config=KalmanConfig(q1_s2_per_s=1.0e-20, q2_per_s=1.0e-26),
    )
    uncertainty = offset_uncertainty(
        fit,
        resolution_ns=1,
        asymmetry_source=AsymmetrySource.DECLARED_BOUND,
        asymmetry_bound_s=1.600e-3,
    )
    residual = fit.detrended(estimate.midpoint_s, estimate.offset_s)
    stability = allan_table(residual, 1.0, noise_kind="white_fm")

    out("acquisition")
    out(report.as_text())
    out("")
    out("baseline 1 -- symmetric-delay two-way estimator")
    out(estimate.as_text())
    out("")
    out("baseline 2a -- least squares, order 2")
    out(fit.as_text())
    out("")
    out("baseline 2b -- two-state Kalman")
    out(kalman.as_text())
    out("")
    out("the measured offset uncertainty")
    out(uncertainty.as_text())
    lower, upper = uncertainty.interval_s()
    out(f"interval            : [{lower:.9e}, {upper:.9e}] s")
    out("")
    out("stability of the detrended residual")
    out(stability.as_text())

    out.rule("against simulation truth")
    truth = report.exchanges.true_offset_s
    assert truth is not None
    true_at_epoch = float(np.interp(uncertainty.epoch_s, estimate.midpoint_s, truth))
    error = uncertainty.offset_s - true_at_epoch
    out(f"declared static asymmetry  : {spec.link.static_asymmetry_s:+.6e} s")
    out(f"closed-form bias A/2       : {spec.link.static_bias_s:+.6e} s")
    out(f"true offset at the epoch   : {true_at_epoch:+.9e} s")
    out(f"estimate                   : {uncertainty.offset_s:+.9e} s")
    out(f"estimate minus truth       : {error:+.6e} s")
    out(f"expanded uncertainty       : {uncertainty.expanded_s:.6e} s")
    out(f"error inside the interval  : {abs(error) <= uncertainty.expanded_s}")
    out(f"error / closed-form bias   : {error / spec.link.static_bias_s:.4f}")
    out("")
    out("The error is dominated by the declared static asymmetry, which is exactly")
    out("what equation (5) of timetransfer.twoway says it must be: the estimate is")
    out("off by A/2 and no amount of data removes it. The reported uncertainty covers")
    out("it because the asymmetry was declared as a bound; with no asymmetry term the")
    out("same estimate carries a narrower interval, by the ratio printed in the next")
    out("section, and does not cover the error. That comparison is the product.")

    out.rule("the same estimate with no asymmetry term, for contrast")
    plain = offset_uncertainty(fit, resolution_ns=1)
    out(plain.as_text())
    out("")
    out(f"u combined with a declared bound : {uncertainty.u_combined_s:.6e} s")
    out(f"u combined with no asymmetry term: {plain.u_combined_s:.6e} s")
    out(f"ratio                            : "
        f"{uncertainty.u_combined_s / plain.u_combined_s:.4f}")
    out(f"error inside the narrow interval : {abs(error) <= plain.expanded_s}")

    out.rule("summary")
    out(f"saved to : {out.save()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
