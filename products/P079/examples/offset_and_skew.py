"""Offset, skew and the measured uncertainty over one acquisition.

Writes ``screenshots/offset_and_skew.png``. What to notice: the per-exchange
offsets scatter by roughly the queueing jitter, both baselines track the same
line through them, and the interval that includes the declared asymmetry bound
is far wider than the one that does not -- and it is the wide one that contains
the truth.
"""

from __future__ import annotations

import numpy as np
from _common import SERIES, bootstrap, save, style

ROOT = bootstrap()
style()

import matplotlib.pyplot as plt  # noqa: E402

from timetransfer import (  # noqa: E402
    AsymmetrySource,
    ClockSpec,
    KalmanConfig,
    LinkSpec,
    SimulatedBackend,
    SimulationSpec,
    acquire,
    fit_least_squares,
    kalman_skew,
    offset_uncertainty,
    two_way_estimate,
)


def main() -> None:
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
    report = acquire(backend, 600, min_exchanges=8)
    backend.close()
    exchanges = report.exchanges
    estimate = two_way_estimate(exchanges, assumed_skew_s_per_s=1.8e-6)
    fit = fit_least_squares(estimate.midpoint_s, estimate.offset_s, order=2)
    kalman = kalman_skew(
        estimate.midpoint_s,
        estimate.offset_s,
        config=KalmanConfig(q1_s2_per_s=1.0e-20, q2_per_s=1.0e-26),
    )
    bounded = offset_uncertainty(
        fit,
        resolution_ns=1,
        asymmetry_source=AsymmetrySource.DECLARED_BOUND,
        asymmetry_bound_s=1.600e-3,
    )
    plain = offset_uncertainty(fit, resolution_ns=1)
    truth = exchanges.true_offset_s
    assert truth is not None

    t = estimate.midpoint_s
    fitted = np.asarray([fit.offset_at(float(v))[0] for v in t])

    fig, axes = plt.subplots(2, 1, figsize=(8.4, 6.2), height_ratios=(2.1, 1.0))
    top, bottom = axes

    top.plot(t, estimate.offset_s * 1e3, ".", color=SERIES[0], ms=2.4, alpha=0.55,
             label="per-exchange symmetric offset (baseline 1)")
    top.plot(t, truth * 1e3, "-", color=SERIES[2], lw=1.6, label="simulation truth")
    top.plot(t, fitted * 1e3, "-", color=SERIES[1], lw=1.4,
             label="least squares, order 2 (baseline 2a)")
    top.plot(kalman.t_s, kalman.states[:, 0] * 1e3, "--", color=SERIES[3], lw=1.2,
             label="Kalman, two states (baseline 2b)")
    for label, unc, colour, style_ in (
        ("k=1.96 with the declared asymmetry bound", bounded, SERIES[1], "-"),
        ("k=1.96 with no asymmetry term", plain, SERIES[4], "--"),
    ):
        lower, upper = unc.interval_s()
        top.vlines(
            unc.epoch_s, lower * 1e3, upper * 1e3, color=colour, lw=2.6, ls=style_,
            label=label,
        )
    top.set_ylabel("remote clock offset [ms]")
    top.set_title(
        f"P079 timetransfer: offset over {exchanges.n} exchanges, "
        f"{exchanges.n_lost} lost; declared asymmetry "
        f"{spec.link.static_asymmetry_s * 1e6:+.0f} us"
    )
    top.legend(loc="upper left", fontsize=7.4, ncols=2)

    residual = fit.detrended(t, estimate.offset_s)
    bottom.plot(t, residual * 1e6, "-", color=SERIES[0], lw=0.7)
    bottom.axhline(0.0, color=SERIES[1], lw=1.0)
    bottom.set_xlabel("local clock time [s]")
    bottom.set_ylabel("fit residual [us]")
    bottom.set_title(
        f"residual about the order-2 fit, sigma = {fit.sigma_s * 1e6:.1f} us; "
        f"u_combined = {bounded.u_combined_s * 1e6:.1f} us against "
        f"{plain.u_combined_s * 1e6:.1f} us without the asymmetry term"
    )

    fig.tight_layout()
    print(save(fig, "offset_and_skew"))
    print(f"exchanges              : {exchanges.n} completed, {exchanges.n_lost} lost")
    print(f"fitted skew            : {fit.skew_s_per_s:.6e} s/s "
          f"(declared {spec.clock.skew_s_per_s:.6e})")
    print(f"u combined, bounded    : {bounded.u_combined_s:.6e} s")
    print(f"u combined, unaccounted: {plain.u_combined_s:.6e} s")
    true_at_epoch = float(np.interp(bounded.epoch_s, t, truth))
    print(f"estimate minus truth   : {bounded.offset_s - true_at_epoch:+.6e} s")
    print(f"inside the wide band   : "
          f"{abs(bounded.offset_s - true_at_epoch) <= bounded.expanded_s}")
    print(f"inside the narrow band : "
          f"{abs(bounded.offset_s - true_at_epoch) <= plain.expanded_s}")


if __name__ == "__main__":
    main()
