"""Uncertainty analysis (Level 3): propagation, discretisation, parameters.

Four error sources are quantified separately.  None of them is combined into a
single figure, because they are not commensurable and a single number would
hide which one dominates.

1. Propagation-input uncertainty -- a stated along-track position error is
   converted to a mean-anomaly perturbation and the contact-window edges are
   recomputed.  The along-track error is an INPUT; this package asserts no
   TLE error magnitude of its own.
2. Discretisation -- the window set of one ISL is recomputed at several
   contact-scan grid steps, exposing both edge-refinement error (bounded by
   the bisection tolerance, independent of step) and missed short windows
   (a function of step).
3. Bisection-tolerance error -- the same window recomputed at several
   refinement tolerances, which bounds the edge error against the propagator.
4. Link-budget parameter uncertainty -- Monte Carlo over terminal parameters,
   reported as percentiles with the standard error of the mean so the draw
   count can be judged.
"""

from __future__ import annotations

import os
import sys
from datetime import UTC, datetime, timedelta

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from constellink.constellation import Constellation, GroundStation, walker_delta  # noqa: E402
from constellink.contacts import contact_windows_isl  # noqa: E402
from constellink.synthdata import default_optical_terminal, default_rf_terminal  # noqa: E402
from constellink.uncertainty import (  # noqa: E402
    grid_step_convergence,
    monte_carlo_capacity,
    window_edge_sensitivity,
)

EPOCH = datetime(2026, 4, 1, tzinfo=UTC)
STATION = GroundStation("AWARUA", -46.53, 168.38, 0.01, 10.0)


def main() -> int:
    const = walker_delta(24, 4, 1, 53.0, 550.0, EPOCH, max_epoch_age_days=2.0)
    print("Uncertainty analysis")
    print("=" * 88)
    print("")

    print("Part 1 -- along-track propagation error to window-edge shift")
    print("  method: perturb the mean anomaly by dM = ds / a (circular-orbit arc")
    print("          length, exact for a circular orbit; Vallado 2013 Ch. 2) and")
    print("          recompute the ground windows. 6 h horizon, 30 s scan.")
    print("  the along-track error is a user input, not a result of this package.")
    head = (f"  {'ds [km]':>10}{'dM [rad]':>13}{'n win':>7}{'max |dt| [s]':>15}"
            f"{'mean dt_open [s]':>19}{'mean d_dur [s]':>17}")
    print(head)
    print("  " + "-" * (len(head) - 2))
    sens = []
    for ds in (0.1, 1.0, 5.0, 20.0):
        es = window_edge_sensitivity(const, "W01-04", STATION, EPOCH,
                                      EPOCH + timedelta(hours=6), 30.0, ds)
        sens.append((ds, es))
        print(f"  {ds:>10.2f}{es.delta_m_rad:>13.3e}{es.n_windows_nominal:>7d}"
              f"{es.max_abs_edge_shift_s:>15.4f}"
              f"{es.d_t_open_s.mean():>19.4f}{es.d_duration_s.mean():>17.4f}")
    slopes = [es.max_abs_edge_shift_s / ds for ds, es in sens]
    print(f"  edge shift per km of along-track error: "
          f"{', '.join(f'{s:.4f}' for s in slopes)} s/km")
    print("  the slope is near the reciprocal of the ground-track speed, which")
    print("  is the expected first-order behaviour and is reported, not asserted.")
    print("  the smallest perturbation sits at the 0.05 s default bisection")
    print("  tolerance floor, which is why its apparent slope is larger; the")
    print("  three larger perturbations are the ones to read.")
    print("")

    print("Part 2 -- contact-scan grid step (discretisation)")
    gc = grid_step_convergence(const, "W00-00", "W02-02", EPOCH,
                               EPOCH + timedelta(hours=6),
                               [5.0, 10.0, 30.0, 60.0, 120.0, 300.0, 600.0])
    print(gc.format_table())
    base_n = int(gc.n_windows[0])
    base_t = float(gc.total_duration_s[0])
    print(f"  reference (5 s step): {base_n} windows, {base_t:.3f} s total")
    for step, n, tot in zip(gc.step_s, gc.n_windows, gc.total_duration_s,
                            strict=True):
        print(f"    step {step:>6.1f} s: window count delta {int(n) - base_n:+d}, "
              f"total duration delta {tot - base_t:+.3f} s")
    print("  windows shorter than the step can be missed entirely; this table is")
    print("  how a user chooses a step for their own shortest window of interest.")
    print("")

    print("Part 3 -- bisection refinement tolerance")
    eph = Constellation([const.satellite("W00-00"),
                         const.satellite("W02-02")]).ephemeris(
        EPOCH, EPOCH + timedelta(hours=6), 30.0)
    a, b = const.satellite("W00-00"), const.satellite("W02-02")
    ref = contact_windows_isl(eph, a, b, refine_tol_s=1e-6)
    head = f"  {'tol [s]':>10}{'n win':>7}{'max |dt_open| [s]':>20}{'max |dt_close| [s]':>21}"
    print(head)
    print("  " + "-" * (len(head) - 2))
    for tol in (1e-6, 1e-3, 0.05, 1.0, 10.0):
        ws = contact_windows_isl(eph, a, b, refine_tol_s=tol)
        if len(ws) != len(ref):
            print(f"  {tol:>10.0e}{len(ws):>7d}{'window count differs':>20}")
            continue
        d_open = max(abs((w.t_open - r.t_open).total_seconds())
                     for w, r in zip(ws, ref, strict=True))
        d_close = max(abs((w.t_close - r.t_close).total_seconds())
                      for w, r in zip(ws, ref, strict=True))
        print(f"  {tol:>10.0e}{len(ws):>7d}{d_open:>20.6f}{d_close:>21.6f}")
    print("  edge error is bounded by the tolerance and is independent of the")
    print("  grid step, because bisection re-propagates rather than interpolating.")
    print("")

    print("Part 4 -- link-budget parameter uncertainty (Monte Carlo)")
    rf_sigmas = {"tx_power_dbw": 0.3, "tx_gain_dbi": 0.5,
                 "rx_g_over_t_db_per_k": 0.7, "other_loss_db": 0.3}
    print("  RF terminal, 1500 km, 1-sigma Gaussian inputs (all stated inputs):")
    for k, v in rf_sigmas.items():
        print(f"    {k:<26} {v:g}")
    dist = monte_carlo_capacity(1500.0, default_rf_terminal(), rf_sigmas,
                                n_draws=4000, seed=11)
    print("  " + dist.format_summary().replace("\n", "\n  "))
    spread_db = 10.0 * np.log10(dist.percentiles[95.0] / dist.percentiles[5.0])
    print(f"  p95/p5 spread: {spread_db:.3f} dB "
          f"(quadrature sum of the input sigmas is "
          f"{2 * 1.6449 * np.sqrt(sum(v ** 2 for v in rf_sigmas.values())):.3f} dB)")
    print("")
    opt_sigmas = {"tx_power_w": 0.05, "beam_divergence_full_rad": 2e-6,
                  "pointing_error_rad": 1e-6, "rx_optics_loss_db": 0.3}
    print("  Optical terminal, 1500 km, 1-sigma Gaussian inputs:")
    for k, v in opt_sigmas.items():
        print(f"    {k:<26} {v:g}")
    dist_o = monte_carlo_capacity(1500.0, default_optical_terminal(), opt_sigmas,
                                  n_draws=4000, seed=12)
    print("  " + dist_o.format_summary().replace("\n", "\n  "))
    print("  input parameters are drawn independently; real terminal parameters")
    print("  are correlated, so this spread is not a calibrated predictive")
    print("  interval. It is a sensitivity measurement.")
    print("")

    print("=" * 88)
    print("This script reports measurements; it has no pass/fail criterion.")
    print("OVERALL: REPORTED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
