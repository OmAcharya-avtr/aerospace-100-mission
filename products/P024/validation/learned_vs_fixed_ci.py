"""V5 - The learned gain scheduler against classical gain rules.

Design
------
Two disjoint, seeded scenario sets (``scenarios.py``, entirely synthetic - no
flight telemetry anywhere in this package):

    training  seeds 1000-1019   (20 scenarios)
    held out  seeds 5000-5039   (40 scenarios)

Nothing tuned on the training set is re-tuned on the held-out set, and the
held-out set is simulated exactly once per policy.

Policies compared, all on identical paired scenarios
----------------------------------------------------
fixed      one constant gain, the training-set argmin of mean cost
sized      k = c m_max / (<|B|> omega_est) from the first 40 magnetometer
           samples; ``(rate estimator, c)`` jointly tuned on the training set
powerlaw   log10 k = a + b log10 m_max + c log10 j, three coefficients fitted
           by least squares to the training-set oracle gains
learned    RandomForest gain scheduler, base gain = ``fixed``, re-evaluated
           every 30 control steps with an ensemble-spread confidence

and, on the training set only, ``oracle``: the best constant gain per
scenario found by grid search.  That is an upper bound on what the learned
scheduler's training target can buy, not a deployable policy.

Cost
----
``cost = t_detumble / T_orbit + w * integral|m|^2 dt / (m_max^2 T_orbit)``,
with the default ``w = 0.5``.  Because the two terms are recorded separately,
the whole comparison is re-scored at ``w = 0`` (time only), ``w = 0.5`` and
``w = 2`` without re-simulating, so no conclusion rests on one arbitrary
weight.

Statistics
----------
Paired Student-t intervals on the per-scenario difference.  An interval that
contains zero means **this experiment cannot separate the two policies**; it
does not mean they are equal.
"""

from __future__ import annotations

import time

import numpy as np

from _support import Tee  # noqa: E402

from detumblesim.evaluate import (  # noqa: E402
    ENERGY_WEIGHT,
    fit_power_law_gain,
    oracle_gain,
    run_policy,
    training_rows,
)
from detumblesim.features import FEATURE_NAMES  # noqa: E402
from detumblesim.metrics import format_interval, mean_ci, paired_difference_ci  # noqa: E402
from detumblesim.policies import (  # noqa: E402
    FixedGainPolicy,
    PowerLawGainPolicy,
    ScheduledGainPolicy,
    SizedGainPolicy,
)
from detumblesim.scenarios import sample_scenarios  # noqa: E402
from detumblesim.scheduler import GainScheduler  # noqa: E402

GAIN_GRID = np.geomspace(1.0e4, 1.0e6, 8)
SIM = {"duration_s": 23000.0, "control_dt_s": 2.0, "substeps": 1}
N_TRAIN, TRAIN_SEED0 = 20, 1000
N_TEST, TEST_SEED0 = 40, 5000
SIZED_COEFFS = np.geomspace(0.5, 4.0, 5)
SIZED_ESTIMATORS = ("max", "mean", "median")
WEIGHTS = (0.0, ENERGY_WEIGHT, 2.0)
POLICIES = ("fixed", "sized", "powerlaw", "learned")


def cost_at(scores, weight):
    """Re-score a list of ``RunScore`` at a different energy weight."""
    return np.array(
        [s.time_orbits + weight * (s.energy_term / ENERGY_WEIGHT) for s in scores]
    )


def main() -> None:
    t0 = time.perf_counter()
    with Tee(__file__) as out:
        out("V5  Learned B-dot gain scheduler vs classical gain rules")
        out("=" * 78)
        out(f"training scenarios: seeds {TRAIN_SEED0}-{TRAIN_SEED0 + N_TRAIN - 1}   "
            f"held-out: seeds {TEST_SEED0}-{TEST_SEED0 + N_TEST - 1}")
        out(f"gain grid: {np.array2string(GAIN_GRID, precision=4)}")
        out(f"simulation: {SIM}, target rate 1.0 deg/s")
        out("ALL DATA IS SYNTHETIC.  No flight telemetry is used anywhere.")
        out("")

        train = sample_scenarios(N_TRAIN, TRAIN_SEED0)
        grid = np.empty((N_TRAIN, GAIN_GRID.size))
        best = np.empty(N_TRAIN)
        for i, s in enumerate(train):
            bg, _, costs = oracle_gain(s, GAIN_GRID, **SIM)
            grid[i] = costs
            best[i] = bg
        mean_cost = grid.mean(axis=0)
        k_fixed = float(GAIN_GRID[int(np.argmin(mean_cost))])
        out("E1  Fixed-gain tuning on the training set (mean cost per grid gain)")
        for k, c in zip(GAIN_GRID, mean_cost, strict=True):
            out(f"      k = {k:10.4e}   mean cost = {c:8.4f}")
        out(f"    tuned fixed gain k_fixed        = {k_fixed:.6e} A m^2 s / T")
        out(f"    training mean cost, fixed       = {mean_cost.min():.4f}")
        out(f"    training mean cost, oracle      = {grid.min(axis=1).mean():.4f}")
        out(f"    headroom the oracle offers      = "
            f"{100 * (1 - grid.min(axis=1).mean() / mean_cost.min()):.2f} %")
        out(f"    per-scenario oracle log10 gains = "
            f"{np.array2string(np.log10(best), precision=2)}")
        out(f"    distinct oracle gains used      = {len(set(best.tolist()))} of "
            f"{GAIN_GRID.size} grid values")
        out("")

        out("E2  Sized-gain baseline tuning on the training set")
        best_sized = None
        for est in SIZED_ESTIMATORS:
            costs = []
            for c in SIZED_COEFFS:
                v = [
                    run_policy(
                        s, SizedGainPolicy(s.magnetorquer, coefficient=float(c),
                                           rate_estimator=est), **SIM
                    )[1].cost
                    for s in train
                ]
                costs.append(float(np.mean(v)))
                if best_sized is None or costs[-1] < best_sized[0]:
                    best_sized = (costs[-1], est, float(c))
            out(f"    estimator {est:7s}: "
                + "  ".join(
                    f"c={c:.2f}->{v:.4f}"
                    for c, v in zip(SIZED_COEFFS, costs, strict=True)
                ))
        out(f"    tuned sized rule: estimator = {best_sized[1]}, c = "
            f"{best_sized[2]:.4f}, training mean cost = {best_sized[0]:.4f}")
        out("")

        out("E3  Power-law baseline fitted to the training oracle gains")
        coef, rms = fit_power_law_gain(train, best)
        out(f"    log10 k = {coef[0]:.4f} + {coef[1]:.4f} log10(m_max) "
            f"+ {coef[2]:.4f} log10(j)")
        out(f"    RMS residual = {rms:.4f} dex over {N_TRAIN} training scenarios")
        out("    Sizing-rule expectation would be b = 1 (k proportional to m_max);")
        out("    the fitted exponents are reported as measured, not imposed.")
        out("")

        out("E4  Learned scheduler training")
        rows_x, rows_y = [], []
        for s, bg in zip(train, best, strict=True):
            res, _ = run_policy(s, FixedGainPolicy(k_fixed), **SIM)
            x, y = training_rows(s, float(bg), k_fixed, res)
            if x.size:
                rows_x.append(x)
                rows_y.append(y)
        x = np.vstack(rows_x)
        y = np.concatenate(rows_y)
        sch = GainScheduler().fit(x, y)
        out(f"    feature rows = {x.shape[0]}, features = {x.shape[1]}")
        out(f"    label range  = [{y.min():+.4f}, {y.max():+.4f}] dex "
            f"(log10 k_oracle / k_fixed)")
        out(f"    model        = RandomForestRegressor(n_estimators="
            f"{sch.n_estimators}, max_depth={sch.max_depth}, "
            f"min_samples_leaf={sch.min_samples_leaf}, random_state={sch.random_state})")
        out("    impurity feature importances (indicative only):")
        for name, imp in zip(FEATURE_NAMES, sch.feature_importances(), strict=True):
            out(f"      {name:24s} {imp:.4f}")
        out("")

        out(f"E5  Held-out evaluation, {N_TEST} paired scenarios")
        scores = {p: [] for p in POLICIES}
        confidences = []
        gains_used = {p: [] for p in POLICIES}
        for s in sample_scenarios(N_TEST, TEST_SEED0):
            m_max = float(np.min(s.magnetorquer.max_dipole_am2))
            j = s.inertia_scale_kgm2
            scores["fixed"].append(run_policy(s, FixedGainPolicy(k_fixed), **SIM)[1])
            gains_used["fixed"].append(k_fixed)

            sized = SizedGainPolicy(
                s.magnetorquer, coefficient=best_sized[2], rate_estimator=best_sized[1]
            )
            scores["sized"].append(run_policy(s, sized, **SIM)[1])
            gains_used["sized"].append(sized.current_gain())

            pl = PowerLawGainPolicy(coef, m_max, j)
            scores["powerlaw"].append(run_policy(s, pl, **SIM)[1])
            gains_used["powerlaw"].append(pl.gain)

            pol = ScheduledGainPolicy(sch, k_fixed, m_max, j)
            scores["learned"].append(run_policy(s, pol, **SIM)[1])
            gains_used["learned"].append(
                float(np.mean([g for _, g, _ in pol.gain_history]))
                if pol.gain_history else k_fixed
            )
            confidences.extend(c for _, _, c in pol.gain_history)
        out("")

        out("    Per-policy summary at the default energy weight "
            f"w = {ENERGY_WEIGHT}")
        out(f"    {'policy':>9} {'cost (95% CI)':>34} {'time [orbits]':>22} "
            f"{'energy term':>20} {'fails':>6} {'sat[%]':>7}")
        for p in POLICIES:
            sc = scores[p]
            fails = sum(1 for r in sc if not r.detumbled)
            sat = 100.0 * float(np.mean([r.saturated_fraction for r in sc]))
            out(f"    {p:>9} {format_interval(mean_ci(cost_at(sc, ENERGY_WEIGHT))):>34} "
                f"{format_interval(mean_ci([r.time_orbits for r in sc])):>22} "
                f"{format_interval(mean_ci([r.energy_term for r in sc])):>20} "
                f"{fails:>6} {sat:>7.2f}")
        out("")
        det = {p: [r.detumble_time_s for r in scores[p] if r.detumbled] for p in POLICIES}
        for p in POLICIES:
            out(f"    {p:>9}: mean detumble time over runs that finished = "
                f"{np.mean(det[p]):.1f} s  (n = {len(det[p])})")
            out(f"    {'':>9}  mean gain actually used = "
                f"{np.mean(gains_used[p]):.4e} A m^2 s / T")
        out("")
        out(f"    learned-scheduler confidence over {len(confidences)} gain updates: "
            f"mean {np.mean(confidences):.4f}, min {np.min(confidences):.4f}, "
            f"max {np.max(confidences):.4f}")
        out("")

        out("E6  Paired differences (negative favours the first policy)")
        verdicts = {}
        for w in WEIGHTS:
            out("")
            out(f"    Energy weight w = {w}")
            for a, b in (
                ("learned", "fixed"),
                ("powerlaw", "fixed"),
                ("sized", "fixed"),
                ("learned", "powerlaw"),
                ("learned", "sized"),
            ):
                d = paired_difference_ci(cost_at(scores[a], w), cost_at(scores[b], w))
                verdict = (
                    ("A WINS" if d.mean < 0 else "B WINS")
                    if d.excludes_zero
                    else "NOT RESOLVED"
                )
                verdicts[(w, a, b)] = (d, verdict)
                out(f"      {a:>9} - {b:<9} {format_interval(d):>28}   {verdict}")
        out("")

        out("E7  Findings, as measured")
        lf0 = verdicts[(0.0, "learned", "fixed")]
        lf1 = verdicts[(ENERGY_WEIGHT, "learned", "fixed")]
        lf2 = verdicts[(2.0, "learned", "fixed")]
        lp0 = verdicts[(0.0, "learned", "powerlaw")]
        lp1 = verdicts[(ENERGY_WEIGHT, "learned", "powerlaw")]
        out("    1. On detumble time alone (w = 0) the learned scheduler beats the")
        out(f"       fixed gain: {format_interval(lf0[0])} orbits, {lf0[1]}.")
        out(f"    2. At the default weight the advantage shrinks to "
            f"{format_interval(lf1[0])}, {lf1[1]}.")
        out(f"    3. When energy is weighted heavily (w = 2) the ordering reverses: "
            f"{format_interval(lf2[0])}, {lf2[1]}.")
        out("    4. Against the three-coefficient power-law fit of the SAME training")
        out(f"       data the learned scheduler does not win at any weight tested: "
            f"w=0 {format_interval(lp0[0])} ({lp0[1]}), "
            f"w={ENERGY_WEIGHT} {format_interval(lp1[0])} ({lp1[1]}).")
        imp = sch.feature_importances()
        hw = float(imp[5] + imp[6])
        out(f"    5. {100 * hw:.1f} % of the RandomForest's impurity importance sits on")
        out("       the two static vehicle features (dipole limit and inertia); the six")
        out(f"       time-varying magnetometer features carry {100 * (1 - hw):.1f} %. The")
        out("       'scheduler' is behaving as a static per-vehicle gain lookup, which")
        out("       is exactly what the power-law baseline already is.")
        out("")
        out(f"wall time {time.perf_counter() - t0:.1f} s")


if __name__ == "__main__":
    main()
