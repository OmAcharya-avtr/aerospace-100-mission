"""Held-out recalibration audit: when Platt and isotonic beat doing nothing.

The baseline is the raw forecast. It is implemented first, it is evaluated on
the same held-out samples as the learned maps, and the result is published
whether or not it wins.

Three things are measured:

1. the sample size at which each learned map stops harming the held-out Brier
   score, with the fraction of replicates it harms at each size;
2. what happens on a forecaster that is already calibrated, where there is
   nothing to fix and the only possible effect is damage;
3. whether the two learned maps recover the known distortion parameters, which
   is the one place where a learned component here can be checked against an
   exact answer rather than against another estimate.
"""

from __future__ import annotations

import numpy as np
from _harness import Recorder

from calibaudit.recalibration import (
    IsotonicCalibration,
    PlattScaling,
    recalibration_audit,
    sample_size_sweep,
)
from calibaudit.scores import brier_score, log_score
from calibaudit.synthetic import get_spec, sample_forecast

SEED = 56
SIZES = (60, 100, 200, 400, 1000, 2500, 6000, 15000)
REPLICATES = 80


def main() -> int:
    rec = Recorder("validate_recalibration")
    rec.header("Held-out recalibration audit - calibaudit 0.1.0")
    rec.say(
        "Baseline: the raw forecast, identity map, fitted on nothing.\n"
        "Learned: Platt scaling (2 parameters, maximum likelihood) and isotonic\n"
        "regression (nonparametric, pool-adjacent-violators). 50/50 split, fresh\n"
        f"data per replicate, {REPLICATES} replicates per sample size."
    )
    rec.say()

    findings: dict[str, int | None] = {}
    for spec_name in ("overconfident", "biased_high", "calibrated"):
        spec = get_spec(spec_name)
        sweep = sample_size_sweep(
            spec, n_samples_grid=SIZES, n_replicates=REPLICATES, seed=SEED
        )
        rec.say(f"=== {spec_name}: {spec.description} ===")
        rec.say(sweep.table())
        rec.say()
        for method in ("platt", "isotonic"):
            crossing = sweep.crossover(method)
            findings[f"{spec_name}/{method}"] = crossing
            rows = [r for r in sweep.rows if r.method == method]
            worst = max(rows, key=lambda r: r.harm_rate)
            rec.say(
                f"{method:>9}: mean held-out Brier change turns negative from "
                f"n_total = {crossing}" if crossing is not None
                else f"{method:>9}: mean held-out Brier change never turns negative "
                "on this grid"
            )
            rec.say(
                f"{'':>9}  worst harm rate {worst.harm_rate:.3f} at n_total = "
                f"{worst.n_samples} (mean change {worst.mean_delta_brier:+.6f})"
            )
        rec.say()

        smallest = min(SIZES)
        small_rows = {r.method: r for r in sweep.rows if r.n_samples == smallest}
        if spec.is_calibrated:
            rec.check(
                f"[{spec_name}] HONEST NEGATIVE: recalibration never helps a forecaster "
                "that is already calibrated",
                reference="population reliability of this spec is exactly 0, so there "
                "is nothing for a recalibration map to remove",
                measured="; ".join(
                    f"{m}: mean change {r.mean_delta_brier:+.6f} at n_total = "
                    f"{r.n_samples}, harm rate {r.harm_rate:.3f}"
                    for m, r in small_rows.items()
                    if m != "raw"
                ),
                expectation="mean held-out Brier change positive, i.e. worse than the "
                "raw baseline, at every swept sample size",
                passed=all(
                    r.mean_delta_brier > 0.0
                    for r in sweep.rows
                    if r.method != "raw"
                ),
            )
        else:
            rec.check(
                f"[{spec_name}] HONEST NEGATIVE: both learned maps harm the score at "
                f"n_total = {smallest}",
                reference="the raw forecast on the same held-out samples",
                measured="; ".join(
                    f"{m}: mean change {r.mean_delta_brier:+.6f}, harm rate "
                    f"{r.harm_rate:.3f}"
                    for m, r in small_rows.items()
                    if m != "raw"
                ),
                expectation="positive mean change for at least one learned map at the "
                "smallest sample size, because the fit's estimation variance exceeds "
                "the miscalibration it can remove",
                passed=any(
                    r.mean_delta_brier > 0.0 for m, r in small_rows.items() if m != "raw"
                ),
            )
            rec.check(
                f"[{spec_name}] isotonic needs more data than Platt",
                reference="Niculescu-Mizil and Caruana 2005, ICML, report isotonic "
                "regression needing of order 1000 samples before it beats Platt scaling",
                measured=f"Platt crosses at n_total = {findings[f'{spec_name}/platt']}, "
                f"isotonic at n_total = {findings[f'{spec_name}/isotonic']}",
                expectation="isotonic's crossing point is at or above Platt's",
                passed=(
                    findings[f"{spec_name}/isotonic"] is None
                    or (
                        findings[f"{spec_name}/platt"] is not None
                        and findings[f"{spec_name}/isotonic"]
                        >= findings[f"{spec_name}/platt"]
                    )
                ),
            )
        rec.say()

    # --- parameter recovery against the exact inverse map ------------------
    rec.say("Platt scaling against the exact inverse of each known distortion")
    rec.say()
    head = (
        f"{'spec':>15} {'n':>8} {'a_exact':>9} {'a_fitted':>9} {'b_exact':>9} "
        f"{'b_fitted':>9} {'|da|':>9} {'|db|':>9}"
    )
    rec.say(head)
    rec.say("-" * len(head))
    worst_a = worst_b = 0.0
    exact = {
        "calibrated": (1.0, 0.0),
        "overconfident": (0.6, 0.0),
        "underconfident": (1.6, 0.0),
        "biased_high": (1.0, -0.6),
    }
    for name, (a_true, b_true) in exact.items():
        for n in (2000, 20000, 200000):
            s = sample_forecast(get_spec(name), n, seed=SEED * 8000 + n)
            model = PlattScaling().fit(s.forecasts, s.outcomes)
            da, db = abs(model.a_ - a_true), abs(model.b_ - b_true)
            if n == 200000:
                worst_a = max(worst_a, da)
                worst_b = max(worst_b, db)
            rec.say(
                f"{name:>15} {n:>8d} {a_true:>9.4f} {model.a_:>9.4f} {b_true:>9.4f} "
                f"{model.b_:>9.4f} {da:>9.5f} {db:>9.5f}"
            )
    rec.say()
    rec.check(
        "Platt scaling recovers the exact inverse distortion at large n",
        reference="the analytic inverse of each spec's logistic distortion, exact",
        measured=f"at n = 200000, worst |da| = {worst_a:.5f}, worst |db| = {worst_b:.5f}",
        expectation="both below 0.03, which is about five standard errors of a "
        "two-parameter logistic fit at this sample size",
        passed=worst_a < 0.03 and worst_b < 0.03,
    )

    # --- the two scores disagree about isotonic ----------------------------
    rec.say("Where the Brier and logarithmic scores disagree about isotonic regression")
    rec.say()
    head = (
        f"{'n_total':>8} {'raw_BS':>9} {'iso_BS':>9} {'dBS':>10} "
        f"{'raw_LS':>9} {'iso_LS':>9} {'dLS':>10} {'disagree':>9}"
    )
    rec.say(head)
    rec.say("-" * len(head))
    disagreements = 0
    rows_checked = 0
    for n in (200, 400, 1000, 4000, 20000):
        draw = sample_forecast(get_spec("overconfident"), n, seed=SEED * 9000 + n)
        audit = recalibration_audit(
            draw.forecasts, draw.outcomes, n_bootstrap=200, seed=SEED
        )
        raw = audit.by_method("raw")
        iso = audit.by_method("isotonic")
        d_bs = iso.brier - raw.brier
        d_ls = iso.log_score - raw.log_score
        disagree = (d_bs < 0.0) != (d_ls < 0.0)
        disagreements += disagree
        rows_checked += 1
        rec.say(
            f"{n:>8d} {raw.brier:>9.6f} {iso.brier:>9.6f} {d_bs:>+10.6f} "
            f"{raw.log_score:>9.6f} {iso.log_score:>9.6f} {d_ls:>+10.6f} "
            f"{str(disagree):>9}"
        )
    rec.say()
    rec.check(
        "the Brier and logarithmic scores can disagree about isotonic regression",
        reference="same held-out split, same predictions, two proper scoring rules",
        measured=f"{disagreements} of {rows_checked} sample sizes disagree in sign",
        expectation=">= 1 disagreement, because pool-adjacent-violators returns "
        "exactly 0 and 1 on pure blocks and the logarithmic score punishes that "
        "while the Brier score barely notices",
        passed=disagreements >= 1,
    )

    # --- a demonstration of the exact-zero problem -------------------------
    s = sample_forecast(get_spec("overconfident"), 400, seed=SEED)
    half = s.n_samples // 2
    iso = IsotonicCalibration().fit(s.forecasts[:half], s.outcomes[:half])
    held = iso.predict(s.forecasts[half:])
    n_extreme = int(np.sum((held == 0.0) | (held == 1.0)))
    wrong_extreme = int(
        np.sum(((held == 0.0) & (s.outcomes[half:] == 1.0))
               | ((held == 1.0) & (s.outcomes[half:] == 0.0)))
    )
    rec.say(
        f"isotonic on a 200-sample training split emitted an exact 0 or 1 for "
        f"{n_extreme} of {s.n_samples - half} held-out forecasts, and "
        f"{wrong_extreme} of those were wrong."
    )
    rec.say(
        f"held-out log score: raw {log_score(s.forecasts[half:], s.outcomes[half:]):.6f}, "
        f"isotonic {log_score(held, s.outcomes[half:]):.6f}; "
        f"held-out Brier: raw {brier_score(s.forecasts[half:], s.outcomes[half:]):.6f}, "
        f"isotonic {brier_score(held, s.outcomes[half:]):.6f}"
    )
    rec.say()
    rec.check(
        "isotonic regression emits exact 0 and 1 on held-out data",
        reference="pool-adjacent-violators returns the block mean, which is 0 or 1 on "
        "a pure block",
        measured=f"{n_extreme} of {s.n_samples - half} held-out predictions were exactly "
        f"0 or 1, {wrong_extreme} of them wrong",
        expectation="> 0 extreme predictions on a 200-sample training split",
        passed=n_extreme > 0,
    )
    return rec.finish()


if __name__ == "__main__":
    raise SystemExit(main())
