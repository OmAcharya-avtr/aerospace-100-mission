"""Agreement with scikit-learn, and the 1.9 API change that breaks prefit recalibration.

scikit-learn is already a runtime dependency here, so these are real
cross-implementation checks: where this package computes something sklearn
also computes, the two must agree to machine precision or the difference must
be explained.

The recorded API change matters to anyone doing what this package is for.
``CalibratedClassifierCV(estimator, cv="prefit")`` was the documented way to
recalibrate an already-fitted classifier. On scikit-learn 1.9.1 it raises,
because ``cv`` no longer accepts the string and the prefit path is spelled
``CalibratedClassifierCV(FrozenEstimator(estimator))``. The exact exception
text is captured below so a reader who hits it can match it.
"""

from __future__ import annotations

import numpy as np
from _harness import Recorder
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.frozen import FrozenEstimator
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss
from sklearn.utils._param_validation import InvalidParameterError

from calibaudit.recalibration import IsotonicCalibration, PlattScaling
from calibaudit.reliability import reliability_curve
from calibaudit.scores import brier_score, log_score
from calibaudit.synthetic import SPEC_NAMES, get_spec, sample_forecast

SEED = 56


def main() -> int:
    rec = Recorder("validate_sklearn_interop")
    rec.header("Agreement with scikit-learn 1.9.1 - calibaudit 0.1.0")

    # --- scores ------------------------------------------------------------
    rec.say("Scores against sklearn.metrics, over every shipped spec at n = 20000")
    rec.say()
    head = (
        f"{'spec':>19} {'ours BS':>12} {'sklearn BS':>12} {'|d|':>10} "
        f"{'ours LS':>12} {'sklearn LS':>12} {'|d|':>10}"
    )
    rec.say(head)
    rec.say("-" * len(head))
    worst_bs = worst_ls = 0.0
    for name in SPEC_NAMES:
        s = sample_forecast(get_spec(name), 20000, seed=SEED)
        ours_bs = brier_score(s.forecasts, s.outcomes)
        theirs_bs = float(brier_score_loss(s.outcomes, s.forecasts))
        ours_ls = log_score(s.forecasts, s.outcomes)
        theirs_ls = float(log_loss(s.outcomes, s.forecasts, labels=[0, 1]))
        worst_bs = max(worst_bs, abs(ours_bs - theirs_bs))
        worst_ls = max(worst_ls, abs(ours_ls - theirs_ls))
        rec.say(
            f"{name:>19} {ours_bs:>12.9f} {theirs_bs:>12.9f} {abs(ours_bs - theirs_bs):>10.2e} "
            f"{ours_ls:>12.9f} {theirs_ls:>12.9f} {abs(ours_ls - theirs_ls):>10.2e}"
        )
    rec.say()
    rec.check(
        "Brier score agrees with sklearn.metrics.brier_score_loss",
        reference="sklearn 1.9.1 brier_score_loss on the same arrays",
        measured=f"worst absolute difference {worst_bs:.3e} over 6 specs at n = 20000",
        expectation="<= 1e-14",
        passed=worst_bs <= 1e-14,
    )
    rec.check(
        "logarithmic score agrees with sklearn.metrics.log_loss",
        reference="sklearn 1.9.1 log_loss with labels=[0, 1]",
        measured=f"worst absolute difference {worst_ls:.3e}",
        expectation="<= 1e-12; the two clip differently in principle but not on any "
        "forecast in these samples",
        passed=worst_ls <= 1e-12,
    )

    # --- reliability curve -------------------------------------------------
    rec.say("Reliability curve against sklearn.calibration.calibration_curve")
    rec.say()
    head = (
        f"{'spec':>19} {'B':>4} {'our bins':>9} {'their bins':>11} "
        f"{'worst |d obs|':>14} {'worst |d mean_f|':>17}"
    )
    rec.say(head)
    rec.say("-" * len(head))
    worst_obs = worst_mean = 0.0
    aligned = True
    for name in SPEC_NAMES:
        s = sample_forecast(get_spec(name), 8000, seed=SEED + 1)
        for n_bins in (10, 20):
            prob_true, prob_pred = calibration_curve(
                s.outcomes, s.forecasts, n_bins=n_bins, strategy="uniform"
            )
            ours = reliability_curve(
                s.forecasts, s.outcomes, n_bins=n_bins, strategy="equal_width"
            )
            same = ours.n_occupied == prob_true.size
            aligned = aligned and same
            d_obs = (
                float(np.max(np.abs(ours.observed_frequency - prob_true))) if same else np.nan
            )
            d_mean = (
                float(np.max(np.abs(ours.mean_forecast - prob_pred))) if same else np.nan
            )
            if same:
                worst_obs = max(worst_obs, d_obs)
                worst_mean = max(worst_mean, d_mean)
            rec.say(
                f"{name:>19} {n_bins:>4d} {ours.n_occupied:>9d} {prob_true.size:>11d} "
                f"{d_obs:>14.2e} {d_mean:>17.2e}"
            )
    rec.say()
    rec.check(
        "equal-width reliability curve reproduces sklearn's uniform-strategy curve",
        reference="sklearn.calibration.calibration_curve(strategy='uniform')",
        measured=f"bin counts matched in every case: {aligned}; worst |d observed| = "
        f"{worst_obs:.3e}, worst |d mean forecast| = {worst_mean:.3e}",
        expectation="identical bin counts and differences <= 1e-12",
        passed=aligned and worst_obs <= 1e-12 and worst_mean <= 1e-12,
    )

    # --- isotonic ----------------------------------------------------------
    s = sample_forecast(get_spec("overconfident"), 8000, seed=SEED + 2)
    ours_iso = IsotonicCalibration().fit(s.forecasts, s.outcomes)
    theirs_iso = IsotonicRegression(
        y_min=0.0, y_max=1.0, increasing=True, out_of_bounds="clip"
    ).fit(s.forecasts, s.outcomes)
    grid = np.linspace(0.0, 1.0, 1001)
    d_iso = float(np.max(np.abs(ours_iso.predict(grid) - theirs_iso.predict(grid))))
    rec.say(
        f"isotonic map, 1001-point grid, worst absolute difference against a direct "
        f"sklearn fit: {d_iso:.3e}  (fitted step count {ours_iso.n_steps})"
    )
    rec.say()
    rec.check(
        "IsotonicCalibration is sklearn's IsotonicRegression, with the arguments fixed",
        reference="sklearn.isotonic.IsotonicRegression(y_min=0, y_max=1, "
        "increasing=True, out_of_bounds='clip')",
        measured=f"worst absolute difference {d_iso:.3e} over 1001 grid points",
        expectation="exactly 0; this package wraps rather than reimplements",
        passed=d_iso == 0.0,
    )

    # --- Platt against an unregularised logistic regression ----------------
    rec.say("Platt scaling against sklearn.linear_model.LogisticRegression on logit(f)")
    rec.say()
    head = (
        f"{'spec':>19} {'n':>8} {'our a':>10} {'their a':>10} {'our b':>10} "
        f"{'their b':>10} {'|da|':>9} {'|db|':>9}"
    )
    rec.say(head)
    rec.say("-" * len(head))
    worst_a = worst_b = 0.0
    for name in ("calibrated", "overconfident", "underconfident", "biased_high"):
        for n in (4000, 40000):
            s = sample_forecast(get_spec(name), n, seed=SEED + 3)
            z = np.log(s.forecasts / (1.0 - s.forecasts)).reshape(-1, 1)
            theirs = LogisticRegression(C=1e8, tol=1e-12, max_iter=5000).fit(z, s.outcomes)
            ours = PlattScaling().fit(s.forecasts, s.outcomes)
            da = abs(ours.a_ - float(theirs.coef_[0, 0]))
            db = abs(ours.b_ - float(theirs.intercept_[0]))
            worst_a, worst_b = max(worst_a, da), max(worst_b, db)
            rec.say(
                f"{name:>19} {n:>8d} {ours.a_:>10.6f} {float(theirs.coef_[0, 0]):>10.6f} "
                f"{ours.b_:>10.6f} {float(theirs.intercept_[0]):>10.6f} {da:>9.2e} {db:>9.2e}"
            )
    rec.say()
    rec.check(
        "Platt scaling agrees with a near-unregularised logistic regression",
        reference="sklearn LogisticRegression(C=1e8) on the single feature logit(f)",
        measured=f"worst |da| = {worst_a:.3e}, worst |db| = {worst_b:.3e}",
        expectation="<= 1e-3; sklearn still applies L2 at C = 1e8 and stops on its own "
        "tolerance, so exact agreement is not expected and this package fits the two "
        "parameters directly rather than delegating",
        passed=worst_a <= 1e-3 and worst_b <= 1e-3,
    )

    # --- the 1.9 prefit removal -------------------------------------------
    rec.say("The scikit-learn 1.9 prefit removal, which this package had to route around")
    rec.say()
    rng = np.random.default_rng(SEED)
    x = rng.normal(size=(400, 3))
    y = (x[:, 0] + rng.normal(scale=0.5, size=400) > 0).astype(int)
    base = LogisticRegression().fit(x, y)

    raised = None
    try:
        CalibratedClassifierCV(base, cv="prefit", method="sigmoid").fit(x, y)
    except InvalidParameterError as exc:
        raised = exc
    rec.say("  attempted: CalibratedClassifierCV(base, cv='prefit', method='sigmoid')")
    if raised is None:
        rec.say("  no exception raised")
    else:
        rec.say(f"  raised   : {type(raised).__name__}")
        rec.say(f"  message  : {raised}")
    rec.say()
    rec.check(
        "CalibratedClassifierCV(cv='prefit') raises on this scikit-learn",
        reference="sklearn 1.9.1 parameter validation for CalibratedClassifierCV.cv",
        measured=f"{type(raised).__name__ if raised else 'no exception'}",
        expectation="InvalidParameterError; the documented prefit path of earlier "
        "versions is gone and code copied from an older tutorial will fail here",
        passed=isinstance(raised, InvalidParameterError),
    )

    frozen = CalibratedClassifierCV(FrozenEstimator(base), method="sigmoid").fit(x, y)
    p_frozen = frozen.predict_proba(x)[:, 1]
    rec.say(
        "  replacement: CalibratedClassifierCV(FrozenEstimator(base), method='sigmoid')"
    )
    rec.say(
        f"  fitted, held-in Brier {brier_score(p_frozen, y.astype(float)):.9f} against "
        f"raw {brier_score(base.predict_proba(x)[:, 1], y.astype(float)):.9f}"
    )
    rec.say()
    rec.check(
        "FrozenEstimator is a working replacement",
        reference="sklearn.frozen.FrozenEstimator, new in 1.6",
        measured=f"fitted and produced {p_frozen.size} probabilities, all in [0, 1]: "
        f"{bool(np.all((p_frozen >= 0) & (p_frozen <= 1)))}",
        expectation="fits without error and returns valid probabilities",
        passed=bool(np.all((p_frozen >= 0.0) & (p_frozen <= 1.0))),
    )
    rec.say(
        "This package never takes that path: it fits calibration maps on forecast\n"
        "values directly, so it has no estimator to freeze. The check is here because\n"
        "anyone recalibrating a prefit classifier on scikit-learn 1.9 will hit it, and\n"
        "because an earlier session in this mission lost time to it."
    )
    return rec.finish()


if __name__ == "__main__":
    raise SystemExit(main())
