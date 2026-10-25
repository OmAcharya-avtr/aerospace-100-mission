"""The worked example from README.md, executed, with its real output.

Everything printed below appears verbatim in README.md. It is re-run as the
last step before hand-off, so a quoted number cannot drift from what the code
produces.

The example is the question the product exists for. An operator holds a
two-line element set, needs the object's position to better than a declared
tolerance for something they are about to do, and wants to know whether the
element set is still good enough -- with a figure attached to the answer rather
than a rule of thumb.
"""

from __future__ import annotations

import sys

from _bootstrap import Tee, Timer, bootstrap, environment_lines, is_fast, load_dataset

bootstrap()

import numpy as np  # noqa: E402
from sgp4.api import Satrec  # noqa: E402

from orbitdrift import (  # noqa: E402
    AnalyticGrowthPredictor,
    ConformalPredictor,
    build_report,
    features_from_satrec,
    model_divergence,
    split_by_object,
    staleness_verdict,
)
from orbitdrift.calibration import interval_metrics  # noqa: E402

LINE1 = "1 25544U 98067A   26015.50000000  .00016717  00000-0  10270-3 0  9003"
LINE2 = "2 25544  51.6416 247.4627 0006703 130.5360 325.0288 15.72125391563537"
TOLERANCE_KM = 5.0


def main() -> int:
    out = Tee("worked_example_output.txt")
    out.rule("orbitdrift -- worked example")
    for line in environment_lines():
        out(line)

    out.rule("step 1: the element set")
    out("  The catalogue number and the orbit are those of a real object. The")
    out("  element values are a plausible made-up set and are NOT a real published")
    out("  element set for it; nothing in this repository has read the catalogue.")
    out("")
    out(f"  {LINE1}")
    out(f"  {LINE2}")
    satrec = Satrec.twoline2rv(LINE1, LINE2)
    features = features_from_satrec(satrec)
    out("")
    out(f"  perigee altitude              : {features.perigee_altitude_km:.2f} km")
    out(f"  apogee altitude               : {features.apogee_altitude_km:.2f} km")
    out(f"  eccentricity                  : {features.eccentricity:.7f}")
    out(f"  inclination                   : {features.inclination_deg:.4f} deg")
    out(f"  B*                            : {features.bstar:.6e} 1/earth-radii")
    out(f"  period                        : {features.period_min:.4f} min")

    out.rule("step 2: the divergence curve (baseline 2)")
    out("  SGP4 against the independent numerical force model, started from the")
    out("  SGP4 state at the element-set epoch so the curve is zero at zero age.")
    timer = Timer()
    span_days = 0.3 if is_fast() else 5.0
    with timer:
        curve = model_divergence(
            satrec,
            span_days=span_days,
            steps_per_rev=360,
            report_every_hours=12.0,
            sample_every=4,
        )
    report = build_report(satrec, curve)
    out("")
    for line in report.summary().splitlines():
        out(f"  {line}")

    out.rule("step 3: a predictor with a MEASURED interval, fitted on held-out objects")
    out("  A single element set cannot calibrate an interval: one curve gives one")
    out("  fit and no held-out data. The interval below comes from the committed")
    out("  population, fitted on the training objects and calibrated by split")
    out("  conformal prediction on a disjoint set of objects, and its coverage is")
    out("  measured on the test objects -- which is the number that makes the")
    out("  horizon in step 4 mean anything.")
    table = load_dataset()
    train, test = split_by_object(table, test_fraction=0.3)
    design = table.design()
    drift = table.column("drift_km")
    objects = table.column("object_index")
    predictor = ConformalPredictor(
        AnalyticGrowthPredictor(form="power_altitude"), coverage=0.90
    )
    predictor.fit(design[train], drift[train], groups=objects[train])
    lower, upper = predictor.predict_interval_km(design[test])
    metrics = interval_metrics(lower, upper, drift[test], nominal=0.90)
    out("")
    out(f"  predictor                     : {predictor.name}")
    out(f"  training objects              : {int(np.unique(objects[train]).size)}")
    out(f"  calibration rows              : {predictor.n_calibration_}")
    out(f"  held-out objects              : {int(np.unique(objects[test]).size)}")
    out(f"  held-out rows                 : {metrics.n}")
    out(f"  nominal coverage              : {metrics.nominal:.4f}")
    out(f"  MEASURED coverage             : {metrics.coverage:.4f} "
        f"[{metrics.coverage_ci[0]:.4f}, {metrics.coverage_ci[1]:.4f}]")
    out(f"  median interval width         : {metrics.median_width_km:.4f} km")
    out(f"  median log interval width     : {metrics.median_log_width:.4f}")
    out(f"  mean interval score           : {metrics.interval_score:.4f} km")

    out.rule("step 4: the decision")
    out("  The question: this element set is three days old and I need the")
    out(f"  position to better than {TOLERANCE_KM:.1f} km. Is it still good enough?")
    verdict = staleness_verdict(
        predictor,
        features,
        TOLERANCE_KM,
        age_days=3.0,
        purpose="ephemeris staleness check",
        coverage=0.90,
        measured_coverage=metrics.coverage,
    )
    out("")
    for line in verdict.describe().splitlines():
        out(f"  {line}")
    out("")
    out("  Read it as follows. The median horizon is the optimistic answer and the")
    out("  one a point prediction gives: half the time the drift is already worse")
    out("  than it says. The conservative horizon is the one to use, and it is only")
    out("  as good as the measured coverage printed beside it.")

    out.rule("step 5: what this is not")
    out("  The verdict above is a staleness horizon under a declared position")
    out("  tolerance. It is not a covariance and it is not an input to a")
    out("  conjunction assessment. Asking for either is refused:")
    from orbitdrift.errors import CovarianceRefused

    for label, call in (
        ("report.as_covariance()", report.as_covariance),
        ("verdict.as_covariance()", verdict.as_covariance),
    ):
        try:
            call()
            out(f"    [NOT REFUSED] {label}")
            out.check(f"{label} raises", False, "it returned")
        except CovarianceRefused as error:
            out(f"    {label} -> CovarianceRefused")
            out(f"      {str(error)[:76]}")
            out.check(f"{label} raises and names P071", "P071" in str(error), "")

    out.rule("checks on the numbers above")
    out.check(
        "the divergence curve is monotone increasing",
        bool(np.all(np.diff(curve.drift_km) > 0.0)),
        f"{curve.drift_km.size} ages",
    )
    out.check(
        "the conservative horizon is not later than the median horizon",
        verdict.conservative_horizon_days <= verdict.median_horizon_days,
        f"{verdict.conservative_horizon_days:.4f} against "
        f"{verdict.median_horizon_days:.4f} days",
    )
    out.check(
        "the verdict reports a measured coverage rather than a nominal one",
        np.isfinite(verdict.measured_coverage),
        f"{verdict.measured_coverage:.4f} measured against "
        f"{verdict.nominal_coverage:.4f} nominal",
    )
    out.check(
        "the serialised report carries no covariance-shaped key",
        "covariance" not in str(sorted(report.to_dict())),
        f"{len(report.to_dict())} top-level keys",
    )

    out.rule("timings (UNSTABLE: wall clock on a shared two-core container)")
    out(f"  divergence curve              : {timer.elapsed_s:.1f} s")

    out("")
    out(f"wrote {out.save()}")
    return 1 if out.failures else 0


if __name__ == "__main__":
    sys.exit(main())
