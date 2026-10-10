"""Command line interface: ``python -m conformalband``.

Exit status
-----------
``0``
    The command ran and produced its report.
``2``
    A refused request (for example a calibration size too small for the
    requested ``alpha``), or, with ``--fail-on-undercoverage``, an audit in
    which at least one method was measured as demonstrably below nominal
    coverage. A build that treats under-coverage as a defect can rely on this.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict

import numpy as np

from . import __version__
from .audit import (
    DEFAULT_ALPHA,
    DEFAULT_N_CALIBRATION,
    DEFAULT_N_FIT,
    DEFAULT_N_TEST,
    DEFAULT_SEED,
    METHODS,
    MODELS,
    breaking_point_sweep,
    coverage_audit,
    stratified_coverage,
)
from .baseline import GaussianResidualInterval, PhysicsRegressor
from .bounds import split_conformal_coverage_bound
from .conformal import SplitConformal
from .data import FEATURE_NAMES, FEATURE_UNITS, make_audit_split
from .learned import LearnedRegressor
from .physics import DEFAULT_AIRFRAME
from .shift import SHIFT_LEVELS, CovariateShift

EXIT_REFUSED = 2


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m conformalband",
        description=(
            "Distribution-free prediction intervals for a UAV energy regression, with "
            "coverage measured under a declared covariate shift. Research-grade; not "
            "flight-qualified, not certified, not approved for operational aerospace use."
        ),
    )
    parser.add_argument("--version", action="version", version=f"conformalband {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    info = sub.add_parser("info", help="airframe, covariate declaration and shift levels")
    info.add_argument("--json", action="store_true", help="machine-readable output")

    bound = sub.add_parser("bound", help="finite-sample split-conformal coverage bound")
    bound.add_argument("--n", type=int, default=DEFAULT_N_CALIBRATION, help="calibration size")
    bound.add_argument("--alpha", type=float, default=DEFAULT_ALPHA, help="miscoverage level")
    bound.add_argument("--json", action="store_true")

    baseline = sub.add_parser(
        "baseline", help="fit the analytic baseline and the learned model on one replicate"
    )
    baseline.add_argument("--seed", type=int, default=DEFAULT_SEED)
    baseline.add_argument("--n-fit", type=int, default=DEFAULT_N_FIT)
    baseline.add_argument("--n-calibration", type=int, default=DEFAULT_N_CALIBRATION)
    baseline.add_argument("--n-test", type=int, default=DEFAULT_N_TEST)
    baseline.add_argument("--severity", type=float, default=0.0)
    baseline.add_argument("--alpha", type=float, default=DEFAULT_ALPHA)
    baseline.add_argument("--json", action="store_true")

    audit = sub.add_parser("audit", help="the coverage audit table")
    audit.add_argument("--replicates", type=int, default=20)
    audit.add_argument("--alpha", type=float, default=DEFAULT_ALPHA)
    audit.add_argument("--n-fit", type=int, default=DEFAULT_N_FIT)
    audit.add_argument("--n-calibration", type=int, default=DEFAULT_N_CALIBRATION)
    audit.add_argument("--n-test", type=int, default=DEFAULT_N_TEST)
    audit.add_argument("--seed", type=int, default=DEFAULT_SEED)
    audit.add_argument("--severities", type=float, nargs="+", default=list(SHIFT_LEVELS))
    audit.add_argument("--models", nargs="+", default=list(MODELS), choices=list(MODELS))
    audit.add_argument("--methods", nargs="+", default=list(METHODS), choices=list(METHODS))
    audit.add_argument(
        "--fail-on-undercoverage",
        action="store_true",
        help=f"exit {EXIT_REFUSED} if any row is demonstrably below nominal coverage",
    )
    audit.add_argument("--json", action="store_true")

    breaking = sub.add_parser(
        "breaking-point", help="weighted conformal coverage against misspecified weights"
    )
    breaking.add_argument("--true-severity", type=float, default=2.0)
    breaking.add_argument("--replicates", type=int, default=20)
    breaking.add_argument("--alpha", type=float, default=DEFAULT_ALPHA)
    breaking.add_argument("--n-calibration", type=int, default=DEFAULT_N_CALIBRATION)
    breaking.add_argument("--n-test", type=int, default=DEFAULT_N_TEST)
    breaking.add_argument("--seed", type=int, default=DEFAULT_SEED)
    breaking.add_argument("--model", default="learned", choices=list(MODELS))
    breaking.add_argument("--json", action="store_true")

    strata = sub.add_parser("strata", help="per-tercile coverage, split against Mondrian")
    strata.add_argument("--replicates", type=int, default=10)
    strata.add_argument("--severity", type=float, default=2.0)
    strata.add_argument("--alpha", type=float, default=DEFAULT_ALPHA)
    strata.add_argument("--model", default="learned", choices=list(MODELS))
    strata.add_argument("--seed", type=int, default=DEFAULT_SEED)
    strata.add_argument("--json", action="store_true")
    return parser


def _cmd_info(args: argparse.Namespace) -> int:
    airframe = DEFAULT_AIRFRAME
    payload = {
        "version": __version__,
        "airframe": {
            "wing_area_m2": airframe.wing_area,
            "wingspan_m": airframe.wingspan,
            "aspect_ratio": airframe.aspect_ratio,
            "cd0": airframe.cd0,
            "oswald": airframe.oswald,
            "eta_prop": airframe.eta_prop,
            "avionics_power_W": airframe.avionics_power,
            "design_airspeed_mps": airframe.design_airspeed,
            "eta_curvature": airframe.eta_curvature,
        },
        "features": dict(zip(FEATURE_NAMES, FEATURE_UNITS, strict=True)),
        "shift_levels": [CovariateShift(severity=s).describe() for s in SHIFT_LEVELS],
        "methods": list(METHODS),
        "models": list(MODELS),
    }
    if args.json:
        print(json.dumps(payload, indent=2))
        return 0
    print(f"conformalband {__version__}")
    print("airframe (illustrative, not a measured vehicle)")
    for key, value in payload["airframe"].items():
        print(f"  {key:22s}: {value}")
    print("covariates")
    for name, unit in zip(FEATURE_NAMES, FEATURE_UNITS, strict=True):
        print(f"  {name:22s}: {unit}")
    print("declared covariate shifts")
    for line in payload["shift_levels"]:
        print(f"  {line}")
    print(f"interval methods: {', '.join(METHODS)}")
    print(f"point predictors: {', '.join(MODELS)}")
    return 0


def _cmd_bound(args: argparse.Namespace) -> int:
    try:
        bound = split_conformal_coverage_bound(args.n, args.alpha)
    except ValueError as exc:
        print(f"refused: {exc}", file=sys.stderr)
        return EXIT_REFUSED
    payload = asdict(bound) | {"conservatism": bound.conservatism}
    if args.json:
        print(json.dumps(payload, indent=2))
        return 0
    print("split conformal finite-sample coverage bound")
    print(f"  n_calibration     : {bound.n_calibration}")
    print(f"  alpha             : {bound.alpha}")
    print(f"  rank k            : {bound.rank}  = ceil((n+1)(1-alpha))")
    print(f"  exact coverage    : {bound.exact:.12f}  = k/(n+1)")
    print(f"  guaranteed lower  : {bound.lower:.12f}  = 1-alpha")
    print(f"  matching upper    : {bound.upper:.12f}  = 1-alpha+1/(n+1)")
    print(f"  forced over-cover : {bound.conservatism:.12f}")
    print("  assumption        : exchangeability of the n+1 conformity scores")
    return 0


def _cmd_baseline(args: argparse.Namespace) -> int:
    split = make_audit_split(
        seed=args.seed,
        n_fit=args.n_fit,
        n_calibration=args.n_calibration,
        n_test=args.n_test,
        severity=args.severity,
    )
    physics = PhysicsRegressor().fit(split.fit.features, split.fit.energy)
    learned = LearnedRegressor(random_state=args.seed).fit(split.fit.features, split.fit.energy)
    payload: dict[str, object] = {
        "seed": args.seed,
        "severity": args.severity,
        "physics_parameters": physics.parameter_table(),
        "models": {},
    }
    for name, model, n_parameters in (
        ("physics", physics, PhysicsRegressor.n_parameters),
        ("learned", learned, 0),
    ):
        pred_cal = model.predict(split.calibration.features)
        pred_test = model.predict(split.test.features)
        rmse_cal = float(np.sqrt(np.mean((split.calibration.energy - pred_cal) ** 2)))
        rmse_test = float(np.sqrt(np.mean((split.test.energy - pred_test) ** 2)))
        parametric = GaussianResidualInterval(args.alpha, n_parameters=n_parameters).fit(
            split.calibration.energy, pred_cal
        )
        conformal = SplitConformal(args.alpha).calibrate(split.calibration.energy, pred_cal)
        parametric_interval = parametric.interval(pred_test)
        conformal_interval = conformal.interval(pred_test)
        payload["models"][name] = {
            "rmse_calibration_Wh": rmse_cal,
            "rmse_test_Wh": rmse_test,
            "sigma_hat_Wh": parametric.sigma,
            "parametric_half_width_Wh": parametric.half_width,
            "conformal_quantile_Wh": conformal.quantile,
            "parametric_coverage": float(parametric_interval.covers(split.test.energy).mean()),
            "conformal_coverage": float(conformal_interval.covers(split.test.energy).mean()),
            "parametric_width_Wh": float(np.mean(parametric_interval.width)),
            "conformal_width_Wh": float(np.mean(conformal_interval.width)),
        }
    if args.json:
        print(json.dumps(payload, indent=2))
        return 0
    print(f"one replicate, seed {args.seed}, declared severity {args.severity}")
    print("analytic baseline fitted coefficients")
    for key, value in physics.parameter_table().items():
        print(f"  {key:16s}: {value:.9f}")
    header = (
        f"{'model':8s} {'rmse_cal':>9s} {'rmse_test':>10s} {'sigma_Wh':>9s} "
        f"{'param_half':>10s} {'conf_q':>8s} {'param_cov':>10s} {'conf_cov':>9s}"
    )
    print(header)
    print("-" * len(header))
    for name in ("physics", "learned"):
        m = payload["models"][name]
        print(
            f"{name:8s} {m['rmse_calibration_Wh']:9.6f} {m['rmse_test_Wh']:10.6f} "
            f"{m['sigma_hat_Wh']:9.6f} {m['parametric_half_width_Wh']:10.6f} "
            f"{m['conformal_quantile_Wh']:8.6f} {m['parametric_coverage']:10.6f} "
            f"{m['conformal_coverage']:9.6f}"
        )
    return 0


def _cmd_audit(args: argparse.Namespace) -> int:
    result = coverage_audit(
        alpha=args.alpha,
        replicates=args.replicates,
        n_fit=args.n_fit,
        n_calibration=args.n_calibration,
        n_test=args.n_test,
        seed=args.seed,
        severities=tuple(args.severities),
        models=tuple(args.models),
        methods=tuple(args.methods),
    )
    failures = [r for r in result.rows if r.below_nominal]
    if args.json:
        print(
            json.dumps(
                {
                    "config": {
                        "alpha": result.alpha,
                        "replicates": result.replicates,
                        "n_fit": result.n_fit,
                        "n_calibration": result.n_calibration,
                        "n_test": result.n_test,
                        "seed": result.seed,
                        "severities": list(result.severities),
                    },
                    "rows": [asdict(r) for r in result.rows],
                    "demonstrably_below_nominal": [
                        {"model": r.model, "method": r.method, "severity": r.severity}
                        for r in failures
                    ],
                },
                indent=2,
            )
        )
    else:
        print(result.table())
        print()
        print(
            f"rows demonstrably below nominal coverage (ci_high < {1.0 - args.alpha}): "
            f"{len(failures)} of {len(result.rows)}"
        )
        for row in failures:
            print(
                f"  {row.model}/{row.method} at severity {row.severity:.1f}: "
                f"coverage {row.coverage:.5f}, 95 % CI [{row.ci_low:.5f}, {row.ci_high:.5f}]"
            )
    if args.fail_on_undercoverage and failures:
        print(
            f"refused: {len(failures)} row(s) demonstrably below nominal coverage",
            file=sys.stderr,
        )
        return EXIT_REFUSED
    return 0


def _cmd_breaking_point(args: argparse.Namespace) -> int:
    result = breaking_point_sweep(
        true_severity=args.true_severity,
        alpha=args.alpha,
        replicates=args.replicates,
        n_calibration=args.n_calibration,
        n_test=args.n_test,
        seed=args.seed,
        model=args.model,
    )
    if args.json:
        print(
            json.dumps(
                {
                    "true_severity": result.true_severity,
                    "model": result.model,
                    "alpha": result.alpha,
                    "replicates": result.replicates,
                    "last_holding_fraction": result.last_holding_fraction,
                    "breaking_fraction": result.breaking_fraction,
                    "rows": [asdict(r) for r in result.rows],
                },
                indent=2,
            )
        )
        return 0
    print(result.table())
    print()
    print(f"true severity              : {result.true_severity}")
    print(f"last holding fraction <= 1 : {result.last_holding_fraction}")
    print(f"breaking fraction          : {result.breaking_fraction}")
    print(
        "breaking fraction is the largest assumed-shift fraction whose 95 % coverage "
        "interval lies entirely below nominal"
    )
    return 0


def _cmd_strata(args: argparse.Namespace) -> int:
    tally = stratified_coverage(
        alpha=args.alpha,
        replicates=args.replicates,
        severity=args.severity,
        seed=args.seed,
        model=args.model,
    )
    if args.json:
        print(json.dumps(tally, indent=2))
        return 0
    print(f"per-tercile coverage, severity {args.severity}, model {args.model}")
    header = f"{'method':10s} {'tercile':>8s} {'coverage':>9s} {'width_Wh':>9s} {'n':>7s}"
    print(header)
    print("-" * len(header))
    for method, per_bin in tally.items():
        for b in sorted(per_bin):
            coverage, width, _, count = per_bin[b]
            print(f"{method:10s} {b + 1:8d} {coverage:9.5f} {width:9.5f} {count:7d}")
    return 0


_DISPATCH = {
    "info": _cmd_info,
    "bound": _cmd_bound,
    "baseline": _cmd_baseline,
    "audit": _cmd_audit,
    "breaking-point": _cmd_breaking_point,
    "strata": _cmd_strata,
}


def main(argv: list[str] | None = None) -> int:
    """Entry point. Returns the process exit status."""
    args = _build_parser().parse_args(argv)
    return _DISPATCH[args.command](args)


if __name__ == "__main__":
    raise SystemExit(main())
