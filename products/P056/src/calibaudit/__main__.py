"""Command line interface: ``python -m calibaudit``.

Subcommands
-----------
``specs``        list the shipped synthetic forecasters and their exact scores
``decompose``    Murphy decomposition, exact and binned, with both residuals
``ece``          raw ECE, its estimated binning bias, and the debiased value
``ece-bias``     the bias of the ECE estimator over a (bins, samples) grid
``reliability``  reliability table with pointwise bootstrap bands
``recalibrate``  held-out audit of Platt and isotonic against the raw forecast

Exit status is 0 on success, 1 on a usage or input error, and 2 only from
``recalibrate --require-improvement`` when no learned recalibrator improves
the held-out Brier score at the stated level, so a build can treat "the
recalibration did not help" as a failure.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict

import numpy as np

from . import __version__
from .decomposition import binned_decomposition, murphy_decomposition
from .ece import debiased_ece, ece_bias_curve
from .recalibration import METHOD_NAMES, recalibration_audit
from .reliability import bootstrap_reliability
from .scores import brier_score, brier_skill_score, log_score, log_skill_score
from .synthetic import SPEC_NAMES, analytic_truth, get_spec, sample_forecast

EXIT_OK = 0
EXIT_USAGE = 1
EXIT_NO_IMPROVEMENT = 2


def _spec_sample(args: argparse.Namespace):
    spec = get_spec(args.spec)
    return spec, sample_forecast(spec, args.n_samples, seed=args.seed)


def _print_json(payload: dict) -> None:
    def default(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        raise TypeError(f"not JSON serialisable: {type(obj).__name__}")

    print(json.dumps(payload, indent=2, default=default, sort_keys=True))


def _cmd_specs(args: argparse.Namespace) -> int:
    rows = []
    for name in SPEC_NAMES:
        spec = get_spec(name)
        truth = analytic_truth(spec)
        rows.append(
            {
                "name": name,
                "beta_a": spec.beta_a,
                "beta_b": spec.beta_b,
                "kind": spec.kind,
                "param": spec.param,
                "calibrated": spec.is_calibrated,
                "base_rate": truth.base_rate,
                "uncertainty": truth.uncertainty,
                "resolution": truth.resolution,
                "reliability": truth.reliability,
                "brier": truth.brier,
                "ece": truth.ece,
                "log_score": truth.log_score,
                "description": spec.description,
            }
        )
    if args.json:
        _print_json({"specs": rows})
        return EXIT_OK
    head = (
        f"{'name':>19} {'cal':>5} {'obar':>7} {'UNC':>9} {'RES':>9} "
        f"{'REL':>11} {'BS':>9} {'ECE':>9} {'LS':>9}"
    )
    print(head)
    print("-" * len(head))
    for r in rows:
        print(
            f"{r['name']:>19} {str(r['calibrated']):>5} {r['base_rate']:>7.4f} "
            f"{r['uncertainty']:>9.6f} {r['resolution']:>9.6f} {r['reliability']:>11.3e} "
            f"{r['brier']:>9.6f} {r['ece']:>9.6f} {r['log_score']:>9.6f}"
        )
    print()
    print("Population values. UNC, RES and obar are closed-form Beta moments;")
    print("REL, ECE and LS are quadrature integrals (scipy.integrate.quad).")
    return EXIT_OK


def _cmd_decompose(args: argparse.Namespace) -> int:
    spec, s = _spec_sample(args)
    truth = analytic_truth(spec)
    exact = murphy_decomposition(np.round(s.forecasts, args.round_digits), s.outcomes)
    binned = binned_decomposition(
        s.forecasts, s.outcomes, n_bins=args.bins, strategy=args.strategy
    )
    if args.json:
        _print_json(
            {
                "spec": spec.name,
                "n_samples": s.n_samples,
                "seed": args.seed,
                "population": {
                    k: v
                    for k, v in asdict(truth).items()
                    if k != "spec"
                },
                "exact_on_rounded": {
                    "round_digits": args.round_digits,
                    "brier": exact.brier,
                    "reliability": exact.reliability,
                    "resolution": exact.resolution,
                    "uncertainty": exact.uncertainty,
                    "three_term_residual": exact.three_term_residual,
                    "identity_residual": exact.identity_residual,
                    "n_bins_occupied": exact.n_bins_occupied,
                },
                "binned": {
                    "n_bins": args.bins,
                    "strategy": args.strategy,
                    "brier": binned.brier,
                    "reliability": binned.reliability,
                    "resolution": binned.resolution,
                    "uncertainty": binned.uncertainty,
                    "within_bin_variance": binned.within_bin_variance,
                    "within_bin_covariance": binned.within_bin_covariance,
                    "three_term_residual": binned.three_term_residual,
                    "identity_residual": binned.identity_residual,
                    "n_bins_occupied": binned.n_bins_occupied,
                },
            }
        )
        return EXIT_OK
    print(f"spec {spec.name}: {spec.description}")
    print(f"population BS = {truth.brier:.12f}, REL = {truth.reliability:.6e}, "
          f"RES = {truth.resolution:.12f}, UNC = {truth.uncertainty:.12f}")
    print()
    print(f"exact Murphy decomposition, forecasts rounded to {args.round_digits} digits")
    print(exact.report())
    print()
    print("binned decomposition on the unrounded forecasts")
    print(binned.report())
    print()
    ref = np.full(s.n_samples, float(s.outcomes.mean()))
    print(f"Brier score        : {brier_score(s.forecasts, s.outcomes):.12f}")
    print(f"log score (nats)   : {log_score(s.forecasts, s.outcomes):.12f}")
    print(
        "Brier skill vs sample base rate : "
        f"{brier_skill_score(s.forecasts, s.outcomes, ref):+.12f}"
    )
    print(
        "log skill vs sample base rate   : "
        f"{log_skill_score(s.forecasts, s.outcomes, ref):+.12f}"
    )
    return EXIT_OK


def _cmd_ece(args: argparse.Namespace) -> int:
    spec, s = _spec_sample(args)
    truth = analytic_truth(spec)
    result = debiased_ece(
        s.forecasts,
        s.outcomes,
        n_bins=args.bins,
        strategy=args.strategy,
        n_replicates=args.replicates,
        seed=args.seed + 1,
    )
    if args.json:
        payload = {k: v for k, v in asdict(result).items() if k != "null_values"}
        payload["spec"] = spec.name
        payload["population_ece"] = truth.ece
        payload["population_ece_abserr"] = truth.ece_abserr
        _print_json(payload)
        return EXIT_OK
    print(f"spec {spec.name}: {spec.description}")
    print(f"population ECE            : {truth.ece:.9f} (quad abserr {truth.ece_abserr:.1e})")
    print(result.report())
    print(f"  raw minus population    : {result.raw - truth.ece:+.9f}")
    print(f"  debiased minus pop.     : {result.debiased - truth.ece:+.9f}")
    return EXIT_OK


def _cmd_ece_bias(args: argparse.Namespace) -> int:
    spec = get_spec(args.spec)
    curve = ece_bias_curve(
        spec,
        n_bins_grid=args.bins_grid,
        n_samples_grid=args.samples_grid,
        strategy=args.strategy,
        n_replicates=args.replicates,
        seed=args.seed,
        debias_replicates=args.debias_replicates,
    )
    if args.json:
        _print_json(
            {
                "spec": curve.spec_name,
                "strategy": curve.strategy,
                "rows": [asdict(r) for r in curve.rows],
            }
        )
        return EXIT_OK
    print(f"spec {spec.name}: {spec.description}")
    print(f"strategy {curve.strategy}, {args.replicates} replicates per cell")
    print(curve.table())
    try:
        slope, intercept, r2 = curve.power_law_fit()
        print()
        print(
            f"power-law fit  bias ~ exp({intercept:.6f}) (B/n)^{slope:.6f}, "
            f"R^2 = {r2:.6f}  [theory: exponent 0.5]"
        )
    except ValueError as exc:
        print()
        print(f"power-law fit not available: {exc}")
    return EXIT_OK


def _cmd_reliability(args: argparse.Namespace) -> int:
    spec, s = _spec_sample(args)
    curve = bootstrap_reliability(
        s.forecasts,
        s.outcomes,
        n_bins=args.bins,
        strategy=args.strategy,
        n_bootstrap=args.bootstrap,
        level=args.level,
        seed=args.seed + 2,
    )
    if args.json:
        _print_json(
            {
                "spec": spec.name,
                "n_samples": curve.n_samples,
                "n_bins": curve.n_bins,
                "strategy": curve.strategy,
                "level": curve.level,
                "n_bootstrap": curve.n_bootstrap,
                "bin_index": curve.bin_index,
                "counts": curve.counts,
                "mean_forecast": curve.mean_forecast,
                "observed_frequency": curve.observed_frequency,
                "lower": curve.lower,
                "upper": curve.upper,
                "diagonal_excluded": curve.diagonal_excluded(),
            }
        )
        return EXIT_OK
    print(f"spec {spec.name}: {spec.description}")
    print(curve.table())
    excluded = int(np.sum(curve.diagonal_excluded()))
    print()
    print(
        f"{excluded} of {curve.n_occupied} occupied bins have a "
        f"{curve.level:.0%} band excluding the mean forecast."
    )
    print(
        "The level is pointwise, not simultaneous: see validation/VALIDATION.md "
        "section 5 for the measured simultaneous coverage."
    )
    return EXIT_OK


def _cmd_recalibrate(args: argparse.Namespace) -> int:
    spec, s = _spec_sample(args)
    audit = recalibration_audit(
        s.forecasts,
        s.outcomes,
        methods=tuple(args.methods),
        test_fraction=args.test_fraction,
        n_bins=args.bins,
        strategy=args.strategy,
        n_bootstrap=args.bootstrap,
        level=args.level,
        seed=args.seed + 3,
        ensemble_size=args.ensemble,
    )
    improved = [r.method for r in audit.results if r.verdict == "improved"]
    if args.json:
        _print_json(
            {
                "spec": spec.name,
                "n_train": audit.n_train,
                "n_test": audit.n_test,
                "level": audit.level,
                "improved": improved,
                "results": [asdict(r) for r in audit.results],
            }
        )
    else:
        print(f"spec {spec.name}: {spec.description}")
        print(
            f"split {audit.n_train} train / {audit.n_test} test, "
            f"{audit.n_bootstrap} paired bootstrap replicates, level {audit.level:.2f}"
        )
        print(audit.table())
        print()
        if improved:
            print(f"improved over the raw baseline: {', '.join(improved)}")
        else:
            print("no learned recalibrator improved on the raw forecast at this level.")
        print(
            "dBrier is held-out BS minus the raw baseline's on the same samples; "
            "negative is better."
        )
    if args.require_improvement and not improved:
        print(
            "require-improvement: no method improved on the raw baseline", file=sys.stderr
        )
        return EXIT_NO_IMPROVEMENT
    return EXIT_OK


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser. Exposed so tests can inspect it."""
    parser = argparse.ArgumentParser(
        prog="python -m calibaudit",
        description=(
            "Audit a binary probabilistic forecast: decompose the Brier score, "
            "quantify the calibration estimator's own bias, and test whether "
            "recalibration helps. Research-grade; not flight-qualified, not "
            "certified, not approved for operational aerospace use."
        ),
    )
    parser.add_argument("--version", action="version", version=f"calibaudit {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    def add_sample_args(p: argparse.ArgumentParser, default_n: int) -> None:
        p.add_argument("--spec", default="overconfident", choices=SPEC_NAMES)
        p.add_argument("-n", "--n-samples", type=int, default=default_n)
        p.add_argument("--seed", type=int, default=56)
        p.add_argument("--json", action="store_true")

    p_specs = sub.add_parser("specs", help="list the shipped synthetic forecasters")
    p_specs.add_argument("--json", action="store_true")
    p_specs.set_defaults(func=_cmd_specs)

    p_dec = sub.add_parser("decompose", help="Murphy decomposition, exact and binned")
    add_sample_args(p_dec, 4000)
    p_dec.add_argument("--bins", type=int, default=10)
    p_dec.add_argument("--strategy", default="equal_width", choices=("equal_width", "equal_mass"))
    p_dec.add_argument(
        "--round-digits",
        type=int,
        default=2,
        help="digits to round forecasts to before the exact decomposition",
    )
    p_dec.set_defaults(func=_cmd_decompose)

    p_ece = sub.add_parser("ece", help="ECE with its binning bias estimated")
    add_sample_args(p_ece, 2000)
    p_ece.add_argument("--bins", type=int, default=15)
    p_ece.add_argument("--strategy", default="equal_mass", choices=("equal_width", "equal_mass"))
    p_ece.add_argument("--replicates", type=int, default=400)
    p_ece.set_defaults(func=_cmd_ece)

    p_bias = sub.add_parser("ece-bias", help="the ECE estimator's bias over a grid")
    p_bias.add_argument("--spec", default="calibrated", choices=SPEC_NAMES)
    p_bias.add_argument("--seed", type=int, default=56)
    p_bias.add_argument("--json", action="store_true")
    p_bias.add_argument("--bins-grid", type=int, nargs="+", default=[5, 10, 20, 50])
    p_bias.add_argument("--samples-grid", type=int, nargs="+", default=[200, 1000, 5000])
    p_bias.add_argument("--strategy", default="equal_width", choices=("equal_width", "equal_mass"))
    p_bias.add_argument("--replicates", type=int, default=100)
    p_bias.add_argument("--debias-replicates", type=int, default=0)
    p_bias.set_defaults(func=_cmd_ece_bias)

    p_rel = sub.add_parser("reliability", help="reliability table with bootstrap bands")
    add_sample_args(p_rel, 2000)
    p_rel.add_argument("--bins", type=int, default=10)
    p_rel.add_argument("--strategy", default="equal_width", choices=("equal_width", "equal_mass"))
    p_rel.add_argument("--bootstrap", type=int, default=500)
    p_rel.add_argument("--level", type=float, default=0.9)
    p_rel.set_defaults(func=_cmd_reliability)

    p_rec = sub.add_parser("recalibrate", help="held-out recalibration audit")
    add_sample_args(p_rec, 2000)
    p_rec.add_argument("--bins", type=int, default=10)
    p_rec.add_argument("--strategy", default="equal_mass", choices=("equal_width", "equal_mass"))
    p_rec.add_argument("--methods", nargs="+", default=list(METHOD_NAMES), choices=METHOD_NAMES)
    p_rec.add_argument("--test-fraction", type=float, default=0.5)
    p_rec.add_argument("--bootstrap", type=int, default=400)
    p_rec.add_argument("--level", type=float, default=0.9)
    p_rec.add_argument("--ensemble", type=int, default=0)
    p_rec.add_argument(
        "--require-improvement",
        action="store_true",
        help=f"exit {EXIT_NO_IMPROVEMENT} if no learned method beats the raw baseline",
    )
    p_rec.set_defaults(func=_cmd_recalibrate)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Entry point. Returns the process exit status."""
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except (ValueError, KeyError, TypeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_USAGE


if __name__ == "__main__":
    sys.exit(main())
