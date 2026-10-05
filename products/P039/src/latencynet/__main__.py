"""Command-line interface: ``python -m latencynet``.

Subcommands
-----------
``analytic``
    Evaluate the sum-of-stages model on stage means and standard deviations
    given on the command line, in microseconds, and print the end-to-end
    moments, the Fenton-Wilkinson quantiles and the GUM prediction interval.

``sample``
    Draw an injected pipeline and print the empirical moments and quantiles
    next to the analytic prediction, so the approximation error is visible.

``tail``
    Print the order-statistic standard error of a sample quantile for a given
    lognormal, tail probability and sample count, and the smallest sample size
    at which that probability is interior to the sample.

``compare``
    Run the three-way held-out comparison in one dependence regime and print
    the table. Defaults are sized for a few seconds on one core.

Every latency on the command line is in microseconds, because that is the
natural unit for an embedded pipeline; everything inside the package is in
seconds.
"""

from __future__ import annotations

import argparse
import sys

import numpy as np

from . import __version__
from .analytic import SumOfStagesModel
from .compare import compare_models, format_comparison
from .dataset import build_dataset
from .features import summarise_probe_trace
from .pipeline import make_lognormal_pipeline, sample_stage_latencies
from .tails import lognormal_quantile, lognormal_quantile_se, quantile, quantile_min_samples
from .units import s_to_us, us_to_s

_PROBABILITIES = (0.5, 0.9, 0.99, 0.999)


def _stage_arrays(means_us: list[float], stds_us: list[float]) -> tuple[np.ndarray, np.ndarray]:
    if len(means_us) != len(stds_us):
        raise SystemExit(
            f"--mean-us and --std-us must have the same number of values, "
            f"got {len(means_us)} and {len(stds_us)}"
        )
    means = np.array([us_to_s(v) for v in means_us], dtype=float)
    stds = np.array([us_to_s(v) for v in stds_us], dtype=float)
    return means, stds


def _cmd_analytic(args: argparse.Namespace) -> int:
    means, stds = _stage_arrays(args.mean_us, args.std_us)
    model = SumOfStagesModel(assume_independent=not args.use_covariance)
    if args.use_covariance:
        spec = make_lognormal_pipeline(means, stds, latent_rho=args.rho)
        cov = spec.injected_covariance()
    else:
        cov = None
    pred = model.predict(means, stds, _PROBABILITIES, cov)
    print(f"latencynet {__version__} -- analytic sum-of-stages ({model.name})")
    print(f"stages: {means.size}   latent rho: {args.rho if args.use_covariance else 0.0}")
    for i, (m, s) in enumerate(zip(means, stds, strict=True)):
        print(f"  stage {i}: mean {s_to_us(m):10.3f} us   sd {s_to_us(s):10.3f} us")
    print(f"end-to-end mean   {s_to_us(pred.mean_s):10.3f} us   (eq. 1, exact)")
    print(f"end-to-end sd     {s_to_us(pred.std_s):10.3f} us   (eq. 2)")
    m4 = 3.0 * stds**4
    for p in _PROBABILITIES:
        lo, pt, hi = model.predict_quantile_interval(
            means, stds, m4, args.n_probe, p, level=args.level, stage_cov_s2=cov
        )
        print(
            f"  q({p:<6}) {s_to_us(pt):10.3f} us   "
            f"{int(args.level * 100)} % PI [{s_to_us(lo):.3f}, {s_to_us(hi):.3f}] us"
        )
    print("Fenton-Wilkinson quantiles are an approximation; the interval covers")
    print(f"probe sampling uncertainty at n_probe={args.n_probe} only.")
    return 0


def _cmd_sample(args: argparse.Namespace) -> int:
    means, stds = _stage_arrays(args.mean_us, args.std_us)
    spec = make_lognormal_pipeline(means, stds, latent_rho=args.rho)
    trace = sample_stage_latencies(spec, args.n, args.seed)
    total = trace.sum(axis=1)
    probe = summarise_probe_trace(trace)
    pred = SumOfStagesModel().predict(
        np.asarray(probe.stage_mean_s), np.asarray(probe.stage_std_s), _PROBABILITIES
    )
    print(f"latencynet {__version__} -- injected sample, n={args.n}, seed={args.seed}")
    print(f"injected mean {s_to_us(spec.injected_mean_s()):10.3f} us   "
          f"injected sd {s_to_us(spec.injected_std_s()):10.3f} us   rho={spec.latent_rho}")
    print(f"sampled  mean {s_to_us(total.mean()):10.3f} us   "
          f"sampled  sd {s_to_us(total.std(ddof=1)):10.3f} us")
    print(f"{'p':>8}{'sampled us':>14}{'analytic us':>14}{'rel diff':>12}")
    for p in _PROBABILITIES:
        emp = quantile(total, p, method="linear")
        ana = pred.quantile_s[p]
        print(f"{p:>8}{s_to_us(emp):>14.3f}{s_to_us(ana):>14.3f}{(ana - emp) / emp:>12.5f}")
    return 0


def _cmd_tail(args: argparse.Namespace) -> int:
    q = lognormal_quantile(args.mu, args.sigma, args.p)
    se = lognormal_quantile_se(args.mu, args.sigma, args.p, args.n)
    print(f"latencynet {__version__} -- order-statistic tail uncertainty")
    print(f"lognormal mu={args.mu} sigma={args.sigma}   p={args.p}   n={args.n}")
    print(f"exact quantile          {s_to_us(q):12.4f} us")
    print(f"analytic SE             {s_to_us(se):12.4f} us  ({se / q * 100:.3f} % of the quantile)")
    print(f"SE scales as n^-0.5; smallest interior n for p={args.p}: "
          f"{quantile_min_samples(args.p)}")
    return 0


def _cmd_compare(args: argparse.Namespace) -> int:
    dataset = build_dataset(
        args.regime,
        n_train=args.n_train,
        n_calibration=args.n_calibration,
        n_test=args.n_test,
        seed=args.seed,
        n_probe=args.n_probe,
        n_reference=args.n_reference,
        n_reference_test=args.n_reference,
    )
    print(f"latencynet {__version__} -- held-out three-way comparison")
    print(format_comparison(compare_models(dataset, args.p, level=args.level)))
    return 0


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser for ``python -m latencynet``."""
    parser = argparse.ArgumentParser(
        prog="python -m latencynet",
        description=(
            "End-to-end latency prediction for staged pipelines: analytic "
            "sum-of-stages model, linear-regression baseline and a learned "
            "predictor, all with prediction intervals. Latencies on the "
            "command line are in microseconds. Research-grade; not "
            "flight-qualified, not certified."
        ),
    )
    parser.add_argument("--version", action="version", version=f"latencynet {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    a = sub.add_parser("analytic", help="analytic sum-of-stages prediction from stage moments")
    a.add_argument("--mean-us", type=float, nargs="+", required=True, help="stage means, us")
    a.add_argument("--std-us", type=float, nargs="+", required=True, help="stage sds, us")
    a.add_argument("--rho", type=float, default=0.0, help="latent stage equicorrelation")
    a.add_argument(
        "--use-covariance",
        action="store_true",
        help="use the covariance-aware variance (eq. 2 exact) instead of assuming independence",
    )
    a.add_argument("--n-probe", type=int, default=256, help="probe passes behind the moments")
    a.add_argument("--level", type=float, default=0.9, help="nominal interval coverage")
    a.set_defaults(func=_cmd_analytic)

    s = sub.add_parser(
        "sample", help="draw an injected pipeline and compare with the analytic model"
    )
    s.add_argument("--mean-us", type=float, nargs="+", required=True, help="stage means, us")
    s.add_argument("--std-us", type=float, nargs="+", required=True, help="stage sds, us")
    s.add_argument("--rho", type=float, default=0.0, help="latent stage equicorrelation")
    s.add_argument("--n", type=int, default=20000, help="number of passes to draw")
    s.add_argument("--seed", type=int, default=20260401, help="generator seed")
    s.set_defaults(func=_cmd_sample)

    t = sub.add_parser("tail", help="order-statistic standard error of a tail quantile")
    t.add_argument("--mu", type=float, default=-8.5, help="lognormal mu, log-seconds")
    t.add_argument("--sigma", type=float, default=0.4, help="lognormal sigma")
    t.add_argument("--p", type=float, default=0.99, help="tail probability")
    t.add_argument("--n", type=int, default=10000, help="sample count")
    t.set_defaults(func=_cmd_tail)

    c = sub.add_parser("compare", help="three-way held-out comparison in one dependence regime")
    c.add_argument("--regime", choices=("independent", "correlated"), default="correlated")
    c.add_argument("--p", type=float, default=0.99, help="tail probability to predict")
    c.add_argument("--level", type=float, default=0.9, help="nominal interval coverage")
    c.add_argument("--n-train", type=int, default=120)
    c.add_argument("--n-calibration", type=int, default=40)
    c.add_argument("--n-test", type=int, default=60)
    c.add_argument("--n-probe", type=int, default=256)
    c.add_argument("--n-reference", type=int, default=20000)
    c.add_argument("--seed", type=int, default=20260402)
    c.set_defaults(func=_cmd_compare)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Entry point. Returns a process exit status."""
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except (ValueError, TypeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
