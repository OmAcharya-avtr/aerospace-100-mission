"""Command line interface: ``python -m hilverdict <command>``.

Every command prints tables and nothing else. No command writes a file, and no
command has a hidden default: the seed, profile, amplitude, tolerance and
clock behaviour are all arguments with declared values, printed back in the
header of every run so that an output pasted into a report carries its own
provenance.
"""

from __future__ import annotations

import argparse
import math
import sys

import numpy as np

from . import __version__
from .certificate import CertificateModel, certified_overrun_s, certify_run
from .clock import estimate_clock
from .features import FEATURE_NAMES, feature_matrix
from .learned import DEFAULT_SPEC, MissClassifier
from .plant import LoopModel
from .report import (
    format_table,
    render_certificates,
    render_clock,
    render_resimulations,
    render_run_certificate,
    render_scores,
    render_verdict,
)
from .resimulate import resimulate_run
from .scoring import score_decision, threshold_at_recall
from .synthetic import CLOCK_CLEAN, TIMING_PROFILES, ClockSpec, generate_population, generate_run
from .verdict import run_verdict

TOLERANCE = 1.0


def _header(arguments: argparse.Namespace) -> str:
    fields = [f"hilverdict {__version__}", f"command {arguments.command}"]
    for key in (
        "seed",
        "profile",
        "amplitude_deg",
        "steps",
        "tolerance",
        "drift_ppm",
        "resets",
        "train_runs",
        "test_runs",
        "target_recall",
    ):
        value = getattr(arguments, key, None)
        if value is not None:
            fields.append(f"{key} {value}")
    return " | ".join(fields)


def _clock_spec(arguments: argparse.Namespace) -> ClockSpec:
    resets = tuple(
        float(piece) for piece in str(arguments.resets).split(",") if piece.strip()
    )
    return ClockSpec(
        drift_ppm=float(arguments.drift_ppm),
        jitter_s=CLOCK_CLEAN.jitter_s,
        tick_s=CLOCK_CLEAN.tick_s,
        reset_fractions=resets,
        name="cli",
    )


def _run(arguments: argparse.Namespace):
    return generate_run(
        seed=arguments.seed,
        profile=arguments.profile,
        amplitude_deg=arguments.amplitude_deg,
        steps=arguments.steps,
        clock=_clock_spec(arguments),
    )


def _certify(arguments: argparse.Namespace) -> int:
    log = _run(arguments)
    certificate = certify_run(log, arguments.tolerance)
    print(render_run_certificate(certificate))
    print()
    if certificate.certificates:
        print(render_certificates(certificate.certificates, limit=arguments.limit))
    else:
        print("no iteration in this run exceeded its deadline")
    return 0


def _resimulate(arguments: argparse.Namespace) -> int:
    log = _run(arguments)
    certificate = certify_run(log, arguments.tolerance)
    subset = None if arguments.all else tuple(
        c.event.index for c in certificate.undecided
    )
    results = resimulate_run(log, arguments.tolerance, subset=subset)
    if not results:
        print(
            "nothing to re-simulate: the certificate dismissed every miss in this run"
        )
        return 0
    print(render_resimulations(results))
    print()
    calls = sum(result.plant_calls for result in results)
    rows = [
        ["misses recorded", str(len(certificate.certificates))],
        ["dismissed in closed form", str(len(certificate.dismissed))],
        ["re-simulated here", str(len(results))],
        ["plant evaluations spent", str(calls)],
        [
            "plant evaluations if every miss were re-simulated",
            str(
                sum(
                    4 * log.model.sampling.substeps * max(c.event.steps_remaining, 0)
                    for c in certificate.certificates
                )
            ),
        ],
    ]
    print(format_table(["quantity", "value"], rows))
    return 0


def _clock(arguments: argparse.Namespace) -> int:
    log = _run(arguments)
    estimate = estimate_clock(log.host_time_s, log.device_time_s, arguments.allowance_ppm)
    print(render_clock(estimate))
    return 0


def _verdict(arguments: argparse.Namespace) -> int:
    log = _run(arguments)
    print(render_verdict(run_verdict(log, arguments.tolerance)))
    return 0


def _threshold(arguments: argparse.Namespace) -> int:
    model = LoopModel()
    constants = CertificateModel.build(model, arguments.steps)
    rows = []
    for degrees in (0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0):
        step = math.radians(degrees)
        threshold = certified_overrun_s(
            constants, step, arguments.steps, arguments.tolerance
        )
        rows.append(
            [
                f"{degrees:g}",
                f"{threshold * 1e3:.6f}",
                f"{threshold / model.sampling.period_s:.6f}",
            ]
        )
    print(
        format_table(
            ["input step [deg]", "certified overrun [ms]", "in periods"], rows
        )
    )
    print()
    print("equation (14), at the full horizon. Any miss shorter than the value in")
    print("column 2 cannot have moved the trajectory past the declared tolerance,")
    print("for an input step no larger than column 1. That is a proof about the")
    print("declared linearisation, not an estimate.")
    print()
    rows = [
        ["mu_S [1/s]", f"{constants.mu_s:.6f}"],
        ["|| S B ||_2 [1/(s rad)]", f"{constants.input_gain:.6f}"],
        ["c_free, eq (11)", f"{constants.free_factor:.6f}"],
        ["gamma_sup, eq (12)", f"{constants.gamma_sup(arguments.steps):.6f}"],
        ["metric anisotropy", f"{model.metric.anisotropy:.6f}"],
        ["closed-loop spectral radius", f"{model.spectral_radius():.6f}"],
    ]
    print(format_table(["model constant", "value"], rows))
    return 0


def _learned(arguments: argparse.Namespace) -> int:
    train = generate_population(
        tuple(range(1000, 1000 + arguments.train_runs)), steps=arguments.steps
    )
    test = generate_population(
        tuple(range(2000, 2000 + arguments.test_runs)), steps=arguments.steps
    )

    def collect(runs):
        features = []
        labels = []
        bounds = []
        for log in runs:
            certificate = certify_run(log, arguments.tolerance)
            if certificate.refusals:
                continue
            matrix, events = feature_matrix(log)
            if not events:
                continue
            results = resimulate_run(log, arguments.tolerance)
            features.append(matrix)
            labels.append(np.array([r.consequential for r in results], dtype=bool))
            bounds.append(np.array([c.bound for c in certificate.certificates]))
        return (
            np.vstack(features),
            np.concatenate(labels),
            np.concatenate(bounds),
        )

    train_features, train_labels, _ = collect(train)
    test_features, test_labels, test_bounds = collect(test)
    classifier = MissClassifier(spec=DEFAULT_SPEC).fit(train_features, train_labels)
    probability = classifier.predict_proba(test_features)
    certificate_flags = test_bounds >= arguments.tolerance
    scores = [
        score_decision(
            "analytic certificate",
            certificate_flags,
            test_labels,
            sound=True,
            note="baseline 1, equation (13)",
        )
    ]
    try:
        threshold, _, _ = threshold_at_recall(
            probability, test_labels, arguments.target_recall
        )
        classifier.set_threshold(threshold)
        scores.append(
            score_decision(
                f"learned, recall>={arguments.target_recall:g}",
                probability >= threshold,
                test_labels,
                sound=False,
            )
        )
    except ValueError as error:
        print(f"[not available] learned at the target recall: {error}")
    matched = np.quantile(probability, 1.0 - float(np.mean(certificate_flags)))
    scores.append(
        score_decision(
            "learned, flag rate matched",
            probability >= matched,
            test_labels,
            sound=False,
            note="threshold set so it flags as many misses as the certificate",
        )
    )
    print(render_scores(scores))
    print()
    reliability = classifier.reliability(test_features, test_labels)
    print(f"Brier score on held-out misses: {reliability.brier:.6f}")
    print(f"worst calibration gap over bins: {reliability.worst_gap:.6f}")
    print()
    importance = classifier.feature_importance
    rows = [
        [name, f"{importance[name]:.6f}"]
        for name in sorted(FEATURE_NAMES, key=lambda key: -importance[key])
    ]
    print(format_table(["feature", "impurity importance"], rows))
    print()
    print("This model is not certified for operational flight use.")
    return 0


def build_parser() -> argparse.ArgumentParser:
    """The argument parser. Exposed so the tests can exercise it directly."""
    parser = argparse.ArgumentParser(
        prog="python -m hilverdict",
        description=(
            "Decide whether a hardware-in-the-loop run was timing-faithful enough "
            "for its result to mean anything, and separate the deadline misses "
            "that changed the trajectory from the ones that provably did not."
        ),
    )
    parser.add_argument("--version", action="version", version=f"hilverdict {__version__}")
    subparsers = parser.add_subparsers(dest="command", required=True)

    def add_run_arguments(target: argparse.ArgumentParser) -> None:
        target.add_argument("--seed", type=int, default=101)
        target.add_argument(
            "--profile", choices=sorted(TIMING_PROFILES), default="loaded"
        )
        target.add_argument("--amplitude-deg", type=float, default=6.0)
        target.add_argument("--steps", type=int, default=200)
        target.add_argument("--tolerance", type=float, default=TOLERANCE)
        target.add_argument("--drift-ppm", type=float, default=0.0)
        target.add_argument(
            "--resets",
            type=str,
            default="",
            help="comma-separated fractions of the run at which the device clock resets",
        )

    certify = subparsers.add_parser(
        "certify", help="baseline 1: the closed-form certificate for every miss"
    )
    add_run_arguments(certify)
    certify.add_argument("--limit", type=int, default=20)
    certify.set_defaults(handler=_certify)

    resimulate = subparsers.add_parser(
        "resimulate", help="baseline 2: re-simulate the residual undecided misses"
    )
    add_run_arguments(resimulate)
    resimulate.add_argument(
        "--all", action="store_true", help="re-simulate every miss, not only the residual"
    )
    resimulate.set_defaults(handler=_resimulate)

    clock = subparsers.add_parser(
        "clock", help="host-to-device drift and reset detection"
    )
    add_run_arguments(clock)
    clock.add_argument("--allowance-ppm", type=float, default=100.0)
    clock.set_defaults(handler=_clock)

    verdict = subparsers.add_parser("verdict", help="the run-level timing verdict")
    add_run_arguments(verdict)
    verdict.set_defaults(handler=_verdict)

    threshold = subparsers.add_parser(
        "threshold", help="equation (14): the certified overrun threshold"
    )
    threshold.add_argument("--steps", type=int, default=200)
    threshold.add_argument("--tolerance", type=float, default=TOLERANCE)
    threshold.set_defaults(handler=_threshold)

    learned = subparsers.add_parser(
        "learned", help="the learned classifier, scored against baseline 1"
    )
    learned.add_argument("--train-runs", type=int, default=18)
    learned.add_argument("--test-runs", type=int, default=9)
    learned.add_argument("--steps", type=int, default=200)
    learned.add_argument("--tolerance", type=float, default=TOLERANCE)
    learned.add_argument("--target-recall", type=float, default=1.0)
    learned.set_defaults(handler=_learned)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Parse ``argv`` and run the requested command."""
    parser = build_parser()
    arguments = parser.parse_args(argv)
    if getattr(arguments, "tolerance", 1.0) <= 0.0:
        parser.error(
            f"--tolerance must be strictly positive, got {arguments.tolerance}; the "
            "divergence metric is scaled so that the declared tolerance is 1"
        )
    if getattr(arguments, "steps", 200) < 2:
        parser.error(f"--steps must be at least 2, got {arguments.steps}")
    if not 0.0 < getattr(arguments, "target_recall", 1.0) <= 1.0:
        parser.error(
            f"--target-recall must lie in (0, 1], got {arguments.target_recall}"
        )
    print(_header(arguments))
    print()
    return int(arguments.handler(arguments))


if __name__ == "__main__":
    sys.exit(main())
