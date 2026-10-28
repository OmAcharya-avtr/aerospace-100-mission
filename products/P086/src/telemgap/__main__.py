"""Command line interface: ``python -m telemgap <command>``.

Six commands, each printing one section of what the library computes. Every
command that classifies prints the determinability verdict first, because that
is the order the library returns it in.

``info``
    Environment, channel schema and the determinability verdict of a default
    record.
``check``
    The integrity report: screen counts, verdict counts and the first anomalies.
``policy``
    The gap policy, including every refusal and the gate that refused it.
``refuse``
    The constructed long-outage case, which the policy must refuse.
``determinability``
    The seven limits, optionally with the validity flag or the raw words
    withheld, so that the change in the undecided count is visible.
``benchmark``
    The rules, the learned model and a majority-class guess on one held-out
    split, with the per-class verdict stated plainly.
"""

from __future__ import annotations

import argparse
import sys

from . import __version__, benchmark, classify, dataset, determinability, features, interpolate
from . import policy as policy_module
from . import report as report_module
from .learned import LearnedClassifier, MajorityClassifier

#: Reconstruction tolerances used by the ``policy`` and ``refuse`` commands,
#: in each channel's declared unit. Chosen so that the four channels exercise
#: four different outcomes; see README.md.
DEFAULT_TOLERANCES = {
    "battery_voltage": 0.05,
    "valve_current": 0.01,
    "gyro_rate": 0.02,
    "tank_pressure": 0.02,
}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="telemgap",
        description=(
            "Separate missing telemetry from corrupt telemetry from a reader that is "
            "wrong, and emit a refusable gap policy. Research grade; not flight-qualified, "
            "not certified, not approved for operational aerospace use."
        ),
    )
    parser.add_argument("--version", action="version", version=f"telemgap {__version__}")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("info", help="environment, schema and determinability verdict")

    check = subparsers.add_parser("check", help="integrity report over a generated record")
    check.add_argument("--samples", type=int, default=4000, help="frames to generate")
    check.add_argument("--seed", type=int, default=dataset.TEST_SEED, help="generator seed")
    check.add_argument(
        "--mode",
        choices=classify.MODES,
        default="strengthened",
        help="specified runs only the four screens the product specification names",
    )

    gap_policy = subparsers.add_parser("policy", help="gap policy with every refusal")
    gap_policy.add_argument("--samples", type=int, default=4000)
    gap_policy.add_argument("--seed", type=int, default=dataset.TEST_SEED)
    gap_policy.add_argument(
        "--max-gap-slots", type=float, default=policy_module.DEFAULT_MAX_GAP_SLOTS
    )
    gap_policy.add_argument(
        "--no-resolvability-gate",
        action="store_true",
        help="disable the curvature resolvability gate, which is how its value was measured",
    )

    refuse = subparsers.add_parser("refuse", help="the constructed long outage, which is refused")
    refuse.add_argument("--outage-frames", type=int, default=21_600)

    limits = subparsers.add_parser("determinability", help="the seven limits")
    limits.add_argument("--samples", type=int, default=2000)
    limits.add_argument(
        "--withhold",
        choices=("none", "validity", "raw"),
        default="none",
        help="withhold the validity flag or the raw words and show what becomes undecidable",
    )

    bench = subparsers.add_parser("benchmark", help="rules, learned model, majority-class guess")
    bench.add_argument("--samples", type=int, default=6000)

    return parser


def _generated(samples: int, seed: int) -> dataset.GeneratedRecord:
    return dataset.generate(samples=samples, seed=seed)


def command_info() -> int:
    """Print the environment, the channel schema and a determinability verdict."""
    import numpy
    import sklearn

    generated = _generated(1200, dataset.TEST_SEED)
    record = generated.record
    print(f"telemgap {__version__}")
    print(f"python       {sys.version.split()[0]}")
    print(f"numpy        {numpy.__version__}")
    print(f"scikit-learn {sklearn.__version__}")
    print()
    print("channel schema of the generated records:")
    print()
    rows = [
        [
            channel.name,
            channel.unit,
            f"{channel.low:g}",
            f"{channel.high:g}",
            f"{channel.scale:g}",
            f"{channel.offset:g}",
            str(channel.word_bits),
            "yes" if channel.signed else "no",
            f"{channel.max_slew:.6g}",
            f"{channel.max_curvature:.6g}" if channel.max_curvature else "not declared",
            f"{channel.noise_sigma:.6g}" if channel.noise_sigma else "not declared",
        ]
        for channel in record.schema.channels
    ]
    print(
        report_module.format_table(
            [
                "channel",
                "unit",
                "low",
                "high",
                "scale",
                "offset",
                "bits",
                "signed",
                "max_slew",
                "max_curvature",
                "noise_sigma",
            ],
            rows,
        )
    )
    print()
    print(report_module.render_determinability(determinability.assess(record)))
    print()
    print(
        "Research grade. Not flight-qualified, not certified, not approved for "
        "operational aerospace use."
    )
    return 0


def command_check(samples: int, seed: int, mode: str) -> int:
    """Print the determinability verdict and then the integrity report."""
    generated = _generated(samples, seed)
    report = classify.classify(generated.record, mode=mode)
    print(report_module.render_determinability(report.determinability))
    print()
    print(report_module.render_integrity(report))
    print()
    print("ground truth of the generated record, for comparison:")
    print()
    print(
        report_module.format_table(
            ["quantity", "value"],
            [[key, str(value)] for key, value in generated.truth_counts().items()],
        )
    )
    print()
    print("A clean integrity report does not mean the telemetry is correct.")
    return 0


def command_policy(
    samples: int, seed: int, max_gap_slots: float, no_resolvability_gate: bool
) -> int:
    """Print the gap policy and the measured reconstruction error beside it."""
    generated = dataset.generate(samples=samples, seed=seed, n_clock_resets=0)
    record = generated.record
    report = classify.classify(record)
    plan = policy_module.plan(
        record,
        report,
        DEFAULT_TOLERANCES,
        max_gap_slots=max_gap_slots,
        resolvability_threshold=None if no_resolvability_gate else
        policy_module.DEFAULT_RESOLVABILITY_THRESHOLD,
    )
    print(report_module.render_policy(plan))
    reconstruction = interpolate.reconstruct(record, report.findings.gaps, plan)
    rows = []
    for column, channel in enumerate(record.schema.channels):
        error, count = interpolate.measured_error(
            reconstruction, generated.truth_times, generated.truth_values[:, column], column
        )
        permitted = [
            decision.predicted_error
            for decision in plan.decisions
            if decision.channel == column and not decision.refused
        ]
        worst = max(permitted) if permitted else float("nan")
        rows.append(
            [
                channel.name,
                str(count),
                f"{error:.6g}",
                f"{worst:.6g}",
                f"{DEFAULT_TOLERANCES[channel.name]:g}",
                "ok" if not count or error <= worst else "EXCEEDED",
            ]
        )
    print()
    print("realised reconstruction error against the noiseless truth:")
    print()
    print(
        report_module.format_table(
            ["channel", "slots", "max |error|", "worst predicted", "tolerance", "verdict"], rows
        )
    )
    return 0


def command_refuse(outage_frames: int) -> int:
    """Print the long-outage case and assert in output that it was refused."""
    generated = dataset.long_outage_record(outage_frames=outage_frames)
    record = generated.record
    report = classify.classify(record)
    plan = policy_module.plan(record, report, DEFAULT_TOLERANCES)
    print(f"constructed outage of {outage_frames} cadence intervals")
    print(f"gaps found              : {len(report.findings.gaps)}")
    print(f"frames the gaps imply   : {report.missing_frame_total}")
    print(f"policy decisions        : {len(plan.decisions)}")
    print(f"refusals                : {len(plan.refusals)}")
    print(f"permissions             : {len(plan.permissions)}")
    print()
    for decision in plan.refusals:
        print(f"{decision.channel_name}: {decision.reason}")
    print()
    if plan.permissions:
        print("REFUSAL CHECK FAILED: the policy permitted an interpolation across this outage")
        return 1
    print("REFUSAL CHECK PASSED: every channel was refused across this outage")
    return 0


def command_determinability(samples: int, withhold: str) -> int:
    """Print the limits, and the undecided count, with a field withheld."""
    generated = _generated(samples, dataset.TEST_SEED)
    record = generated.record
    if withhold == "validity":
        record = record.without_validity()
    elif withhold == "raw":
        record = record.without_raw_words()
    report = classify.classify(record)
    print(f"withheld: {withhold}")
    print()
    print(report_module.render_determinability(report.determinability))
    print()
    print(
        report_module.format_table(
            ["verdict", "count"], [[key, str(value)] for key, value in report.counts.items()]
        )
    )
    return 0


def command_benchmark(samples: int) -> int:
    """Train on one split, score rules, learned model and majority on another."""
    train = dataset.training_record(samples=samples)
    test = dataset.test_record(samples=samples)
    train_features, _ = features.build(train.record)
    train_labels = features.truth_labels(train.value_truth)
    test_features, _ = features.build(test.record)
    test_labels = features.truth_labels(test.value_truth)

    result = benchmark.BenchmarkResult()
    for mode, key in (("specified", "rules_specified"), ("strengthened", "rules_strengthened")):
        report = classify.classify(test.record, mode=mode)
        predicted = benchmark.rule_labels(test.record, report).reshape(-1)
        result.add(f"all_{key}", benchmark.score(test_labels, predicted, f"all/{key}"))
    model = LearnedClassifier().fit(train_features, train_labels)
    prediction = model.predict(test_features)
    result.add(
        "all_learned", benchmark.score(test_labels, prediction.labels, "all/learned")
    )
    majority = MajorityClassifier().fit(train_labels)
    result.add(
        "all_majority",
        benchmark.score(test_labels, majority.predict(test_features).labels, "all/majority"),
    )
    for key in ("all_rules_specified", "all_rules_strengthened", "all_learned", "all_majority"):
        print(report_module.render_scores(result.tables[key]))
        print()
    print("baseline versus learned, per class and metric:")
    print()
    for line in result.verdict_lines():
        print(f"  {line}")
    print()
    print("This model is not certified for operational flight use.")
    return 0


def main(argv: list[str] | None = None) -> int:
    """Entry point. Returns a process exit status."""
    arguments = _parser().parse_args(argv)
    if arguments.command == "info":
        return command_info()
    if arguments.command == "check":
        return command_check(arguments.samples, arguments.seed, arguments.mode)
    if arguments.command == "policy":
        return command_policy(
            arguments.samples,
            arguments.seed,
            arguments.max_gap_slots,
            arguments.no_resolvability_gate,
        )
    if arguments.command == "refuse":
        return command_refuse(arguments.outage_frames)
    if arguments.command == "determinability":
        return command_determinability(arguments.samples, arguments.withhold)
    return command_benchmark(arguments.samples)


if __name__ == "__main__":
    raise SystemExit(main())
