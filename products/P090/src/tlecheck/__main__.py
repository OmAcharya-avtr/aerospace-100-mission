"""Command line for ``tlecheck``. The only module in the package that prints.

Subcommands
-----------
``info``
    Versions, declared defaults and the constants every number rests on.
``check``
    Screen one element set and print the determinability report and the
    verdict. Reads two lines from a file with ``--file``, or screens a
    constructed element set from the generator with ``--corruption``.
``determinability``
    Print only the determinability report.
``consequence``
    Measure the consequence of each corruption family by propagating with
    ``sgp4``, and print the kilometres the class labels rest on.
``screen``
    Screen a generated split and print the rule baseline's per-class scores
    against the measured consequence classes.
``benchmark``
    The rule baseline, the three learned configurations and the majority-class
    baseline, scored on the same held-out split.
``divergence``
    Two sources for one object, compared in kilometres, with the statement
    that which one is wrong is not determinable.

Every subcommand takes ``--objects`` where a generated corpus is involved, so
the whole command line can be exercised quickly in a test.
"""

from __future__ import annotations

import argparse
import math
import sys
from collections.abc import Sequence
from pathlib import Path

from . import __version__, benchmark, dataset, determinability, divergence, report
from .consequence import CLASSES, DEFAULT_SILENT_THRESHOLD_KM, measure
from .elements import parse_tle
from .learned import CONFIGURATIONS, LearnedClassifier, MajorityClassifier
from .verdict import screen

DEFAULT_OBJECTS = 420
DEFAULT_HISTORY = 12


def _read_pair(path: Path) -> tuple[str, str]:
    """Read two element set lines from a text file.

    Raises
    ------
    ValueError
        If the file does not contain at least two non-empty lines.
    """
    lines = [line for line in path.read_text().splitlines() if line.strip()]
    if len(lines) < 2:
        raise ValueError(
            f"{path.name} holds {len(lines)} non-empty lines; two element set lines "
            "are needed. A three-line record with a name line is accepted by taking "
            "the last two lines."
        )
    return lines[-2], lines[-1]


def _pick(records: list[dataset.Record], corruption: str) -> dataset.Record:
    """First record of the requested corruption family.

    Raises
    ------
    ValueError
        If no record of that family is present.
    """
    for record in records:
        if record.corruption == corruption:
            return record
    raise ValueError(
        f"no record with corruption {corruption!r} in this sample; increase "
        "--objects or choose another family"
    )


def _command_info(arguments: argparse.Namespace) -> int:
    from importlib.metadata import PackageNotFoundError, version

    from .bounds import (
        EARTH_RADIUS_KM,
        MU_KM3_S2,
        BoundsPolicy,
        mean_motion_ceiling_rev_per_day,
    )
    from .history import MAD_CONSISTENCY, QUANTISATION

    def installed(name: str) -> str:
        try:
            return version(name)
        except PackageNotFoundError:
            return "not installed"

    policy = BoundsPolicy()
    lines = [
        f"tlecheck {__version__}",
        "",
        "versions",
        f"  python               {sys.version.split()[0]}",
        f"  numpy                {installed('numpy')}",
        f"  scipy                {installed('scipy')}",
        f"  scikit-learn         {installed('scikit-learn')}",
        f"  sgp4                 {installed('sgp4')}",
        f"  matplotlib           {installed('matplotlib')}",
        f"  astropy              {installed('astropy')}  (named in the alternatives "
        "table; no number here is compared against it)",
        f"  skyfield             {installed('skyfield')}  (named in the alternatives "
        "table; no number here is compared against it)",
        "",
        "constants (WGS-72, as the SGP4 reference implementation carries them)",
        f"  mu                   {MU_KM3_S2} km^3 s^-2",
        f"  earth radius         {EARTH_RADIUS_KM} km",
        f"  J2                   {dataset.J2}",
        f"  xke                  {dataset.XKE}",
        f"  mean motion ceiling  {mean_motion_ceiling_rev_per_day():.6f} rev/day"
        "  (equation 3)",
        f"  MAD consistency      {MAD_CONSISTENCY}  (equation 7)",
        "",
        "declared policy defaults, every one a choice and not a law",
        f"  silent threshold     {DEFAULT_SILENT_THRESHOLD_KM} km",
        "  epoch requirement    7 days",
        "  history z threshold  4.0",
        "  continuity z thresh  4.0",
        f"  min perigee altitude {policy.min_perigee_altitude_km} km",
        f"  max eccentricity     {policy.max_eccentricity}",
        f"  max |bstar|          {policy.max_abs_bstar} inverse Earth radii",
        f"  min mean motion      {policy.min_mean_motion_rev_per_day} rev/day",
        "",
        "element field quantisation, one count of each text field",
    ]
    for name, step in sorted(QUANTISATION.items()):
        lines.append(f"  {name:24s} {step:g}")
    lines.append("")
    lines.append(
        "This package ships no element catalogue. The authoritative source of "
        "element sets is an operator's catalogue and it is not redistributable; "
        "every element set here is constructed from declared element values."
    )
    lines.append(
        "Research grade. Not flight-qualified, not certified, not approved for "
        "operational aerospace use."
    )
    print("\n".join(lines))
    return 0


def _command_check(arguments: argparse.Namespace) -> int:
    if arguments.file is not None:
        line1, line2 = _read_pair(Path(arguments.file))
        element_set = parse_tle(line1, line2, source="file")
        history: list = []
        others: list = []
        reference_jd = None
        from .epoch import epoch_julian_date

        reference_jd = epoch_julian_date(element_set) + arguments.use_offset_days
    else:
        records = dataset.split(
            arguments.split, objects=arguments.objects, history_length=DEFAULT_HISTORY
        )
        record = _pick(records, arguments.corruption)
        element_set = record.candidate
        history = list(record.history)
        others = list(record.other_sources)
        reference_jd = record.use_jd
        print(
            f"constructed element set: object {record.satnum}, family "
            f"{record.family}, injected corruption {record.corruption!r}"
        )
        print(element_set.line1)
        print(element_set.line2)
        print("")
    verdict = screen(
        element_set,
        history=history or None,
        other_sources=others or None,
        reference_jd=reference_jd,
        epoch_requirement_days=arguments.epoch_requirement_days,
        silent_threshold_km=arguments.silent_threshold_km,
    )
    print(report.verdict_text(verdict))
    return 0


def _command_determinability(arguments: argparse.Namespace) -> int:
    records = dataset.split(
        arguments.split, objects=arguments.objects, history_length=arguments.history
    )
    record = records[0]
    limits = determinability.report(
        record.candidate,
        list(record.history),
        sources=1 + len(record.other_sources),
        epoch_requirement_days=arguments.epoch_requirement_days,
        silent_threshold_km=arguments.silent_threshold_km,
    )
    print(
        f"object {record.satnum}, family {record.family}, "
        f"{len(record.history)} prior element sets, "
        f"{1 + len(record.other_sources)} source(s)"
    )
    print("")
    print(report.determinability_text(limits))
    return 0


def _command_consequence(arguments: argparse.Namespace) -> int:
    records = dataset.split(
        arguments.split, objects=arguments.objects, history_length=DEFAULT_HISTORY
    )
    print(
        f"measured consequence by injected corruption family, {len(records)} objects, "
        f"horizon {arguments.horizon_days:g} days, {arguments.samples} samples, "
        f"declared silent threshold {arguments.silent_threshold_km:g} km"
    )
    print("")
    header = f"{'injected corruption':28s} {'n':>4s}"
    for label in CLASSES:
        header += f" {label:>18s}"
    header += f" {'median max error km':>21s}"
    print(header)
    buckets: dict[str, list] = {}
    for record in records:
        consequence = measure(
            record.candidate,
            record.reference,
            record.use_jd,
            horizon_days=arguments.horizon_days,
            samples=arguments.samples,
            silent_threshold_km=arguments.silent_threshold_km,
        )
        buckets.setdefault(record.corruption, []).append(consequence)
    import numpy as np

    for family in dataset.CORRUPTION_FAMILIES:
        group = buckets.get(family)
        if not group:
            continue
        line = f"{family:28s} {len(group):>4d}"
        for label in CLASSES:
            line += f" {sum(1 for c in group if c.consequence_class == label):>18d}"
        finite = [c.max_error_km for c in group if math.isfinite(c.max_error_km)]
        line += f" {np.median(finite) if finite else float('nan'):>21.6f}"
        print(line)
    return 0


def _command_screen(arguments: argparse.Namespace) -> int:
    records = dataset.split(
        arguments.split, objects=arguments.objects, history_length=DEFAULT_HISTORY
    )
    prepared = benchmark.prepare(
        records,
        epoch_requirement_days=arguments.epoch_requirement_days,
        horizon_days=arguments.horizon_days,
        samples=arguments.samples,
        silent_threshold_km=arguments.silent_threshold_km,
    )
    scores = benchmark.per_class(prepared.measured_classes, prepared.rule_classes)
    print(
        report.class_table(
            scores,
            f"rule baseline against measured consequence, {len(records)} objects of "
            f"split {arguments.split!r}",
        )
    )
    print("")
    print(f"accuracy {benchmark.accuracy(prepared.measured_classes, prepared.rule_classes):.4f}")
    positive = prepared.measured_classes == "silently_wrong"
    print(
        report.operating_point_text(
            "rules at their own threshold",
            benchmark.point_operating_point(prepared.rule_classes, positive),
        )
    )
    print(
        report.operating_point_text(
            f"rules swept, at false-positive {arguments.target_fpr:g}",
            benchmark.recall_at_fpr(prepared.rule_scores, positive, arguments.target_fpr),
        )
    )
    area = benchmark.area_under_curve(prepared.rule_scores, positive)
    print(f"rule operating-characteristic area {area:.4f}")
    return 0


def _command_benchmark(arguments: argparse.Namespace) -> int:
    train = benchmark.prepare(
        dataset.split("train", objects=arguments.objects, history_length=DEFAULT_HISTORY),
        silent_threshold_km=arguments.silent_threshold_km,
    )
    test = benchmark.prepare(
        dataset.split("test", objects=arguments.objects, history_length=DEFAULT_HISTORY),
        silent_threshold_km=arguments.silent_threshold_km,
    )
    positive = test.measured_classes == "silently_wrong"
    print(
        f"train {len(train.records)} objects, test {len(test.records)} objects, "
        f"silent threshold {arguments.silent_threshold_km:g} km, "
        f"target false-positive rate {arguments.target_fpr:g}"
    )
    print("")
    print(
        report.operating_point_text(
            "rules at their own threshold",
            benchmark.point_operating_point(test.rule_classes, positive),
        )
    )
    print(
        report.operating_point_text(
            "rules swept",
            benchmark.recall_at_fpr(test.rule_scores, positive, arguments.target_fpr),
        )
    )
    print(
        f"{'rules':34s} area={benchmark.area_under_curve(test.rule_scores, positive):.4f}  "
        f"accuracy={benchmark.accuracy(test.measured_classes, test.rule_classes):.4f}"
    )
    majority = MajorityClassifier().fit(train.measured_classes)
    predicted = majority.predict(len(test.records))
    print(
        f"{'majority-class baseline':34s} area=0.5000 by construction  "
        f"accuracy={benchmark.accuracy(test.measured_classes, predicted.labels):.4f}  "
        f"always predicts {majority.majority!r}"
    )
    for name, configuration in CONFIGURATIONS.items():
        feature_configuration = str(configuration["feature_configuration"])
        model = LearnedClassifier(**configuration)  # type: ignore[arg-type]
        model.fit(train.matrices[feature_configuration], train.measured_classes)
        prediction = model.predict(test.matrices[feature_configuration])
        print(
            report.operating_point_text(
                f"learned {name}",
                benchmark.recall_at_fpr(
                    prediction.silent_probability, positive, arguments.target_fpr
                ),
            )
        )
        print(
            f"{'  area and accuracy':34s} "
            f"area={benchmark.area_under_curve(prediction.silent_probability, positive):.4f}  "
            f"accuracy={benchmark.accuracy(test.measured_classes, prediction.labels):.4f}  "
            f"mean margin={prediction.margin.mean():.4f}"
        )
    return 0


def _command_divergence(arguments: argparse.Namespace) -> int:
    records = dataset.split(
        arguments.split, objects=arguments.objects, history_length=DEFAULT_HISTORY
    )
    record = _pick(records, "cross_source_divergence")
    result = divergence.compare(
        record.candidate,
        record.other_sources[0],
        record.use_jd,
        horizon_days=arguments.horizon_days,
        samples=arguments.samples,
    )
    print(f"object {record.satnum}, family {record.family}")
    print(f"sources compared     : {result.sources[0]!r} and {result.sources[1]!r}")
    print(f"usable               : {result.usable}")
    if result.usable:
        print(f"max separation       : {result.max_separation_km:.6f} km")
        print(f"rms separation       : {result.rms_separation_km:.6f} km")
        print(f"separation at start  : {result.initial_separation_km:.6f} km")
        print(f"threshold            : {result.threshold_km:g} km (declared policy)")
        print(f"diverged             : {result.diverged}")
    else:
        print(f"reason               : {result.reason}")
    print(f"which source is right: {result.which_source_is_correct}")
    print(
        "Equation (12) is symmetric in its two arguments. Which source is wrong is "
        "not determinable from two element sets and a propagator; it needs an "
        "independent observation this package does not have."
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    """The argument parser, built separately so tests can introspect it."""
    parser = argparse.ArgumentParser(
        prog="tlecheck",
        description=(
            "Screen a two-line element set before propagating it, and report which of "
            "its problems the propagator will not tell you about."
        ),
    )
    parser.add_argument("--version", action="version", version=f"tlecheck {__version__}")
    subparsers = parser.add_subparsers(dest="command", required=True)

    def common(sub: argparse.ArgumentParser, objects: int = DEFAULT_OBJECTS) -> None:
        sub.add_argument("--objects", type=int, default=objects)
        sub.add_argument(
            "--split", default="test", choices=("train", "test", "calibration")
        )
        sub.add_argument("--horizon-days", type=float, default=1.0)
        sub.add_argument("--samples", type=int, default=25)
        sub.add_argument(
            "--silent-threshold-km", type=float, default=DEFAULT_SILENT_THRESHOLD_KM
        )
        sub.add_argument("--epoch-requirement-days", type=float, default=7.0)
        sub.add_argument("--target-fpr", type=float, default=0.05)

    info = subparsers.add_parser("info", help="versions, constants and declared defaults")
    info.set_defaults(handler=_command_info)

    check = subparsers.add_parser("check", help="screen one element set")
    common(check, objects=60)
    check.add_argument("--file", default=None, help="text file holding two element lines")
    check.add_argument("--corruption", default="sign_flip", choices=dataset.CORRUPTION_FAMILIES)
    check.add_argument("--use-offset-days", type=float, default=1.0)
    check.set_defaults(handler=_command_check)

    determinability_parser = subparsers.add_parser(
        "determinability", help="print only the determinability report"
    )
    common(determinability_parser, objects=12)
    determinability_parser.add_argument("--history", type=int, default=DEFAULT_HISTORY)
    determinability_parser.set_defaults(handler=_command_determinability)

    consequence = subparsers.add_parser(
        "consequence", help="measure consequence by corruption family with sgp4"
    )
    common(consequence, objects=120)
    consequence.set_defaults(handler=_command_consequence)

    screen_parser = subparsers.add_parser("screen", help="score the rule baseline")
    common(screen_parser, objects=120)
    screen_parser.set_defaults(handler=_command_screen)

    benchmark_parser = subparsers.add_parser(
        "benchmark", help="rules, learned configurations and majority baseline"
    )
    common(benchmark_parser, objects=180)
    benchmark_parser.set_defaults(handler=_command_benchmark)

    divergence_parser = subparsers.add_parser(
        "divergence", help="compare two sources for one object, in kilometres"
    )
    common(divergence_parser, objects=60)
    divergence_parser.set_defaults(handler=_command_divergence)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point. Returns a process exit status and never raises on bad input."""
    parser = build_parser()
    arguments = parser.parse_args(argv)
    try:
        return int(arguments.handler(arguments))
    except (ValueError, OSError) as exception:
        print(f"tlecheck: {exception}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
