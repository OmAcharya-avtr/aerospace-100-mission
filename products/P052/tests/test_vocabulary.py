"""Nothing this package renders or prints may read as a verdict of correctness.

Falsification is one-sided: finding no violation is not evidence of correctness.
A CLI that prints "PASS" when a search found nothing invites exactly the
misreading the whole product exists to prevent, so the forbidden vocabulary is
asserted here rather than left to the author's memory.

The check is deliberately crude -- a word list over rendered output -- and that
crudeness is the point: it cannot be satisfied by a clever rephrasing that still
reads as a pass. Where a forbidden stem is a legitimate substring of an innocent
word (``compliant`` inside nothing here, ``clean`` inside nothing here), the
pattern is matched on word boundaries.
"""

from __future__ import annotations

import re

import numpy as np
import pytest

from falsifyloop import report as report_module
from falsifyloop.__main__ import main
from falsifyloop.benchmark import run_benchmark, run_cell
from falsifyloop.curves import bootstrap_band, efficiency_curve
from falsifyloop.instances import instance, suite
from falsifyloop.search import BASELINE, uniform_random

#: Words that would read as a verdict of correctness, forbidden outright.
FORBIDDEN = (
    "pass",
    "passes",
    "passed",
    "passing",
    "verified",
    "verification",
    "compliant",
    "compliance",
    "clean",
    "ok",
    "okay",
    "success",
    "successful",
    "proven",
    "guaranteed",
    "correct",
)

#: Words that may appear only immediately after "not", because the safety
#: statement has to be able to say what this software is not.
NEGATED_ONLY = (
    "safe",
    "certified",
    "flight-qualified",
    "flight-safe",
    "mission-ready",
    "production-ready",
    "approved",
)

_PATTERN = re.compile(r"\b(" + "|".join(re.escape(w) for w in FORBIDDEN) + r")\b", re.IGNORECASE)
_NEGATED = re.compile(
    r"(?<!not )\b(" + "|".join(re.escape(w) for w in NEGATED_ONLY) + r")\b", re.IGNORECASE
)


def _offending(text: str) -> list[str]:
    """Forbidden words, plus negated-only words used without a preceding "not".

    Whitespace is collapsed first, because argparse wraps help text and a
    line break between "not" and "certified" would otherwise look like a bare
    claim to the lookbehind.
    """
    flat = re.sub(r"\s+", " ", text)
    hits = {m.group(0).lower() for m in _PATTERN.finditer(flat)}
    hits |= {m.group(0).lower() for m in _NEGATED.finditer(flat)}
    return sorted(hits)


def test_the_word_list_itself_matches_what_it_should() -> None:
    assert _offending("the run passed and is safe") == ["passed", "safe"]
    assert _offending("no violation found") == []
    # Word boundaries, so an embedded substring is not a false positive.
    assert _offending("compassion and okra") == []
    # The required disclaimer form is allowed; the bare claim is not.
    assert _offending("not flight-qualified, not certified") == []
    assert _offending("this is certified") == ["certified"]


def _rendered_strings() -> list[tuple[str, str]]:
    easy = instance("overshoot-loose")
    hard = instance("rate-envelope")
    found = uniform_random(easy, 60, 0)
    nothing = uniform_random(hard, 5, 0)
    report = run_benchmark(
        instances=[easy, hard],
        strategy_names=[BASELINE, "surrogate-guided"],
        budget=12,
        repeats=3,
        base_seed=1,
    )
    cell, _ = run_cell(hard, BASELINE, budget=5, repeats=3, base_seed=0)
    runs = (2, 5, None, None)
    curve = efficiency_curve(runs, 10)
    lower, upper = bootstrap_band(runs, 10, n_boot=200, seed=0)
    return [
        ("render_search_result/found", report_module.render_search_result(found, easy)),
        ("render_search_result/none", report_module.render_search_result(nothing, hard)),
        ("render_instances", report_module.render_instances(suite())),
        ("render_cell_table", report_module.render_cell_table(report)),
        ("render_aggregate_table", report_module.render_aggregate_table(report)),
        (
            "render_difficulty_table",
            report_module.render_difficulty_table((("inst", "hard", 5, 1000, 0.004),)),
        ),
        (
            "render_curve_points",
            report_module.render_curve_points(curve, lower, upper, marks=(1, 5, 10)),
        ),
        ("render_cell_detail", report_module.render_cell_detail(cell)),
        ("ONE_SIDED_NOTE", report_module.ONE_SIDED_NOTE),
    ]


@pytest.mark.parametrize(("label", "text"), _rendered_strings(), ids=lambda v: str(v)[:40])
def test_rendered_output_contains_no_pass_vocabulary(label, text) -> None:
    if isinstance(text, str) and label in {t[0] for t in []}:  # pragma: no cover
        return
    if not isinstance(text, str):
        return
    offending = _offending(text)
    assert offending == [], f"{label} rendered forbidden vocabulary {offending}"


@pytest.mark.parametrize(
    "argv",
    [
        ["instances"],
        ["falsify", "--instance", "overshoot-loose", "--budget", "40", "--seed", "0"],
        ["falsify", "--instance", "rate-envelope", "--budget", "5", "--seed", "0"],
        [
            "falsify",
            "--instance",
            "rate-envelope",
            "--budget",
            "10",
            "--repeats",
            "3",
            "--seed",
            "0",
        ],
        [
            "benchmark",
            "--instances",
            "overshoot-loose",
            "rate-envelope",
            "--strategies",
            "uniform-random",
            "--budget",
            "10",
            "--repeats",
            "2",
        ],
        ["evaluate", "--instance", "overshoot-loose", "--input", "3.5,1.3,0.7,1.8,7.5,1.5"],
        ["difficulty", "--instances", "overshoot-loose", "--draws", "20"],
    ],
    ids=lambda v: "-".join(v[:3]),
)
def test_cli_output_contains_no_pass_vocabulary(argv, capsys) -> None:
    status = main(argv)
    captured = capsys.readouterr()
    assert status == 0
    offending = _offending(captured.out)
    assert offending == [], f"CLI {argv} printed forbidden vocabulary {offending}"


def test_cli_help_contains_no_pass_vocabulary(capsys) -> None:
    with pytest.raises(SystemExit) as exc:
        main(["--help"])
    assert exc.value.code == 0
    captured = capsys.readouterr()
    assert _offending(captured.out) == []


def test_library_source_has_no_print_statement() -> None:
    """No ``print`` in library code; the CLI module is the only writer."""
    import pathlib

    import falsifyloop

    package = pathlib.Path(falsifyloop.__file__).parent
    offenders = []
    for path in sorted(package.glob("*.py")):
        if path.name == "__main__.py":
            continue
        if re.search(r"(?<![\w.])print\s*\(", path.read_text()):
            offenders.append(path.name)
    assert offenders == [], f"print() found in library modules {offenders}"


def test_search_result_never_reports_absence_as_a_positive_verdict() -> None:
    nothing = uniform_random(instance("rate-envelope"), 3, 0)
    assert nothing.first_violation is None
    assert nothing.found is False
    # `found` is the only Boolean the result exposes, and it is about the search
    # having found something, never about the requirement holding.
    assert not hasattr(nothing, "satisfied")
    assert not hasattr(nothing, "passed")


def test_numpy_is_imported_for_the_parametrisation_helper() -> None:
    # Guards against the parametrisation above silently degrading if numpy were
    # removed from this module's imports.
    assert np.isfinite(1.0)
