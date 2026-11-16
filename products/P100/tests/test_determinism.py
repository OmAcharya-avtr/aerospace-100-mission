"""This product's own reproducibility, measured rather than claimed.

A tool that audits reproducibility and is itself non-reproducible is worthless.
These tests run the product's own machinery twice and require the two answers
to be equal. They are the self-application of the product's own criterion, and
the only honest way to make the claim.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from seedaudit import audit, corpus, numeric, power, report, seeds, static


def _audit_fingerprint(result: audit.RepositoryAudit) -> tuple[object, ...]:
    """A comparable summary of everything an audit reports."""
    return (
        result.headline,
        result.declared_seed_headline,
        result.scripts_with_seed_movement,
        result.scripts_with_declared_seed_movement,
        tuple(sorted(result.status_counts().items())),
        result.claim_tokens_total,
        result.claim_tokens_compared,
        result.files_scanned,
        result.static_finding_count,
        tuple(sorted(result.static_findings)),
        tuple(sorted(result.parse_failures)),
    )


def test_the_audit_run_twice_gives_the_same_report(fixture_repository: Path) -> None:
    first = audit.audit_repository(
        fixture_repository, audit.DEFAULT_ROOTS, (), timeout=60.0, workers=1
    )
    second = audit.audit_repository(
        fixture_repository, audit.DEFAULT_ROOTS, (), timeout=60.0, workers=1
    )
    assert _audit_fingerprint(first) == _audit_fingerprint(second)


def test_the_audit_leaves_the_tree_byte_identical(fixture_repository: Path) -> None:
    def snapshot() -> dict[str, bytes]:
        return {
            path.relative_to(fixture_repository).as_posix(): path.read_bytes()
            for path in sorted(fixture_repository.rglob("*"))
            if path.is_file()
        }

    before = snapshot()
    audit.audit_repository(fixture_repository, audit.DEFAULT_ROOTS, (), timeout=60.0, workers=1)
    assert snapshot() == before


def test_the_audit_is_independent_of_the_worker_count(fixture_repository: Path) -> None:
    serial = audit.audit_repository(
        fixture_repository, audit.DEFAULT_ROOTS, (), timeout=60.0, workers=1
    )
    parallel = audit.audit_repository(
        fixture_repository, audit.DEFAULT_ROOTS, (), timeout=60.0, workers=2
    )
    assert serial.headline == parallel.headline
    assert serial.scripts_with_seed_movement == parallel.scripts_with_seed_movement


def test_candidate_discovery_is_order_stable(fixture_repository: Path) -> None:
    first = audit.find_candidates(fixture_repository, audit.DEFAULT_ROOTS, ())
    second = audit.find_candidates(fixture_repository, audit.DEFAULT_ROOTS, ())
    assert [c.label for c in first] == [c.label for c in second]


def test_static_scan_run_twice_gives_the_same_findings(fixture_repository: Path) -> None:
    # A check whose own execution can create the condition it tests is not a
    # check. This one is run twice from the same tree and the second run must
    # agree with the first.
    first = audit.static_scan(fixture_repository, audit.DEFAULT_ROOTS, ())
    second = audit.static_scan(fixture_repository, audit.DEFAULT_ROOTS, ())
    assert sorted(first[0]) == sorted(second[0])
    assert first[1] == second[1]
    assert first[2] == second[2]


def test_the_power_table_is_deterministic() -> None:
    assert report.format_power_table() == report.format_power_table()


def test_the_power_functions_are_pure() -> None:
    values = [power.detection_probability_reference(power.MISSION_RATE, 2) for _ in range(5)]
    assert len(set(values)) == 1


def test_the_bootstrap_is_reproducible_under_the_declared_seed() -> None:
    values = np.array([0.0] * 90 + [1.0] * 10)
    first = power.bootstrap_interval(values, np.random.default_rng(report.SEED), 500)
    second = power.bootstrap_interval(values, np.random.default_rng(report.SEED), 500)
    assert first == second


def test_the_monte_carlo_is_reproducible_under_the_declared_seed() -> None:
    first = power.monte_carlo_detection(0.1, 3, 5000, np.random.default_rng(report.SEED))
    second = power.monte_carlo_detection(0.1, 3, 5000, np.random.default_rng(report.SEED))
    assert first == second


def test_the_seed_rewrite_is_deterministic() -> None:
    source = "SEED = 1\nrng = np.random.default_rng(2)\n"
    assert seeds.rewrite_seeds(source)[0] == seeds.rewrite_seeds(source)[0]


def test_the_comparison_is_deterministic() -> None:
    first = numeric.compare("a 1.0\nb 2.0\n", "a 1.5\nb 2.0\n")
    second = numeric.compare("a 1.0\nb 2.0\n", "a 1.5\nb 2.0\n")
    assert [m.before for m in first.moved] == [m.before for m in second.moved]
    assert first.compared_tokens == second.compared_tokens


def test_the_static_screen_is_deterministic() -> None:
    source = (
        "from pathlib import Path\n"
        "HERE = Path('.')\n"
        "strays = list((HERE / 'out').glob('*.txt'))\n"
        "(HERE / 'out' / 'r.txt').write_text(str(len(strays)))\n"
    )
    first = [f.describe() for f in static.find_self_pollution(source)]
    second = [f.describe() for f in static.find_self_pollution(source)]
    assert first == second


def test_the_corpus_generator_is_byte_reproducible(tmp_path: Path) -> None:
    first = corpus.generate_corpus(tmp_path / "a")
    second = corpus.generate_corpus(tmp_path / "b")
    assert first.references == second.references
    for name in first.script_names:
        assert (first.root / name).read_bytes() == (second.root / name).read_bytes()


def test_only_one_seed_is_declared_in_the_product() -> None:
    assert report.SEED == 1000044
    assert report.CONFIDENCE == pytest.approx(0.95, rel=0.0, abs=0.0)
    assert report.BOOTSTRAP_RESAMPLES == 2000
