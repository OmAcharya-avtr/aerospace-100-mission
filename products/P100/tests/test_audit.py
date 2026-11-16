"""Auditing a repository: candidate discovery, one claim, and a whole tree.

Every test here runs against a synthetic repository built in ``tmp_path``. A
test suite for a tool that audits a live repository must not depend on that
repository's state, or its own result depends on whatever else is happening in
the tree while it runs.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from seedaudit import audit


def _candidate(root: Path, product: str) -> audit.Candidate:
    """The fixture candidate belonging to one product directory."""
    return next(
        c
        for c in audit.find_candidates(root, audit.DEFAULT_ROOTS, ())
        if c.source_dir == f"products/{product}"
    )


def test_the_declared_roots_and_exclusions_each_carry_a_reason() -> None:
    # An exclusion without a stated reason is a silent skip.
    assert audit.DEFAULT_ROOTS == ("products", "scripts")
    for item in audit.DEFAULT_EXCLUSIONS:
        assert item in audit.EXCLUSION_REASONS
        assert len(audit.EXCLUSION_REASONS[item]) > 20


def test_file_discovery_is_sorted_and_honours_exclusions(fixture_repository: Path) -> None:
    files = audit.python_files(fixture_repository, audit.DEFAULT_ROOTS, ())
    assert "products/PA/validation/seeded_script.py" in files
    assert len(files) == 3
    assert list(files) == sorted(files)
    excluded = audit.python_files(fixture_repository, audit.DEFAULT_ROOTS, ("products/PA",))
    assert len(excluded) == 2
    assert not any(path.startswith("products/PA/") for path in excluded)
    assert audit.python_files(fixture_repository, ("absent",), ()) == ()


def test_file_discovery_skips_cache_directories(fixture_repository: Path) -> None:
    cache = fixture_repository / "products" / "PA" / "validation" / "__pycache__"
    cache.mkdir()
    (cache / "x.py").write_text("x = 1\n", encoding="utf-8")
    files = audit.python_files(fixture_repository, audit.DEFAULT_ROOTS, ())
    assert not any("__pycache__" in path for path in files)


def test_candidate_discovery_records_the_script_output_and_token_count(
    fixture_repository: Path,
) -> None:
    candidates = audit.find_candidates(fixture_repository, audit.DEFAULT_ROOTS, ())
    assert len(candidates) == 3
    assert {c.source_dir for c in candidates} == {"products/PA", "products/PB", "products/PC"}
    first = _candidate(fixture_repository, "PA")
    assert first.script == "validation/seeded_script.py"
    assert first.output == "validation/seeded_script_output.txt"
    assert first.claim_tokens == 2
    assert all(c.label.startswith("products/") for c in candidates)


def test_a_script_making_no_claim_or_declaring_no_seed_is_not_a_candidate(
    fixture_repository: Path,
) -> None:
    directory = fixture_repository / "products" / "PA" / "validation"
    (directory / "unclaimed.py").write_text("SEED = 1\n", encoding="utf-8")
    (directory / "unseeded.py").write_text("x = 1\n", encoding="utf-8")
    (directory / "unseeded_output.txt").write_text("value 1.0\n", encoding="utf-8")
    (directory / "broken.py").write_text("SEED = 1\ndef (:\n", encoding="utf-8")
    (directory / "broken_output.txt").write_text("value 1.0\n", encoding="utf-8")
    source = fixture_repository / "products" / "PA" / "src"
    source.mkdir()
    (source / "lib.py").write_text("SEED = 1\n", encoding="utf-8")
    (source / "lib_output.txt").write_text("value 1.0\n", encoding="utf-8")
    labels = {c.label for c in audit.find_candidates(fixture_repository, audit.DEFAULT_ROOTS, ())}
    assert "products/PA/validation/unclaimed.py" not in labels
    assert "products/PA/validation/unseeded.py" not in labels
    assert "products/PA/validation/broken.py" not in labels
    assert "products/PA/src/lib.py" not in labels


def test_one_claim_whose_number_depends_on_the_seed(fixture_repository: Path) -> None:
    result = audit.audit_claim(fixture_repository, _candidate(fixture_repository, "PA"), 60.0)
    assert result.status == "compared"
    assert result.evaluable
    assert len(result.seed_attributable) == 1
    assert result.seed_attributable[0].before == "32.375000"
    # The constant 42 is not a seeded quantity and must not be reported.
    assert all(moved.before != "42" for moved in result.seed_attributable)
    # The unchanged-seed control is clean, so the finding is seed-attributable.
    assert result.declared_seed_moved == []


def test_one_claim_whose_number_does_not_depend_on_the_seed(fixture_repository: Path) -> None:
    result = audit.audit_claim(fixture_repository, _candidate(fixture_repository, "PB"), 60.0)
    assert result.status == "compared"
    assert result.seed_attributable == []
    assert result.declared_seed_moved == []


def test_a_claim_whose_script_writes_no_output_is_unevaluable_not_clean(
    fixture_repository: Path,
) -> None:
    # Reporting it as clean would be a verdict obtained by comparing the
    # committed file against itself.
    result = audit.audit_claim(fixture_repository, _candidate(fixture_repository, "PC"), 60.0)
    assert result.status == "no_output"
    assert not result.evaluable


def test_auditing_every_claim_leaves_the_tree_byte_identical(fixture_repository: Path) -> None:
    def snapshot() -> dict[str, bytes]:
        return {
            path.relative_to(fixture_repository).as_posix(): path.read_bytes()
            for path in sorted(fixture_repository.rglob("*"))
            if path.is_file()
        }

    before = snapshot()
    for candidate in audit.find_candidates(fixture_repository, audit.DEFAULT_ROOTS, ()):
        audit.audit_claim(fixture_repository, candidate, timeout=60.0)
    assert snapshot() == before


def test_a_claim_that_times_out_is_reported_as_a_timeout(fixture_repository: Path) -> None:
    directory = fixture_repository / "products" / "PA" / "validation"
    (directory / "slow.py").write_text("SEED = 1\nimport time\ntime.sleep(30)\n", encoding="utf-8")
    (directory / "slow_output.txt").write_text("value 1.0\n", encoding="utf-8")
    candidate = next(
        c
        for c in audit.find_candidates(fixture_repository, audit.DEFAULT_ROOTS, ())
        if c.script == "validation/slow.py"
    )
    assert audit.audit_claim(fixture_repository, candidate, timeout=1.0).status == "timeout"


def test_the_static_scan_reports_its_coverage_and_its_parse_failures(
    fixture_repository: Path,
) -> None:
    findings, failures, scanned = audit.static_scan(fixture_repository, audit.DEFAULT_ROOTS, ())
    assert scanned == 3
    assert failures == {}
    assert isinstance(findings, dict)
    broken = fixture_repository / "products" / "PA" / "validation" / "broken.py"
    broken.write_text("def (:\n", encoding="utf-8")
    _findings, failures, _scanned = audit.static_scan(fixture_repository, audit.DEFAULT_ROOTS, ())
    assert "products/PA/validation/broken.py" in failures
    assert "SyntaxError" in failures["products/PA/validation/broken.py"]


def test_the_static_scan_flags_the_write_under_glob_pattern(fixture_repository: Path) -> None:
    directory = fixture_repository / "products" / "PA" / "validation"
    (directory / "scanner.py").write_text(
        "from pathlib import Path\n"
        "HERE = Path('.')\n"
        "strays = list((HERE / 'out').glob('*.txt'))\n"
        "(HERE / 'out' / 'r.txt').write_text(str(len(strays)))\n",
        encoding="utf-8",
    )
    findings, _failures, _scanned = audit.static_scan(fixture_repository, audit.DEFAULT_ROOTS, ())
    assert "products/PA/validation/scanner.py" in findings


def test_the_whole_tree_audit_reproduces_the_fixture_ground_truth(
    fixture_repository: Path,
) -> None:
    # Ground truth for this tree: one number moves, one does not, one claim is
    # unevaluable.
    result = audit.audit_repository(
        fixture_repository, audit.DEFAULT_ROOTS, (), timeout=60.0, workers=1
    )
    assert result.headline == 1
    assert result.declared_seed_headline == 0
    assert result.scripts_with_seed_movement == ("products/PA/validation/seeded_script.py",)
    assert result.scripts_with_declared_seed_movement == ()
    counts = result.status_counts()
    assert counts["compared"] == 2
    assert counts["no_output"] == 1
    assert len(result.candidates) == 3
    assert len(result.results) == 3
    assert result.claim_tokens_total == 5
    assert result.claim_tokens_compared == 4
    # One run per arrangement, so the repeat-axis bound is 1 - 0.05**(1/1).
    assert result.escaped_bound == pytest.approx(0.95, rel=1e-12, abs=0.0)
    assert result.root_name == "fixture_repo"
    assert "/" not in result.root_name


def test_the_audit_honours_a_limit_and_a_deadline_and_reports_both(
    fixture_repository: Path,
) -> None:
    limited = audit.audit_repository(
        fixture_repository, audit.DEFAULT_ROOTS, (), timeout=60.0, workers=1, limit=1
    )
    assert len(limited.results) == 1
    assert len(limited.candidates) == 3
    expired = audit.audit_repository(
        fixture_repository, audit.DEFAULT_ROOTS, (), timeout=60.0, workers=1, deadline_seconds=-1.0
    )
    assert expired.deadline_reached
    assert expired.results == []


def test_the_audit_reports_progress(fixture_repository: Path) -> None:
    seen: list[tuple[int, int, str, str]] = []
    audit.audit_repository(
        fixture_repository,
        audit.DEFAULT_ROOTS,
        (),
        timeout=60.0,
        workers=1,
        progress=lambda i, t, label, status: seen.append((i, t, label, status)),
    )
    assert len(seen) == 3
    assert seen[0][1] == 3


def test_input_validation(fixture_repository: Path, tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="existing directory"):
        audit.audit_repository(tmp_path / "absent")
    with pytest.raises(ValueError, match="positive int"):
        audit.audit_repository(fixture_repository, workers=0)
    assert (
        audit.unparseable_files(fixture_repository, ("products/PA/validation/seeded_script.py",))
        == {}
    )
