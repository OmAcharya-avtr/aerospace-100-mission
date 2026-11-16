"""Baseline 3: the injected corpus and the disjoint-block power measurement."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from seedaudit import corpus


def test_the_injected_rates_span_more_than_three_orders_of_magnitude() -> None:
    assert max(corpus.INJECTED_RATES) / min(corpus.INJECTED_RATES) >= 1000.0
    assert pytest.approx(1.0 / 44.0, rel=1e-15, abs=0.0) in corpus.INJECTED_RATES


def test_rate_tags_are_filesystem_safe_and_distinct() -> None:
    assert corpus.rate_tag(1.0 / 44.0) == "0p022727"
    assert corpus.rate_tag(1.0) == "1p000000"
    assert corpus.rate_script_name(0.1) == "rate_0p100000.py"
    tags = [corpus.rate_tag(rate) for rate in corpus.INJECTED_RATES]
    assert len(set(tags)) == len(tags)


@pytest.mark.parametrize("bad", [0.0, -0.1, 1.5])
def test_a_rate_outside_the_half_open_unit_interval_is_rejected(bad: float) -> None:
    with pytest.raises(ValueError, match="must lie in"):
        corpus.rate_tag(bad)


def test_the_generator_writes_every_declared_member_of_the_corpus(tmp_path: Path) -> None:
    manifest = corpus.generate_corpus(tmp_path / "c")
    for rate in corpus.INJECTED_RATES:
        assert (manifest.root / manifest.rate_scripts[rate]).is_file()
    for name in corpus.NEGATIVE_CONTROLS:
        assert (manifest.root / name).is_file()
        assert name in manifest.references
    assert (manifest.root / corpus.SELF_POLLUTING_SCRIPT).is_file()
    assert manifest.self_polluting == corpus.SELF_POLLUTING_SCRIPT
    assert (manifest.root / corpus.ITERATION_ORDER_SCRIPT).is_file()
    assert (manifest.root / corpus.SUB_PRECISION_SCRIPT).is_file()
    assert (manifest.root / "inputs" / "data.txt").is_file()
    assert (manifest.root / "shared" / "config.txt").is_file()
    assert list(manifest.script_names) == sorted(manifest.script_names)


def test_the_generator_is_byte_reproducible(tmp_path: Path) -> None:
    # It takes no seed: every file it writes is a fixed function of the
    # declared rates, and the randomness lives inside the generated scripts.
    first = corpus.generate_corpus(tmp_path / "a")
    second = corpus.generate_corpus(tmp_path / "b")
    for name in first.script_names:
        assert (first.root / name).read_bytes() == (second.root / name).read_bytes()
    with pytest.raises(ValueError, match="non-empty"):
        corpus.generate_corpus(tmp_path / "c", ())


def test_every_negative_control_matches_its_declared_reference(tmp_path: Path) -> None:
    manifest = corpus.generate_corpus(tmp_path / "c")
    for name in corpus.NEGATIVE_CONTROLS:
        assert corpus.run_script_once(manifest.root, name) == manifest.references[name], name


def test_the_rate_one_script_manifests_on_every_run(tmp_path: Path) -> None:
    manifest = corpus.generate_corpus(tmp_path / "c")
    name = manifest.rate_scripts[1.0]
    indicators = corpus.manifestation_indicators(manifest.root, name, manifest.references[name], 6)
    assert indicators.sum() == 6.0
    assert corpus.REFERENCE_VALUE != corpus.PERTURBED_VALUE


def test_the_iteration_order_script_keeps_its_number_fixed(tmp_path: Path) -> None:
    # A numeric-only comparison is blind to it; this measures the blindness
    # rather than asserting it.
    manifest = corpus.generate_corpus(tmp_path / "c")
    values = set()
    for _ in range(4):
        output = corpus.run_script_once(manifest.root, corpus.ITERATION_ORDER_SCRIPT)
        values.add([line for line in output.splitlines() if line.startswith("value")][0])
    assert len(values) == 1


def test_the_sub_precision_script_moves_only_beyond_six_decimals(tmp_path: Path) -> None:
    manifest = corpus.generate_corpus(tmp_path / "c")
    output = corpus.run_script_once(manifest.root, corpus.SUB_PRECISION_SCRIPT)
    lines = output.splitlines()
    assert [line for line in lines if line.startswith("rounded")][0] == "rounded 0.123457"
    assert [line for line in lines if line.startswith("full")][0] != "full 0.1234567890123"


def test_block_detection_uses_disjoint_blocks_and_drops_a_partial_one() -> None:
    # Overlapping windows share runs and would inflate the apparent number of
    # independent audits, which is the error this product exists to report.
    reference, pairwise = corpus.block_detection(np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0]), 2)
    assert reference.size == 3
    assert list(reference) == [1.0, 0.0, 0.0]
    assert list(pairwise) == [1.0, 0.0, 0.0]
    assert corpus.block_detection(np.zeros(7), 2)[0].size == 3


def test_block_detection_separates_the_two_designs() -> None:
    reference, pairwise = corpus.block_detection(np.ones(4), 2)
    assert list(reference) == [1.0, 1.0]
    # Pairwise cannot fire: the runs agree with each other.
    assert list(pairwise) == [0.0, 0.0]
    assert list(corpus.block_detection(np.array([1.0, 0.0, 1.0, 1.0]), 2)[1]) == [1.0, 0.0]
    assert list(corpus.block_detection(np.array([1.0, 0.0, 1.0]), 1)[1]) == [0.0, 0.0, 0.0]


def test_input_validation(tmp_path: Path) -> None:
    manifest = corpus.generate_corpus(tmp_path / "c")
    with pytest.raises(ValueError, match="no such corpus script"):
        corpus.run_script_once(manifest.root, "absent.py")
    with pytest.raises(ValueError, match="at least 1"):
        corpus.manifestation_indicators(manifest.root, "deterministic_control.py", "", 0)
    with pytest.raises(TypeError, match="must be an int"):
        corpus.manifestation_indicators(
            manifest.root, "deterministic_control.py", "", 2.0  # type: ignore[arg-type]
        )
    with pytest.raises(ValueError, match="cannot form even one block"):
        corpus.block_detection(np.zeros(3), 4)
    with pytest.raises(TypeError, match="must be an int"):
        corpus.block_detection(np.zeros(4), 2.0)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="at least 1"):
        corpus.block_detection(np.zeros(4), 0)


def test_regression_the_deterministic_control_value_is_pinned(tmp_path: Path) -> None:
    # The 64th harmonic number, pinned so that a change to the control is a
    # test failure rather than a silently different negative control.
    manifest = corpus.generate_corpus(tmp_path / "c")
    assert manifest.references["deterministic_control.py"] == (
        "script deterministic_control.py\nvalue 4.743891\n"
    )
