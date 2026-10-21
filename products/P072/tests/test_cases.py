"""YAML case loading, schema validation and round-tripping."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from downlinkpick.cases import (
    AUTOCORRELATION_KEY,
    PRODUCT_KEYS,
    WINDOW_KEYS,
    Case,
    case_to_mapping,
    load_case,
    product_from_mapping,
    window_from_mapping,
)
from downlinkpick.selection import SelectionProblem

ROOT = Path(__file__).resolve().parent.parent
CASE_DIR = ROOT / "examples" / "cases"

SHIPPED = (
    "three_pass_declared.yaml",
    "missing_declarations.yaml",
    "single_short_pass.yaml",
)


@pytest.mark.parametrize("name", SHIPPED)
def test_every_shipped_case_loads(name: str) -> None:
    case = load_case(CASE_DIR / name)
    assert case.name
    assert case.description
    assert len(case.inventory) >= 1
    assert len(case.plan) >= 1
    assert set(case.autocorrelation) == set(case.inventory.names)


def test_reference_case_shape(reference_case: Case) -> None:
    assert len(reference_case.inventory) == 24
    assert reference_case.plan.names == ("pass1", "pass2", "pass3")
    assert reference_case.inventory.missing_declarations() == ()
    assert reference_case.inventory.missing_compressibility() == ()


def test_missing_case_has_gaps(missing_case: Case) -> None:
    assert len(missing_case.inventory.missing_declarations()) == 10
    assert len(missing_case.inventory.missing_compressibility()) == 7
    with pytest.raises(ValueError, match="declared_value is None for"):
        SelectionProblem.from_declared(missing_case.inventory, missing_case.plan)


def test_small_case_is_enumerable(small_case: Case) -> None:
    from downlinkpick.exact import enumeration_size

    problem = SelectionProblem.from_declared(small_case.inventory, small_case.plan)
    assert enumeration_size(problem) <= 2048


def test_missing_file_message_quotes_the_path_as_given() -> None:
    with pytest.raises(FileNotFoundError, match="no case file at examples/cases/absent.yaml"):
        load_case("examples/cases/absent.yaml")


def test_case_must_be_a_mapping(tmp_path: Path) -> None:
    target = tmp_path / "bad.yaml"
    target.write_text("- just\n- a\n- list\n")
    with pytest.raises(ValueError, match="must be a mapping"):
        load_case(target)


def test_case_requires_products_and_windows(tmp_path: Path) -> None:
    target = tmp_path / "bad.yaml"
    target.write_text("name: x\nproducts: []\n")
    with pytest.raises(ValueError, match="missing the required key 'windows'"):
        load_case(target)
    target.write_text("name: x\nwindows: []\n")
    with pytest.raises(ValueError, match="missing the required key 'products'"):
        load_case(target)


def test_unknown_product_key_rejected() -> None:
    with pytest.raises(ValueError, match="unknown product key"):
        product_from_mapping(
            {"name": "p", "raw_bytes": 1.0, "age_s": 0.0, "expiry_s": 1.0, "colour": "red"}
        )


def test_missing_product_key_rejected() -> None:
    with pytest.raises(ValueError, match="missing the required key 'expiry_s'"):
        product_from_mapping({"name": "p", "raw_bytes": 1.0, "age_s": 0.0})


def test_product_entry_must_be_a_mapping() -> None:
    with pytest.raises(ValueError, match="must be a mapping"):
        product_from_mapping(["not", "a", "mapping"])


def test_depends_on_accepts_a_bare_string() -> None:
    product, rho = product_from_mapping(
        {
            "name": "p",
            "raw_bytes": 1.0e6,
            "age_s": 0.0,
            "expiry_s": 100.0,
            "depends_on": "other",
        }
    )
    assert product.depends_on == ("other",)
    assert rho is None


def test_autocorrelation_is_optional_and_parsed() -> None:
    _product, rho = product_from_mapping(
        {
            "name": "p",
            "raw_bytes": 1.0e6,
            "age_s": 0.0,
            "expiry_s": 100.0,
            AUTOCORRELATION_KEY: 0.75,
        }
    )
    assert rho == pytest.approx(0.75)


def test_unknown_window_key_rejected() -> None:
    with pytest.raises(ValueError, match="unknown window key"):
        window_from_mapping(
            {
                "name": "w",
                "start_s": 0.0,
                "end_s": 10.0,
                "rate_breakpoints": [[0.0, 1.0], [10.0, 1.0]],
                "elevation": 42.0,
            }
        )


def test_missing_window_key_rejected() -> None:
    with pytest.raises(ValueError, match="missing the required key 'rate_breakpoints'"):
        window_from_mapping({"name": "w", "start_s": 0.0, "end_s": 10.0})


def test_breakpoints_must_be_pairs() -> None:
    with pytest.raises(ValueError, match="must be a list"):
        window_from_mapping(
            {"name": "w", "start_s": 0.0, "end_s": 10.0, "rate_breakpoints": 5}
        )
    with pytest.raises(ValueError, match="must be a \\[time_s, rate_bps\\] pair"):
        window_from_mapping(
            {
                "name": "w",
                "start_s": 0.0,
                "end_s": 10.0,
                "rate_breakpoints": [[0.0, 1.0, 2.0], [10.0, 1.0]],
            }
        )


def test_round_trip_through_a_mapping(reference_case: Case, tmp_path: Path) -> None:
    mapping = case_to_mapping(reference_case)
    target = tmp_path / "round.yaml"
    target.write_text(yaml.safe_dump(mapping, sort_keys=False))
    reloaded = load_case(target)
    assert reloaded.inventory.names == reference_case.inventory.names
    assert reloaded.plan.names == reference_case.plan.names
    for name in reference_case.inventory.names:
        original = reference_case.inventory[name]
        copy = reloaded.inventory[name]
        assert copy.raw_bytes == pytest.approx(original.raw_bytes, rel=1e-12)
        assert copy.expiry_s == pytest.approx(original.expiry_s, rel=1e-12)
        assert copy.depends_on == original.depends_on
    assert reloaded.autocorrelation.keys() == reference_case.autocorrelation.keys()


def test_round_trip_preserves_a_missing_declaration(
    missing_case: Case, tmp_path: Path
) -> None:
    target = tmp_path / "round.yaml"
    target.write_text(yaml.safe_dump(case_to_mapping(missing_case), sort_keys=False))
    reloaded = load_case(target)
    assert set(reloaded.inventory.missing_declarations()) == set(
        missing_case.inventory.missing_declarations()
    )
    assert set(reloaded.inventory.missing_compressibility()) == set(
        missing_case.inventory.missing_compressibility()
    )


def test_key_sets_are_what_the_dataclasses_accept() -> None:
    assert "raw_bytes" in PRODUCT_KEYS
    assert "rate_breakpoints" in WINDOW_KEYS
    assert AUTOCORRELATION_KEY not in PRODUCT_KEYS
