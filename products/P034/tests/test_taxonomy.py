"""Taxonomy: structure, binning arithmetic and input validation."""

from __future__ import annotations

import math

import pytest

from faultinject.taxonomy import (
    DURATION_FRAC,
    START_FRAC,
    TAXONOMY,
    FaultClass,
    FaultKind,
    ParamSpec,
    Scale,
    iter_param_bins,
    kinds,
    kinds_of_class,
    spec,
    total_cells,
)


def test_sixteen_kinds_and_five_classes():
    assert len(kinds()) == 16
    assert {s.fault_class for s in TAXONOMY.values()} == set(FaultClass)


def test_class_membership_counts():
    # Hand count from the specification scope list.
    assert len(kinds_of_class(FaultClass.SENSOR)) == 5
    assert len(kinds_of_class(FaultClass.ACTUATOR)) == 3
    assert len(kinds_of_class(FaultClass.BUS)) == 3
    assert len(kinds_of_class(FaultClass.TIMING)) == 2
    assert len(kinds_of_class(FaultClass.NUMERICAL)) == 3


def test_total_cells_known_answer():
    # Hand sum: 24+24+8+24+24 + 12+4+12 + 12+12+12 + 8+12 + 12+12+36 = 248
    assert total_cells() == 248


def test_every_kind_has_description_and_reference():
    for kind in kinds():
        s = spec(kind)
        assert s.description.endswith(".")
        assert len(s.reference) > 10
        assert s.channels


def test_log_edges_known_answer():
    # offset in [0.1, 10], 3 log bins:
    #   edge_i = 10 ** (-1 + 2 i / 3) -> 0.1, 10**(-1/3), 10**(1/3), 10
    p = spec(FaultKind.SENSOR_BIAS).param("offset")
    e = p.edges()
    assert len(e) == 4
    assert e[0] == pytest.approx(0.1, abs=1e-15)
    assert e[1] == pytest.approx(10.0 ** (-1.0 / 3.0), rel=1e-15)
    assert e[2] == pytest.approx(10.0 ** (1.0 / 3.0), rel=1e-15)
    assert e[3] == pytest.approx(10.0, rel=1e-15)


def test_linear_edges_known_answer():
    p = ParamSpec("x", "unit", 0.0, 1.0, 4)
    assert p.edges() == (0.0, 0.25, 0.5, 0.75, 1.0)


def test_bin_of_hand_values():
    p = spec(FaultKind.SENSOR_BIAS).param("offset")
    assert p.bin_of(0.1) == 0
    assert p.bin_of(0.2) == 0
    assert p.bin_of(1.0) == 1  # 0.4642 < 1 < 2.1544
    assert p.bin_of(5.0) == 2
    assert p.bin_of(10.0) == 2  # upper edge clamps into the last bin


def test_bin_of_clamps_outside_range():
    p = ParamSpec("x", "unit", 1.0, 2.0, 2)
    assert p.bin_of(-5.0) == 0
    assert p.bin_of(1e9) == 1


def test_representative_is_inside_its_bin():
    for kind in kinds():
        for p in spec(kind).params:
            for b in range(p.n_bins):
                v = p.representative(b)
                assert math.isfinite(v)
                assert p.bin_of(v) == b, f"{kind.value}.{p.name} bin {b} -> {v}"


def test_representative_of_huge_log_bin_is_finite():
    # Regression: the top bin of magnitude reaches 1e300, and lo * hi overflows.
    p = spec(FaultKind.NUMERICAL_OVERFLOW).param("magnitude")
    v = p.representative(p.n_bins - 1)
    assert math.isfinite(v)
    assert v > 1e200


def test_integer_param_rounds():
    p = spec(FaultKind.BUS_DELAY).param("delay_steps")
    assert p.integer is True
    assert p.sample(0.0) == 1.0
    assert p.sample(1.0) == 8.0
    assert float(p.sample(0.5)).is_integer()


def test_universal_dimensions():
    assert START_FRAC.n_bins == 2
    assert DURATION_FRAC.n_bins == 2
    assert START_FRAC.bin_of(0.0) == 0
    assert START_FRAC.bin_of(0.49) == 0
    assert START_FRAC.bin_of(0.5) == 1
    assert DURATION_FRAC.bin_of(0.05) == 0
    assert DURATION_FRAC.bin_of(1.0) == 1


def test_iter_param_bins_counts():
    assert len(list(iter_param_bins(FaultKind.SENSOR_BIAS))) == 3
    assert list(iter_param_bins(FaultKind.SENSOR_STUCK)) == [()]


def test_paramspec_rejects_bad_construction():
    with pytest.raises(ValueError, match="n_bins"):
        ParamSpec("x", "u", 0.0, 1.0, 0)
    with pytest.raises(ValueError, match="hi > lo"):
        ParamSpec("x", "u", 1.0, 1.0, 2)
    with pytest.raises(ValueError, match="log scale requires"):
        ParamSpec("x", "u", 0.0, 1.0, 2, Scale.LOG)


def test_sample_rejects_u_outside_unit_interval():
    p = spec(FaultKind.SENSOR_BIAS).param("offset")
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        p.sample(1.5)


def test_spec_unknown_kind_lists_valid_names():
    with pytest.raises(KeyError, match="valid kinds"):
        spec("not_a_fault")


def test_validate_rejects_wrong_channel():
    s = spec(FaultKind.SENSOR_BIAS)
    with pytest.raises(ValueError, match="not in"):
        s.validate("u", {"offset": 1.0})


def test_validate_rejects_missing_and_extra_params():
    s = spec(FaultKind.SENSOR_BIAS)
    with pytest.raises(ValueError, match="missing"):
        s.validate("pos", {})
    with pytest.raises(ValueError, match="unexpected"):
        s.validate("pos", {"offset": 1.0, "spin": 2.0})


def test_validate_rejects_out_of_range_and_nonfinite():
    s = spec(FaultKind.SENSOR_BIAS)
    with pytest.raises(ValueError, match="outside declared range"):
        s.validate("pos", {"offset": 1e6})
    with pytest.raises(ValueError, match="finite"):
        s.validate("pos", {"offset": float("nan")})


def test_param_lookup_error():
    with pytest.raises(KeyError, match="no parameter"):
        spec(FaultKind.SENSOR_BIAS).param("nope")
