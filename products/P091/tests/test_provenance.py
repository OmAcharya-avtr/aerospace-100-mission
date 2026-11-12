"""The origin lattice and the one-way rule that a synthetic figure is not measured."""

from __future__ import annotations

import itertools

import pytest

from thermalbudget.provenance import (
    NotMeasuredError,
    Origin,
    label,
    meet,
    require_measured,
)

ALL = (Origin.SYNTHETIC, Origin.MEASURED)


def test_meet_is_exhaustively_the_minimum_over_two_elements():
    # Four assignments, all of them, by hand:
    assert meet(Origin.MEASURED, Origin.MEASURED) is Origin.MEASURED
    assert meet(Origin.MEASURED, Origin.SYNTHETIC) is Origin.SYNTHETIC
    assert meet(Origin.SYNTHETIC, Origin.MEASURED) is Origin.SYNTHETIC
    assert meet(Origin.SYNTHETIC, Origin.SYNTHETIC) is Origin.SYNTHETIC


@pytest.mark.parametrize("a,b,c", list(itertools.product(ALL, repeat=3)))
def test_meet_is_associative_and_commutative(a, b, c):
    assert meet(meet(a, b), c) is meet(a, meet(b, c))
    assert meet(a, b) is meet(b, a)


@pytest.mark.parametrize("origin", ALL)
def test_identity_and_absorbing_element(origin):
    assert meet(origin, Origin.MEASURED) is origin
    assert meet(origin, Origin.SYNTHETIC) is Origin.SYNTHETIC
    assert meet(origin, origin) is origin


def test_meet_accepts_iterables_and_mixtures():
    assert meet([Origin.MEASURED, Origin.MEASURED]) is Origin.MEASURED
    assert meet(Origin.MEASURED, [Origin.MEASURED, Origin.SYNTHETIC]) is Origin.SYNTHETIC


def test_meet_of_nothing_is_the_identity_which_is_why_require_measured_exists():
    # The meet of an empty set is MEASURED. That is correct lattice algebra and
    # is exactly the trap: a check run over no files would be labelled measured.
    assert meet() is Origin.MEASURED
    with pytest.raises(NotMeasuredError, match="empty input set"):
        require_measured(meet(), "a figure over no inputs", n_inputs=0)


def test_meet_rejects_a_non_origin():
    with pytest.raises(TypeError, match="Origin"):
        meet("MEASURED")
    with pytest.raises(TypeError, match="Origin"):
        meet([1.0])


def test_require_measured_passes_a_measured_origin_through():
    assert require_measured(Origin.MEASURED, "x", n_inputs=3) is Origin.MEASURED


def test_require_measured_refuses_synthetic_by_name():
    with pytest.raises(NotMeasuredError, match="synthetic input"):
        require_measured(Origin.SYNTHETIC, "the headline", n_inputs=3)


def test_there_is_no_function_that_promotes_synthetic_to_measured():
    # The guarantee is structural: nothing in the module returns MEASURED when
    # given SYNTHETIC. Checked over every public callable that takes an origin.
    for n_inputs in (1, 5, 100):
        assert meet(Origin.SYNTHETIC, Origin.MEASURED) is Origin.SYNTHETIC
        with pytest.raises(NotMeasuredError):
            require_measured(Origin.SYNTHETIC, "x", n_inputs=n_inputs)


def test_labels_distinguish_the_two_origins_in_words():
    assert "MEASURED" in label(Origin.MEASURED)
    assert "MAXN_SUPER" in label(Origin.MEASURED)
    assert "not a measurement" in label(Origin.SYNTHETIC)


def test_rank_orders_synthetic_below_measured():
    assert Origin.SYNTHETIC.rank < Origin.MEASURED.rank
    assert str(Origin.MEASURED) == "MEASURED"
