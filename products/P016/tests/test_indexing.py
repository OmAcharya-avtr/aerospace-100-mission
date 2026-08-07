"""Tests for Noll and OSA/ANSI single-index conventions."""

from __future__ import annotations

import pytest
from hypothesis import given
from hypothesis import strategies as st

from zernkit import (
    mode_name,
    nm_to_noll,
    nm_to_osa,
    noll_to_nm,
    noll_to_osa,
    osa_to_nm,
    osa_to_noll,
    radial_order_from_noll,
    validate_nm,
)

# Known-answer table transcribed from Noll (1976), JOSA 66(3), 207-211, whose
# listed polynomials are:
#   Z1 = 1                      -> (n, m) = (0,  0)
#   Z2 = 2 rho cos(theta)       -> (1, +1)
#   Z3 = 2 rho sin(theta)       -> (1, -1)
#   Z4 = sqrt(3)(2 rho^2 - 1)   -> (2,  0)
#   Z5 = sqrt(6) rho^2 sin(2t)  -> (2, -2)
#   Z6 = sqrt(6) rho^2 cos(2t)  -> (2, +2)
#   Z7 = sqrt(8)(3r^3-2r)sin(t) -> (3, -1)
#   Z8 = sqrt(8)(3r^3-2r)cos(t) -> (3, +1)
#   Z9 = sqrt(8) rho^3 sin(3t)  -> (3, -3)
#   Z10= sqrt(8) rho^3 cos(3t)  -> (3, +3)
#   Z11= sqrt(5)(6r^4-6r^2+1)   -> (4,  0)
#   Z12= sqrt(10)(4r^4-3r^2)cos(2t) -> (4, +2)
#   Z13= sqrt(10)(4r^4-3r^2)sin(2t) -> (4, -2)
#   Z14= sqrt(10) r^4 cos(4t)   -> (4, +4)
#   Z15= sqrt(10) r^4 sin(4t)   -> (4, -4)
NOLL_KNOWN = {
    1: (0, 0),
    2: (1, 1),
    3: (1, -1),
    4: (2, 0),
    5: (2, -2),
    6: (2, 2),
    7: (3, -1),
    8: (3, 1),
    9: (3, -3),
    10: (3, 3),
    11: (4, 0),
    12: (4, 2),
    13: (4, -2),
    14: (4, 4),
    15: (4, -4),
}

# OSA/ANSI: j = (n(n+2) + m)/2, 0-based, m ascending from -n within an order.
OSA_KNOWN = {
    0: (0, 0),
    1: (1, -1),
    2: (1, 1),
    3: (2, -2),
    4: (2, 0),
    5: (2, 2),
    6: (3, -3),
    7: (3, -1),
    8: (3, 1),
    9: (3, 3),
    10: (4, -4),
    11: (4, -2),
    12: (4, 0),
    13: (4, 2),
    14: (4, 4),
}


@pytest.mark.parametrize(("j", "nm"), sorted(NOLL_KNOWN.items()))
def test_noll_to_nm_known_answers(j: int, nm: tuple[int, int]) -> None:
    assert noll_to_nm(j) == nm


@pytest.mark.parametrize(("j", "nm"), sorted(NOLL_KNOWN.items()))
def test_nm_to_noll_known_answers(j: int, nm: tuple[int, int]) -> None:
    assert nm_to_noll(*nm) == j


@pytest.mark.parametrize(("j", "nm"), sorted(OSA_KNOWN.items()))
def test_osa_to_nm_known_answers(j: int, nm: tuple[int, int]) -> None:
    assert osa_to_nm(j) == nm


@pytest.mark.parametrize(("j", "nm"), sorted(OSA_KNOWN.items()))
def test_nm_to_osa_known_answers(j: int, nm: tuple[int, int]) -> None:
    assert nm_to_osa(*nm) == j


def test_osa_closed_form_matches_implementation() -> None:
    # j = (n(n+2) + m)/2 by hand: n=4, m=+2 -> (4*6 + 2)/2 = 26/2 = 13.
    assert nm_to_osa(4, 2) == 13
    assert osa_to_nm(13) == (4, 2)


def test_the_two_conventions_disagree_where_expected() -> None:
    """Noll and OSA agree only at piston; tip/tilt are already swapped."""
    assert noll_to_osa(1) == 0
    assert noll_to_osa(2) == 2  # Noll x-tilt is OSA j=2, not j=1
    assert noll_to_osa(3) == 1
    # Noll j=9 (trefoil, m=-3) maps to OSA j=6, a difference of three places.
    assert noll_to_osa(9) == 6
    disagreements = [j for j in range(1, 40) if noll_to_osa(j) != j - 1]
    assert 2 in disagreements and 3 in disagreements


def test_radial_order_from_noll_block_boundaries() -> None:
    # order n occupies j = n(n+1)/2 + 1 ... (n+1)(n+2)/2
    for n in range(0, 30):
        first = n * (n + 1) // 2 + 1
        last = (n + 1) * (n + 2) // 2
        assert radial_order_from_noll(first) == n
        assert radial_order_from_noll(last) == n


def test_mode_names() -> None:
    assert mode_name(*noll_to_nm(4)) == "defocus"
    assert mode_name(*noll_to_nm(11)) == "primary spherical"
    assert mode_name(6, 2) == "Z(6, 2)"


@pytest.mark.parametrize(
    ("n", "m"),
    [(-1, 0), (2, 3), (3, 0), (4, 1), (2, -3)],
)
def test_validate_nm_rejects_illegal_pairs(n: int, m: int) -> None:
    with pytest.raises(ValueError):
        validate_nm(n, m)


@pytest.mark.parametrize("bad", [1.5, "3", None, True])
def test_validate_nm_type_errors(bad: object) -> None:
    with pytest.raises(TypeError):
        validate_nm(bad, 0)  # type: ignore[arg-type]


def test_index_range_errors() -> None:
    with pytest.raises(ValueError, match="starts at j = 1"):
        noll_to_nm(0)
    with pytest.raises(ValueError, match="starts at j = 0"):
        osa_to_nm(-1)
    with pytest.raises(TypeError):
        noll_to_nm(2.0)  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        osa_to_nm(2.0)  # type: ignore[arg-type]


# --- property-based tests -------------------------------------------------


@given(st.integers(min_value=1, max_value=20000))
def test_noll_roundtrip(j: int) -> None:
    assert nm_to_noll(*noll_to_nm(j)) == j


@given(st.integers(min_value=0, max_value=20000))
def test_osa_roundtrip(j: int) -> None:
    assert nm_to_osa(*osa_to_nm(j)) == j


@given(st.integers(min_value=1, max_value=20000))
def test_noll_osa_noll_roundtrip(j: int) -> None:
    assert osa_to_noll(noll_to_osa(j)) == j


@given(st.integers(min_value=0, max_value=20000))
def test_osa_noll_osa_roundtrip(j: int) -> None:
    assert noll_to_osa(osa_to_noll(j)) == j


@given(st.integers(min_value=1, max_value=5000))
def test_conversions_preserve_nm(j: int) -> None:
    """Both conventions must name the same physical mode."""
    assert noll_to_nm(j) == osa_to_nm(noll_to_osa(j))


@given(st.integers(min_value=1, max_value=5000))
def test_noll_indices_are_a_permutation_within_each_order(j: int) -> None:
    n, m = noll_to_nm(j)
    assert 0 <= abs(m) <= n
    assert (n - abs(m)) % 2 == 0
    # Noll's parity rule: even j -> cosine (m > 0), odd j -> sine (m < 0).
    if m != 0:
        assert (m > 0) == (j % 2 == 0)


def test_noll_order_is_a_bijection_onto_1_to_N() -> None:
    n_max = 12
    total = (n_max + 1) * (n_max + 2) // 2
    seen = {nm_to_noll(n, m) for n in range(n_max + 1) for m in range(-n, n + 1, 2)}
    assert seen == set(range(1, total + 1))


def test_osa_order_is_a_bijection_onto_0_to_N() -> None:
    n_max = 12
    total = (n_max + 1) * (n_max + 2) // 2
    seen = {nm_to_osa(n, m) for n in range(n_max + 1) for m in range(-n, n + 1, 2)}
    assert seen == set(range(total))
