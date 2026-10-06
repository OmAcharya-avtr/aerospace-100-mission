"""GF(2^m) arithmetic: field axioms, table integrity, input validation.

Exercises REQ-06 of docs/REQUIREMENTS.md.
"""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from codedfade.gf import PRIMITIVE_POLYNOMIALS, GF2m


@pytest.mark.parametrize("m", sorted(PRIMITIVE_POLYNOMIALS))
class TestFieldAxioms:
    def test_exponential_table_is_a_permutation_of_the_non_zero_elements(
        self, m: int
    ) -> None:
        f = GF2m(m)
        n = f.order - 1
        assert sorted(f.exp[:n].tolist()) == list(range(1, f.order))

    def test_log_and_exp_are_inverse(self, m: int) -> None:
        f = GF2m(m)
        for a in range(1, f.order):
            assert int(f.exp[f.log[a]]) == a

    def test_multiplication_is_commutative_and_associative(self, m: int) -> None:
        f = GF2m(m)
        els = np.arange(f.order, dtype=np.int64)
        a, b, c = els[:, None, None], els[None, :, None], els[None, None, :]
        if f.order > 16:  # keep the cube small for m = 5, 6, 8
            els = els[:: max(1, f.order // 8)]
            a, b, c = els[:, None, None], els[None, :, None], els[None, None, :]
        assert np.array_equal(f.mul(a, b), f.mul(b, a))
        assert np.array_equal(f.mul(f.mul(a, b), c), f.mul(a, f.mul(b, c)))

    def test_one_is_the_multiplicative_identity(self, m: int) -> None:
        f = GF2m(m)
        els = np.arange(f.order, dtype=np.int64)
        assert np.array_equal(f.mul(els, 1), els)

    def test_inverse_times_element_is_one(self, m: int) -> None:
        f = GF2m(m)
        els = np.arange(1, f.order, dtype=np.int64)
        assert np.all(f.mul(els, f.inv(els)) == 1)

    def test_distributivity_over_xor(self, m: int) -> None:
        f = GF2m(m)
        els = np.arange(f.order, dtype=np.int64)
        a, b, c = els[:, None], els[None, :], 3 % f.order
        assert np.array_equal(f.mul(a ^ b, c), f.mul(a, c) ^ f.mul(b, c))


class TestValidation:
    def test_rejects_unsupported_m(self) -> None:
        with pytest.raises(ValueError, match="m must be one of"):
            GF2m(7)

    def test_inverse_of_zero_raises(self) -> None:
        with pytest.raises(ZeroDivisionError):
            GF2m(4).inv(0)

    def test_division_by_zero_raises(self) -> None:
        with pytest.raises(ZeroDivisionError):
            GF2m(4).div(5, 0)

    def test_out_of_range_element_raises(self) -> None:
        with pytest.raises(ValueError, match=r"field elements must lie in"):
            GF2m(4).mul(16, 1)

    def test_zero_to_negative_power_raises(self) -> None:
        with pytest.raises(ZeroDivisionError):
            GF2m(4).power(0, -1)

    def test_zero_powers(self) -> None:
        f = GF2m(4)
        assert f.power(0, 0) == 1
        assert f.power(0, 3) == 0


class TestKnownAnswers:
    def test_gf16_multiplication_table_entries(self) -> None:
        """Hand-computed in GF(2^4) with p(x) = x^4 + x + 1 (0b10011).

        alpha = 2. alpha^4 = x^4 = x + 1 = 3, so 2*8 = 16 -> 16 ^ 0b10011 = 3.

        7 * 9: 7 = x^2+x+1, 9 = x^3+1.
          (x^2+x+1)(x^3+1) = x^5+x^4+x^3 + x^2+x+1.
          x^4 = x+1 and x^5 = x*x^4 = x^2+x, so
          = (x^2+x) + (x+1) + x^3 + x^2 + x + 1
          = x^3 + x          (every other term cancels in pairs)
          = 0b1010 = 10.
        Cross-check by logs: 7 = alpha^10, 9 = alpha^14, alpha^24 = alpha^9 = 10.
        """
        f = GF2m(4)
        assert int(f.mul(2, 8)) == 3
        assert int(f.mul(7, 9)) == 10
        assert int(f.mul(1, 13)) == 13
        assert int(f.mul(0, 13)) == 0

    def test_alpha_powers_cycle(self) -> None:
        f = GF2m(4)
        assert f.alpha_power(0) == 1
        assert f.alpha_power(15) == 1
        assert f.alpha_power(-1) == f.alpha_power(14)

    def test_poly_eval_known_answer(self) -> None:
        """p(X) = 1 + 2X + 3X^2 over GF(2^4) at X = 2.

        2*2 = 4; 3*(2^2) = 3*4. 3 = x+1, 4 = x^2, so 3*4 = x^3 + x^2 = 12.
        p(2) = 1 ^ 4 ^ 12 = 9.
        """
        f = GF2m(4)
        assert f.poly_eval(np.array([1, 2, 3]), 2) == 9

    def test_poly_mul_known_answer(self) -> None:
        """(1 + X)(1 + X) = 1 + X^2 in characteristic 2 with coefficients in GF(2)."""
        f = GF2m(4)
        assert f.poly_mul(np.array([1, 1]), np.array([1, 1])).tolist() == [1, 0, 1]


@settings(max_examples=60, deadline=None)
@given(
    m=st.sampled_from(sorted(PRIMITIVE_POLYNOMIALS)),
    a=st.integers(min_value=1, max_value=255),
    b=st.integers(min_value=1, max_value=255),
)
def test_division_inverts_multiplication(m: int, a: int, b: int) -> None:
    f = GF2m(m)
    a %= f.order
    b %= f.order
    if a == 0 or b == 0:
        return
    assert int(f.div(int(f.mul(a, b)), b)) == a
