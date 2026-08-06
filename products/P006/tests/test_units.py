"""Unit-conversion tests: known answers and Hypothesis round trips."""

import math

import pytest
from hypothesis import given
from hypothesis import strategies as st

from linkbudgetx import (
    db_to_linear,
    dbm_to_watts,
    km_to_m,
    linear_to_db,
    m_to_km,
    m_to_nm,
    nm_to_m,
    watts_to_dbm,
)


class TestKnownAnswers:
    def test_dbm_anchors(self):
        # 0 dBm = 1 mW, 30 dBm = 1 W, -30 dBm = 1 uW (definition of dBm).
        assert dbm_to_watts(0.0) == pytest.approx(1e-3)
        assert dbm_to_watts(30.0) == pytest.approx(1.0)
        assert dbm_to_watts(-30.0) == pytest.approx(1e-6)
        assert watts_to_dbm(1e-3) == pytest.approx(0.0, abs=1e-12)

    def test_db_anchors(self):
        # 3.0103 dB = factor 2: 10*log10(2) = 3.0102999...
        assert linear_to_db(2.0) == pytest.approx(10.0 * math.log10(2.0))
        assert db_to_linear(10.0) == pytest.approx(10.0)
        assert linear_to_db(1.0) == 0.0

    def test_length_anchors(self):
        assert nm_to_m(1550.0) == pytest.approx(1.55e-6)
        assert km_to_m(10.0) == pytest.approx(10_000.0)


class TestRoundTrips:
    @given(st.floats(min_value=-80.0, max_value=80.0))
    def test_dbm_watts_round_trip(self, p_dbm):
        assert watts_to_dbm(dbm_to_watts(p_dbm)) == pytest.approx(p_dbm, abs=1e-9)

    @given(st.floats(min_value=1e-12, max_value=1e6))
    def test_watts_dbm_round_trip(self, p_w):
        assert dbm_to_watts(watts_to_dbm(p_w)) == pytest.approx(p_w, rel=1e-9)

    @given(st.floats(min_value=-100.0, max_value=100.0))
    def test_db_linear_round_trip(self, x_db):
        assert linear_to_db(db_to_linear(x_db)) == pytest.approx(x_db, abs=1e-9)

    @given(st.floats(min_value=1e-6, max_value=1e9))
    def test_length_round_trips(self, x):
        assert m_to_nm(nm_to_m(x)) == pytest.approx(x, rel=1e-12)
        assert m_to_km(km_to_m(x)) == pytest.approx(x, rel=1e-12)


class TestInvalid:
    @pytest.mark.parametrize("bad", [0.0, -1.0])
    def test_nonpositive_power_raises(self, bad):
        with pytest.raises(ValueError):
            watts_to_dbm(bad)

    @pytest.mark.parametrize("bad", [0.0, -0.5])
    def test_nonpositive_ratio_raises(self, bad):
        with pytest.raises(ValueError):
            linear_to_db(bad)
