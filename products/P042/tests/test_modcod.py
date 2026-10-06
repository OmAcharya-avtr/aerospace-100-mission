"""Tests for the MODCOD ladder, the threshold measurement and serialisation."""

from __future__ import annotations

import json

import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from acmpilot.coding import ReedSolomonCode
from acmpilot.modcod import (
    DEFAULT_TARGET_BER,
    Modcod,
    ModcodTable,
    measure_ber_curve,
    measure_thresholds,
    shipped_modcods,
)


class TestModcod:
    def test_known_answer_spectral_efficiency(self):
        # QPSK carries 2 bits/symbol; RS(255,223) has rate 223/255 = 0.874509...,
        # so eta = 2 * 223/255 = 446/255 = 1.749019607843137 by hand.
        mode = Modcod(name="x", modulation="QPSK", code=ReedSolomonCode(255, 223))
        assert mode.bits_per_symbol == 2
        assert mode.spectral_efficiency == pytest.approx(446 / 255, rel=1e-15)

    def test_shipped_ladder_has_eight_modes(self):
        assert len(shipped_modcods()) == 8

    def test_shipped_ladder_sorted_by_efficiency(self):
        eta = [m.spectral_efficiency for m in shipped_modcods()]
        assert eta == sorted(eta)
        assert len(set(eta)) == len(eta)

    def test_shipped_ladder_spans_four_constellations(self):
        assert {m.modulation for m in shipped_modcods()} == {
            "BPSK", "QPSK", "8PSK", "16QAM"
        }

    def test_shipped_ladder_efficiency_endpoints(self):
        ladder = shipped_modcods()
        assert ladder[0].spectral_efficiency == pytest.approx(127 / 255, rel=1e-15)
        assert ladder[-1].spectral_efficiency == pytest.approx(4 * 239 / 255, rel=1e-15)


class TestModcodTable:
    def test_known_answer_best_supported(self, tiny_table):
        # Thresholds are 3, 5, 7 dB by construction.
        assert int(tiny_table.best_supported(2.9)) == -1
        assert int(tiny_table.best_supported(3.0)) == 0
        assert int(tiny_table.best_supported(4.9)) == 0
        assert int(tiny_table.best_supported(5.0)) == 1
        assert int(tiny_table.best_supported(6.9)) == 1
        assert int(tiny_table.best_supported(7.0)) == 2
        assert int(tiny_table.best_supported(100.0)) == 2

    def test_best_supported_vectorised(self, tiny_table):
        out = tiny_table.best_supported(np.array([0.0, 4.0, 6.0, 8.0]))
        np.testing.assert_array_equal(out, np.array([-1, 0, 1, 2]))

    def test_best_supported_scalar_shape(self, tiny_table):
        assert np.shape(tiny_table.best_supported(6.0)) == ()

    def test_properties(self, tiny_table):
        assert tiny_table.n_modes == 3
        assert tiny_table.is_monotone
        np.testing.assert_allclose(
            tiny_table.spectral_efficiencies,
            np.array([1.0, 2.0, 3.0]) * (253 / 255),
            rtol=1e-12,
        )

    def test_non_monotone_detected(self, tiny_table):
        broken = ModcodTable(
            modcods=tiny_table.modcods,
            thresholds_db=np.array([3.0, 2.0, 7.0]),
            threshold_sigma_db=np.zeros(3),
            target_ber=1e-6,
        )
        assert not broken.is_monotone

    def test_rejects_length_mismatch(self, tiny_table):
        with pytest.raises(ValueError, match="thresholds"):
            ModcodTable(
                modcods=tiny_table.modcods,
                thresholds_db=np.array([1.0, 2.0]),
                threshold_sigma_db=np.zeros(2),
                target_ber=1e-6,
            )

    def test_rejects_empty_table(self):
        with pytest.raises(ValueError, match="at least one mode"):
            ModcodTable(
                modcods=(),
                thresholds_db=np.array([]),
                threshold_sigma_db=np.array([]),
                target_ber=1e-6,
            )

    def test_round_trip_dict(self, tiny_table):
        restored = ModcodTable.from_dict(tiny_table.to_dict())
        np.testing.assert_allclose(restored.thresholds_db, tiny_table.thresholds_db)
        assert [m.name for m in restored.modcods] == [m.name for m in tiny_table.modcods]
        assert restored.target_ber == tiny_table.target_ber

    def test_round_trip_json_file(self, tiny_table, tmp_path):
        path = tmp_path / "table.json"
        tiny_table.save_json(path)
        restored = ModcodTable.load_json(path)
        np.testing.assert_allclose(restored.thresholds_db, tiny_table.thresholds_db)
        np.testing.assert_allclose(
            restored.spectral_efficiencies, tiny_table.spectral_efficiencies
        )

    def test_serialised_form_contains_no_filesystem_path(self, tiny_table, tmp_path):
        path = tmp_path / "table.json"
        tiny_table.save_json(path)
        text = path.read_text()
        assert "/home/" not in text
        assert "/Users/" not in text
        payload = json.loads(text)
        assert set(payload) == {"target_ber", "provenance", "modcods"}

    def test_serialised_rows_carry_every_field(self, tiny_table):
        row = tiny_table.to_dict()["modcods"][0]
        for key in (
            "name", "modulation", "code_n", "code_k", "code_rate", "code_t",
            "bits_per_symbol", "spectral_efficiency_bit_per_symbol",
            "threshold_db", "threshold_sigma_db",
        ):
            assert key in row

    @given(snr=st.floats(min_value=-20.0, max_value=40.0))
    @settings(
        max_examples=40, deadline=None,
        suppress_health_check=[HealthCheck.function_scoped_fixture],
    )
    def test_property_best_supported_monotone(self, tiny_table, snr):
        lower = int(tiny_table.best_supported(snr))
        upper = int(tiny_table.best_supported(snr + 1.0))
        assert upper >= lower


class TestMeasureBerCurve:
    def test_curve_fields_and_lengths(self, rng):
        curve = measure_ber_curve(
            "QPSK", rng=rng, snr_start_db=0.0, snr_step_db=1.0,
            stop_below_ber=1e-2, target_errors=400, max_symbols=40_000,
        )
        n = curve["snr_db"].size
        assert n > 1
        for key in ("ber", "n_bits", "n_errors", "ber_se"):
            assert curve[key].size == n

    def test_curve_is_decreasing_and_terminates_below_threshold(self, rng):
        curve = measure_ber_curve(
            "BPSK", rng=rng, snr_start_db=0.0, snr_step_db=0.5,
            stop_below_ber=1e-3, target_errors=600, max_symbols=200_000,
        )
        assert curve["ber"][-1] < 1e-3
        # Monotone up to Monte Carlo noise.
        assert curve["ber"][0] > curve["ber"][-1]

    def test_standard_error_consistent_with_binomial(self, rng):
        curve = measure_ber_curve(
            "QPSK", rng=rng, snr_start_db=2.0, snr_step_db=1.0,
            stop_below_ber=1e-2, target_errors=400, max_symbols=40_000,
        )
        expected = np.sqrt(curve["ber"] * (1 - curve["ber"]) / curve["n_bits"])
        np.testing.assert_allclose(curve["ber_se"], expected, rtol=1e-12)

    def test_rejects_non_positive_step(self, rng):
        with pytest.raises(ValueError, match="snr_step_db"):
            measure_ber_curve("BPSK", rng=rng, snr_step_db=0.0)


class TestMeasureThresholds:
    def test_default_target_ber(self):
        assert DEFAULT_TARGET_BER == 1e-6

    def test_table_is_monotone(self, table):
        assert table.is_monotone

    def test_eight_thresholds_all_finite(self, table):
        assert table.n_modes == 8
        assert np.all(np.isfinite(table.thresholds_db))
        assert np.all(np.isfinite(table.threshold_sigma_db))

    def test_uncertainties_are_small_and_positive(self, table):
        assert np.all(table.threshold_sigma_db > 0.0)
        assert float(np.max(table.threshold_sigma_db)) < 0.2

    def test_no_dominated_modcod(self, table):
        eta = table.spectral_efficiencies
        thr = table.thresholds_db
        for i in range(table.n_modes):
            for j in range(table.n_modes):
                if i == j:
                    continue
                assert not (thr[i] >= thr[j] and eta[i] <= eta[j])

    def test_thresholds_in_plausible_range(self, table):
        assert 0.0 < float(table.thresholds_db[0]) < 8.0
        assert 10.0 < float(table.thresholds_db[-1]) < 25.0

    def test_reproducible_from_seed(self):
        a, _ = measure_thresholds(
            modcods=shipped_modcods()[:2], seed=5, snr_step_db=1.0,
            stop_below_ber=3e-3, target_errors=300, max_symbols=40_000,
        )
        b, _ = measure_thresholds(
            modcods=shipped_modcods()[:2], seed=5, snr_step_db=1.0,
            stop_below_ber=3e-3, target_errors=300, max_symbols=40_000,
        )
        np.testing.assert_allclose(a.thresholds_db, b.thresholds_db)

    def test_reuses_supplied_curves(self, table, curves):
        again, same = measure_thresholds(curves=curves)
        np.testing.assert_allclose(again.thresholds_db, table.thresholds_db)
        assert same is curves

    def test_lower_target_ber_needs_more_snr(self, curves):
        loose, _ = measure_thresholds(target_ber=1e-4, curves=curves)
        tight, _ = measure_thresholds(target_ber=1e-8, curves=curves)
        assert np.all(tight.thresholds_db >= loose.thresholds_db)

    def test_provenance_recorded_and_path_free(self, table):
        assert "Monte Carlo" in table.provenance
        assert "/home/" not in table.provenance
        assert "/Users/" not in table.provenance

    @pytest.mark.parametrize("bad", [0.0, 1.0, -1e-6, 2.0])
    def test_rejects_invalid_target_ber(self, bad, curves):
        with pytest.raises(ValueError, match="target_ber"):
            measure_thresholds(target_ber=bad, curves=curves)


class TestRegressionAgainstCommittedTable:
    def test_measured_thresholds_match_committed_within_tenth_of_a_db(
        self, table, committed_table
    ):
        """Regression test: re-measuring must reproduce the committed table.

        ``validation/modcod_thresholds.json`` is produced by
        ``validation/validate_modcod_thresholds.py`` with the same default seed,
        so a drift here means either the measurement or the accounting changed.
        """
        if committed_table is None:
            pytest.skip("validation/modcod_thresholds.json is not present")
        assert committed_table.n_modes == table.n_modes
        assert committed_table.target_ber == table.target_ber
        np.testing.assert_allclose(
            committed_table.thresholds_db, table.thresholds_db, atol=0.1
        )

    def test_committed_names_and_efficiencies_match(self, table, committed_table):
        if committed_table is None:
            pytest.skip("validation/modcod_thresholds.json is not present")
        assert [m.name for m in committed_table.modcods] == [
            m.name for m in table.modcods
        ]
        np.testing.assert_allclose(
            committed_table.spectral_efficiencies, table.spectral_efficiencies, rtol=1e-12
        )
