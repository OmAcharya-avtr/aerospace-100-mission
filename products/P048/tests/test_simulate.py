"""End-to-end simulation and demapper-dispatch tests."""

from __future__ import annotations

import numpy as np
import pytest

from softdecode.csi import MultiplicativeCsiError, StaleCsiError
from softdecode.llr import llr_ook_known_csi
from softdecode.metrics import generalised_mutual_information
from softdecode.simulate import DEMAPPERS, decode_ber, demap_ook, simulate_ook


def test_realisation_shapes_and_codewords(code, lognormal, csi_error):
    r = simulate_ook(code, 8.0, lognormal, csi_error, 32, 1)
    assert r.codeword.shape == (32, code.length)
    assert r.messages.shape == (32, code.dimension)
    assert r.h.shape == r.y.shape == r.h_hat.shape == r.codeword.shape
    assert not np.any(code.syndrome(r.codeword))
    assert r.blocks == 32
    assert r.channel_bits == 32 * code.length


def test_seed_reproduces_exactly(code, lognormal, csi_error):
    a = simulate_ook(code, 8.0, lognormal, csi_error, 16, 7)
    b = simulate_ook(code, 8.0, lognormal, csi_error, 16, 7)
    assert np.array_equal(a.y, b.y)
    assert np.array_equal(a.h_hat, b.h_hat)


def test_different_seed_gives_different_data(code, lognormal, csi_error):
    a = simulate_ook(code, 8.0, lognormal, csi_error, 16, 7)
    b = simulate_ook(code, 8.0, lognormal, csi_error, 16, 8)
    assert not np.array_equal(a.y, b.y)


def test_perfect_csi_when_error_is_none(code, lognormal):
    r = simulate_ook(code, 8.0, lognormal, None, 16, 3)
    assert np.array_equal(r.h, r.h_hat)


def test_stale_csi_path_runs(code, lognormal):
    r = simulate_ook(code, 8.0, lognormal, StaleCsiError(0.9), 16, 3)
    assert np.all(r.h_hat > 0.0)
    llr = demap_ook(r, "csi_aware_exact", lognormal, StaleCsiError(0.9))
    assert llr.shape == r.codeword.shape


def test_amplitude_matches_rate_convention(code, lognormal, csi_error, detection):
    r = simulate_ook(code, 6.0, lognormal, csi_error, 4, 1, detection)
    assert r.amplitude == pytest.approx(detection.ook_amplitude(6.0, code.rate))
    assert r.rate == code.rate


def test_all_demappers_produce_finite_llrs(code, lognormal, csi_error):
    r = simulate_ook(code, 8.0, lognormal, csi_error, 24, 5)
    for method in DEMAPPERS:
        llr = demap_ook(r, method, lognormal, csi_error)
        assert llr.shape == r.codeword.shape
        assert np.all(np.isfinite(llr))


def test_plugin_equals_known_csi_with_perfect_estimate(code, lognormal):
    r = simulate_ook(code, 8.0, lognormal, None, 16, 5)
    assert np.allclose(
        demap_ook(r, "plugin", lognormal),
        llr_ook_known_csi(r.y, r.amplitude, r.h, r.sigma),
    )


def test_known_csi_is_the_best_demapper(code, lognormal, csi_error):
    r = simulate_ook(code, 8.0, lognormal, csi_error, 400, 5)
    gmis = {
        m: generalised_mutual_information(demap_ook(r, m, lognormal, csi_error), r.codeword)
        for m in DEMAPPERS
    }
    assert gmis["known_csi"] == max(gmis.values())
    # With the error model known, accounting for it beats ignoring it.
    assert gmis["csi_aware_exact"] > gmis["plugin"]
    assert gmis["csi_aware_exact"] > gmis["no_csi_exact"]


def test_scale_and_clip_are_applied_in_order(code, lognormal, csi_error):
    r = simulate_ook(code, 8.0, lognormal, csi_error, 8, 5)
    base = demap_ook(r, "known_csi", lognormal, csi_error)
    got = demap_ook(r, "known_csi", lognormal, csi_error, scale=0.5, clip=1.0)
    assert np.allclose(got, np.clip(base * 0.5, -1.0, 1.0))


def test_decode_ber_is_lower_than_raw(code, lognormal, csi_error):
    r = simulate_ook(code, 7.0, lognormal, None, 300, 5)
    llr = demap_ook(r, "known_csi", lognormal)
    raw = float(np.mean((llr < 0) != r.codeword))
    assert decode_ber(code, r, llr).rate < raw


def test_unknown_demapper_raises(code, lognormal, csi_error):
    r = simulate_ook(code, 8.0, lognormal, csi_error, 4, 5)
    with pytest.raises(ValueError, match="unknown demapper"):
        demap_ook(r, "nonsense", lognormal, csi_error)
    with pytest.raises(ValueError, match="requires an error model"):
        demap_ook(r, "csi_aware_exact", lognormal, None)


def test_over_estimation_hurts_more_than_under_estimation(code, lognormal):
    # The headline asymmetry, pinned as a test at one configuration so a
    # refactor cannot silently reverse it.
    over = MultiplicativeCsiError(+2.0, 0.0)
    under = MultiplicativeCsiError(-2.0, 0.0)
    results = {}
    for name, error in (("over", over), ("under", under)):
        r = simulate_ook(code, 8.0, lognormal, error, 400, 20261006)
        llr = demap_ook(r, "plugin", lognormal, error)
        results[name] = decode_ber(code, r, llr).rate
    assert results["over"] > 3.0 * results["under"]
