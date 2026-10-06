"""Coded link: detection model, depth sweep behaviour, decoding shortcut.

Exercises REQ-19, REQ-20, REQ-21, REQ-22 of docs/REQUIREMENTS.md.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy import special

from codedfade.channel import ChannelConfig
from codedfade.link import (
    CodedLink,
    conditional_bit_error_probability,
    q_function,
    uncoded_bit_error_rate,
)
from codedfade.reedsolomon import ReedSolomon


class TestDetectionModel:
    def test_q_function_known_answers(self) -> None:
        """Q(0) = 1/2; Q(1) = 0.158655253931...; Q(2) = 0.0227501319482..."""
        assert q_function(0.0) == pytest.approx(0.5)
        assert q_function(1.0) == pytest.approx(0.15865525393145707, rel=1e-12)
        assert q_function(2.0) == pytest.approx(0.022750131948179195, rel=1e-12)

    def test_q_function_matches_scipy_ndtr(self) -> None:
        x = np.linspace(-4, 4, 17)
        assert np.allclose(q_function(x), 1.0 - special.ndtr(x))

    def test_conditional_bep_known_answer(self) -> None:
        """At gbar = 20 dB (100) and I = 1, p_b = Q(10) = 7.6198530e-24."""
        p = conditional_bit_error_probability(np.array([1.0]), 20.0)
        assert float(p[0]) == pytest.approx(7.619853024160593e-24, rel=1e-9)

    def test_conditional_bep_is_monotone_decreasing_in_irradiance(self) -> None:
        p = conditional_bit_error_probability(np.array([0.2, 0.5, 1.0, 2.0]), 14.0)
        assert np.all(np.diff(p) < 0)

    def test_deep_fade_approaches_one_half(self) -> None:
        p = conditional_bit_error_probability(np.array([1e-9]), 14.0)
        assert float(p[0]) == pytest.approx(0.5, abs=1e-4)

    def test_rejects_negative_irradiance(self) -> None:
        with pytest.raises(ValueError, match="irradiance must be >= 0"):
            conditional_bit_error_probability(np.array([-1.0]), 14.0)


class TestUncodedReference:
    def test_fading_raises_the_average_bit_error_rate(self) -> None:
        """Averaging Q(sqrt(gbar) I) over a fading I is worse than the unfaded
        value, because the Q function is convex in the deep-fade region that
        dominates the average."""
        cfg = ChannelConfig(0.6, 2e-4, 1e6, seed=3)
        faded = uncoded_bit_error_rate(cfg, 14.0, 200_000)
        unfaded = float(conditional_bit_error_probability(np.array([1.0]), 14.0)[0])
        assert faded > 100 * unfaded

    def test_higher_snr_lowers_the_average(self) -> None:
        cfg = ChannelConfig(0.6, 2e-4, 1e6, seed=3)
        assert uncoded_bit_error_rate(cfg, 18.0, 100_000) < uncoded_bit_error_rate(
            cfg, 14.0, 100_000
        )


class TestDepthSweep:
    @staticmethod
    @pytest.fixture(scope="class")
    def link() -> CodedLink:
        cfg = ChannelConfig(0.6, 5.0e-5, 1.0e6, "lognormal", "exp", seed=7)
        return CodedLink(ReedSolomon(31, 21, 5), cfg, mean_snr_db=14.0)

    @staticmethod
    @pytest.fixture(scope="class")
    def sweep(link: CodedLink) -> list:
        return link.depth_sweep([1, 16, 256], 512)

    def test_every_point_sees_the_same_channel_record(self, sweep: list) -> None:
        """The sweep must change only the permutation, so the pre-decoding symbol
        error rate is identical at every depth. Otherwise a depth comparison is
        confounded by a different channel draw."""
        rates = {r.raw_symbol_error_rate for r in sweep}
        assert len(rates) == 1
        counts = {r.codewords for r in sweep}
        assert len(counts) == 1

    def test_deep_interleaving_lowers_the_frame_error_rate(self, sweep: list) -> None:
        assert sweep[-1].frame_error_rate < sweep[0].frame_error_rate

    def test_latency_and_memory_grow_linearly_in_depth(self, sweep: list) -> None:
        assert sweep[1].latency_ms == pytest.approx(16 * sweep[0].latency_ms)
        assert sweep[2].memory_bytes == pytest.approx(256 * sweep[0].memory_bytes)

    def test_failure_and_miscorrection_counts_sum_to_the_frame_errors(
        self, sweep: list
    ) -> None:
        for r in sweep:
            assert r.decoder_failures + r.miscorrections == pytest.approx(
                round(r.frame_error_rate * r.codewords)
            )

    def test_frame_error_rate_standard_error(self, sweep: list) -> None:
        r = sweep[0]
        p = r.frame_error_rate
        assert r.frame_error_rate_standard_error == pytest.approx(
            np.sqrt(p * (1 - p) / r.codewords)
        )

    def test_results_are_reproducible(self, link: CodedLink) -> None:
        a = link.run(64, 256)
        b = link.run(64, 256)
        assert a == b

    @pytest.mark.parametrize(
        "depth,codewords,msg",
        [(0, 10, "depth must be >= 1"), (4, 0, "codewords must be >= 1")],
    )
    def test_rejects_bad_inputs(
        self, link: CodedLink, depth: int, codewords: int, msg: str
    ) -> None:
        with pytest.raises(ValueError, match=msg):
            link.run(depth, codewords)

    def test_rejects_empty_depth_list(self, link: CodedLink) -> None:
        with pytest.raises(ValueError, match="depths must be non-empty"):
            link.depth_sweep([], 10)

    def test_rejects_non_positive_depth_in_sweep(self, link: CodedLink) -> None:
        with pytest.raises(ValueError, match="all depths must be >= 1"):
            link.depth_sweep([1, 0], 10)

    def test_exposes_the_correlation_length(self, link: CodedLink) -> None:
        assert link.samples_per_correlation_time == pytest.approx(50.0)
        assert link.symbol_rate_hz == pytest.approx(1e6)


class TestDecodingShortcut:
    def test_shortcut_and_exact_decoding_agree(self) -> None:
        """The shortcut skips the decoder when the symbol error count is <= t,
        which is sound because Reed-Solomon codes are maximum distance separable.
        This test runs both paths on the identical realisation."""
        cfg = ChannelConfig(0.6, 5.0e-5, 1.0e6, seed=9)
        code = ReedSolomon(31, 21, 5)
        fast = CodedLink(code, cfg, 14.0, exact_decode=False).run(16, 256)
        slow = CodedLink(code, cfg, 14.0, exact_decode=True).run(16, 256)
        assert fast.frame_error_rate == pytest.approx(slow.frame_error_rate)
        assert fast.post_bit_error_rate == pytest.approx(slow.post_bit_error_rate)
        assert fast.decoder_failures == slow.decoder_failures
        assert fast.miscorrections == slow.miscorrections
