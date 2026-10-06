"""Configuration layer: offsets, sample-instant bookkeeping, and the scalar fast path."""

from __future__ import annotations

import numpy as np
import pytest

from slotsync.ted import (
    TedConfig,
    evaluate_ted,
    evaluate_ted_scalar,
    symbol_variance,
    ted_decision_offsets,
    ted_fractional_positions,
    ted_sample_slots,
    ted_time_offsets,
)

CONFIGS = [
    TedConfig("early-late", "antipodal", 0.25, "dd"),
    TedConfig("early-late", "antipodal", 0.25, "square"),
    TedConfig("early-late", "antipodal", 0.5, "dd"),
    TedConfig("early-late", "ook", 0.25, "auto"),
    TedConfig("early-late", "ook", 0.3, "plain"),
    TedConfig("gardner", "antipodal"),
    TedConfig("gardner", "ook"),
    TedConfig("mueller-muller", "antipodal"),
    TedConfig("mueller-muller", "ook"),
]


@pytest.mark.parametrize("config", CONFIGS, ids=lambda c: f"{c.detector}-{c.alphabet}-{c.form}")
def test_scalar_path_matches_the_vectorised_path_exactly(config: TedConfig) -> None:
    """The closed loop uses the scalar path; this is what stops the two drifting."""
    rng = np.random.default_rng(4711)
    n_taps = len(ted_time_offsets(config))
    n_dec = len(ted_decision_offsets(config))
    samples = rng.standard_normal((200, n_taps))
    decisions = rng.choice([-1.0, 1.0], size=(200, max(n_dec, 1)))[:, :n_dec]
    vector = evaluate_ted(config, samples, decisions if n_dec else None)
    for row in range(200):
        scalar = evaluate_ted_scalar(
            config, tuple(samples[row]), tuple(decisions[row]) if n_dec else ()
        )
        assert scalar == vector[row]


def test_auto_form_resolution() -> None:
    assert TedConfig("early-late", "antipodal").resolved_form == "dd"
    assert TedConfig("early-late", "ook").resolved_form == "square"
    assert TedConfig("early-late", "ook", form="plain").resolved_form == "plain"


def test_time_offsets_and_tap_counts() -> None:
    assert ted_time_offsets(TedConfig("early-late", delta=0.3)) == (-0.3, 0.0, 0.3)
    assert ted_time_offsets(TedConfig("gardner")) == (-1.0, -0.5, 0.0)
    assert ted_time_offsets(TedConfig("mueller-muller")) == (-1.0, 0.0)
    assert TedConfig("mueller-muller").samples_per_symbol == 2
    assert TedConfig("gardner").samples_per_symbol == 3


def test_decision_offsets() -> None:
    assert ted_decision_offsets(TedConfig("mueller-muller")) == (-1, 0)
    assert ted_decision_offsets(TedConfig("early-late", form="dd")) == (0,)
    assert ted_decision_offsets(TedConfig("early-late", form="square")) == ()
    assert ted_decision_offsets(TedConfig("gardner")) == ()


def test_sample_slots_hand_computed() -> None:
    # Gardner's previous strobe and midpoint belong to symbol k-1; the current
    # strobe to symbol k.  Two distinct fractional positions, 0 and 1/2.
    assert ted_fractional_positions(TedConfig("gardner")) == (0.0, 0.5)
    assert ted_sample_slots(TedConfig("gardner")) == ((1, 0), (1, 1), (0, 0))
    # Mueller-Mueller reuses symbol k-1's single strobe.
    assert ted_sample_slots(TedConfig("mueller-muller")) == ((1, 0), (0, 0))
    # Early-late at delta = 1/4 shares nothing: three distinct fractions.
    assert ted_fractional_positions(TedConfig("early-late", delta=0.25)) == (0.0, 0.25, 0.75)
    # At delta = 1/2 the late gate of symbol k is the early gate of symbol k+1.
    assert ted_fractional_positions(TedConfig("early-late", delta=0.5)) == (0.0, 0.5)
    assert ted_sample_slots(TedConfig("early-late", delta=0.5)) == ((1, 1), (0, 0), (0, 1))


def test_symbol_variance_hand_computed() -> None:
    assert symbol_variance("antipodal") == 1.0  # E[(+-1)^2] = 1
    assert symbol_variance("ook") == 0.5  # E[a^2] = (0 + 1) / 2
    with pytest.raises(ValueError, match="unknown alphabet"):
        symbol_variance("qpsk")


def test_configuration_validation() -> None:
    with pytest.raises(ValueError, match="unknown detector"):
        TedConfig("zero-crossing")
    with pytest.raises(ValueError, match="unknown alphabet"):
        TedConfig("gardner", "qam16")
    with pytest.raises(ValueError, match=r"\(0, 0.5\]"):
        TedConfig("early-late", delta=0.75)
    with pytest.raises(ValueError, match=r"\(0, 0.5\]"):
        TedConfig("early-late", delta=0.0)
    with pytest.raises(ValueError, match="unknown early-late form"):
        TedConfig("early-late", form="magic")


def test_missing_decisions_are_rejected_with_an_actionable_message() -> None:
    config = TedConfig("mueller-muller")
    with pytest.raises(ValueError, match="decision-directed"):
        evaluate_ted(config, np.zeros((4, 2)))
    with pytest.raises(ValueError, match="must have 2 entries"):
        evaluate_ted_scalar(config, (0.0, 0.0), (1.0,))


def test_wrong_tap_count_is_rejected() -> None:
    with pytest.raises(ValueError, match="last axis must have length 3"):
        evaluate_ted(TedConfig("gardner"), np.zeros((5, 2)))
    with pytest.raises(ValueError, match="must have 3 entries"):
        evaluate_ted_scalar(TedConfig("gardner"), (0.0, 0.0))


def test_wrong_decision_count_is_rejected() -> None:
    with pytest.raises(ValueError, match="last axis must have length 2"):
        evaluate_ted(TedConfig("mueller-muller"), np.zeros((5, 2)), np.zeros((5, 1)))
