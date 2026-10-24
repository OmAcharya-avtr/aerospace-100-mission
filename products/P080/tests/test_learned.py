"""Learned predictor tests: determinism, encoder/decoder agreement, uncertainty."""

from __future__ import annotations

import numpy as np
import pytest

from boundcompress import learned
from boundcompress.codec import decode, encode
from boundcompress.quantise import make_spec, quantise


def test_upper_median_known_answers() -> None:
    """Upper median, not the mean of the two central values; exact integers."""
    assert learned.upper_median([1, 2, 3]) == 2
    assert learned.upper_median([1, 2, 3, 4]) == 3
    assert learned.upper_median([5]) == 5
    assert learned.upper_median([0, 0, 0, 0]) == 0


def test_features_from_history_known_answer() -> None:
    """history [0, 1, 3, 6, 10]: diffs [1, 2, 3, 4] -> reversed [4, 3, 2, 1], gain 3.

    Upper median of |[4, 3, 2, 1]| sorted = [1, 2, 3, 4], index 4//2 = 2 -> 3.
    Features are [4/3, 3/3, 2/3, 1/3].
    """
    features, gain = learned.features_from_history([0, 1, 3, 6, 10])
    assert gain == 3
    assert features.tolist() == pytest.approx([4 / 3, 3 / 3, 2 / 3, 1 / 3])


def test_design_matrix_agrees_with_the_sequential_feature_builder() -> None:
    """The vectorised features and gains equal the per-sample ones, row by row."""
    rng = np.random.default_rng(3)
    symbols = np.cumsum(rng.integers(-20, 20, size=200)).astype(np.int64)
    features, gains, targets = design = learned.design_matrix(symbols)
    assert features.shape == (symbols.size - learned.WINDOW, learned.WINDOW - 1)
    for row in range(features.shape[0]):
        history = symbols[row : row + learned.WINDOW].tolist()
        expected_features, expected_gain = learned.features_from_history(history)
        assert int(gains[row]) == expected_gain
        assert np.array_equal(features[row], expected_features)
        assert targets[row] == pytest.approx(
            (int(symbols[row + learned.WINDOW]) - int(symbols[row + learned.WINDOW - 1]))
            / expected_gain
        )
    assert len(design) == 3


def test_unfitted_predictor_refuses_to_predict() -> None:
    model = learned.LearnedPredictor()
    assert model.fitted is False
    with pytest.raises(learned.LearnedError, match="not fitted"):
        model.predict_one([0, 1, 2, 3, 4])


def test_invalid_kind_is_refused() -> None:
    with pytest.raises(learned.LearnedError, match="kind must be one of"):
        learned.LearnedPredictor(kind="transformer")


def test_zero_members_is_refused() -> None:
    with pytest.raises(learned.LearnedError, match="at least one ensemble member"):
        learned.LearnedPredictor(members=0)


def test_fit_with_no_arrays_is_refused() -> None:
    with pytest.raises(learned.LearnedError, match="at least one symbol array"):
        learned.LearnedPredictor().fit([])


def test_fit_with_only_short_arrays_is_refused() -> None:
    with pytest.raises(learned.LearnedError, match="longer than"):
        learned.LearnedPredictor().fit([np.arange(3, dtype=np.int64)])


@pytest.mark.parametrize("fixture_name", ("learned_linear", "learned_mlp"))
def test_batch_prediction_is_bit_identical_to_sequential(
    request: pytest.FixtureRequest, fixture_name: str, signals: dict[str, np.ndarray]
) -> None:
    """The load-bearing determinism property, asserted directly.

    The encoder evaluates the whole array in one batch; the decoder is
    sequential by construction. If the two disagreed by one ulp anywhere, a
    ``round`` would flip and the coder would desynchronise. The forward pass is
    hand-written in elementwise numpy for exactly this reason, and this test is
    what checks that it worked.
    """
    model = request.getfixturevalue(fixture_name)
    values = signals["gyro_rate"]
    spec = make_spec(values, "abs", 1e-5)
    symbols, _, _ = quantise(values, spec)
    batch, dispersion = model.predict_batch(symbols)

    history = symbols[: learned.WINDOW].tolist()
    sequential: list[int] = []
    for index in range(learned.WINDOW, symbols.size):
        sequential.append(model.predict_one(history))
        history.append(int(symbols[index]))
    assert batch.tolist() == sequential
    assert dispersion.shape == batch.shape


@pytest.mark.parametrize("fixture_name", ("learned_linear", "learned_mlp"))
def test_residuals_and_reconstruct_are_inverse(
    request: pytest.FixtureRequest, fixture_name: str, signals: dict[str, np.ndarray]
) -> None:
    model = request.getfixturevalue(fixture_name)
    values = signals["battery_voltage"]
    spec = make_spec(values, "abs", 1e-3)
    symbols, _, _ = quantise(values, spec)
    residuals = model.residuals(symbols)
    assert residuals.size == symbols.size - learned.WINDOW
    rebuilt = model.reconstruct(symbols[: learned.WINDOW].tolist(), residuals)
    assert np.array_equal(rebuilt, symbols)


@pytest.mark.parametrize("fixture_name", ("learned_linear", "learned_mlp"))
def test_codec_round_trip_with_the_learned_predictor(
    request: pytest.FixtureRequest, fixture_name: str, signals: dict[str, np.ndarray]
) -> None:
    model = request.getfixturevalue(fixture_name)
    values = signals["magnetometer"]
    result = encode(values, "abs", 0.01, predictor="learned", model=model)
    assert np.array_equal(decode(result.blob, model=model), result.reconstruction)
    assert result.certificate.bound_holds


def test_refitting_the_same_configuration_gives_the_same_parameters(
    training_symbols: list[np.ndarray]
) -> None:
    """Determinism of the fit itself, not only of prediction."""
    first = learned.LearnedPredictor(kind="linear", members=2, max_rows=2000).fit(
        training_symbols
    )
    second = learned.LearnedPredictor(kind="linear", members=2, max_rows=2000).fit(
        training_symbols
    )
    history = [0, 3, 7, 12, 18]
    assert first.predict_one(history) == second.predict_one(history)
    assert first.parameter_count == second.parameter_count


def test_dispersion_is_non_negative_and_has_the_right_length(
    learned_mlp, signals: dict[str, np.ndarray]
) -> None:
    values = signals["gyro_rate"]
    spec = make_spec(values, "abs", 1e-5)
    symbols, _, _ = quantise(values, spec)
    dispersion = learned_mlp.dispersion(symbols)
    assert dispersion.size == symbols.size - learned.WINDOW
    assert np.all(dispersion >= 0.0)


def test_model_bytes_is_eight_times_the_parameter_count(learned_mlp) -> None:
    assert learned_mlp.model_bytes == 8 * learned_mlp.parameter_count
    assert learned_mlp.parameter_count > 0


def test_single_member_ensemble_has_zero_dispersion(
    training_symbols: list[np.ndarray]
) -> None:
    """One member cannot disagree with itself, so the dispersion must be zero."""
    model = learned.LearnedPredictor(kind="linear", members=1, max_rows=1000).fit(
        training_symbols
    )
    symbols = np.cumsum(np.arange(50)).astype(np.int64)
    assert np.all(model.dispersion(symbols) == 0.0)


def test_warmup_length_is_checked(learned_linear) -> None:
    with pytest.raises(learned.LearnedError, match="exactly"):
        learned_linear.reconstruct([0, 1], np.array([1]))


def test_too_short_array_is_refused(learned_linear) -> None:
    with pytest.raises(learned.LearnedError, match="needs more than"):
        learned_linear.residuals(np.arange(learned.WINDOW, dtype=np.int64))


def test_order_is_the_window(learned_linear) -> None:
    assert learned_linear.order == learned.WINDOW
    assert learned_linear.name == "learned"


def test_clip_changes_the_fit(training_symbols: list[np.ndarray]) -> None:
    """Robust training is load-bearing: the unclipped fit is a different model.

    Measured in the module docstring of boundcompress.learned; asserted here
    only as a difference, because the exact coefficients are a measurement and
    belong in validation output rather than in a test literal.
    """
    clipped = learned.LearnedPredictor(kind="linear", members=1, max_rows=3000).fit(
        training_symbols
    )
    unclipped = learned.LearnedPredictor(
        kind="linear", members=1, max_rows=3000, clip=0.0
    ).fit(training_symbols)
    history = [0, 10, 25, 45, 70]
    assert clipped.predict_one(history) != unclipped.predict_one(history)
