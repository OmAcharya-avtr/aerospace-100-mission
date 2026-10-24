"""The central claim, asserted sample by sample. Not a mean, not a percentile.

This file exists because the one thing this product promises is the thing most
easily faked by reporting a summary statistic. A mean per-sample error is
satisfied by a reconstruction that is wildly wrong on a single sample, and one
sample is what breaks a downstream limit check. So the assertion here is over
*every* element of *every* test array, in *every* declared mode, with *every*
predictor, at *every* bound in the sweep -- a non-strict inequality with no
tolerance on either side.

The number of samples actually checked is accumulated in
:data:`SAMPLES_CHECKED` and asserted non-trivial at the end of the module's
own run, so a configuration change that silently stopped collecting cases
would fail rather than pass quietly.
"""

from __future__ import annotations

import numpy as np
import pytest

from boundcompress import dataset
from boundcompress.certificate import certify
from boundcompress.codec import decode, encode
from boundcompress.predictors import CLASSICAL_NAMES

#: Declared absolute bounds as fractions of each signal's dynamic range.
FRACTIONS = (1e-2, 1e-3, 1e-4, 1e-5)

#: Declared pointwise relative bounds, for the strictly positive signals.
RELATIVE_BOUNDS = (1e-2, 1e-4, 1e-6)

#: Running count of individual samples whose error was compared to the bound.
SAMPLES_CHECKED: dict[str, int] = {"total": 0}


def _assert_every_sample(
    values: np.ndarray, reconstruction: np.ndarray, mode: str, bound: float
) -> int:
    """Assert the bound on every element. Returns the number of samples checked."""
    original = np.ascontiguousarray(values, dtype=np.float64)
    decoded = np.ascontiguousarray(reconstruction, dtype=np.float64)
    assert decoded.shape == original.shape
    error = np.abs(original - decoded)
    if mode == "rel":
        limit = bound * np.abs(original)
    else:
        limit = np.full(original.shape, bound, dtype=np.float64)
    inside = error <= limit
    if not bool(np.all(inside)):
        bad = int(np.flatnonzero(~inside)[0])
        raise AssertionError(
            f"sample {bad} of {original.size} exceeded the declared {mode} bound "
            f"{bound!r}: value {original[bad]!r}, reconstruction {decoded[bad]!r}, "
            f"error {error[bad]!r}, limit {limit[bad]!r}"
        )
    SAMPLES_CHECKED["total"] += int(original.size)
    return int(original.size)


@pytest.mark.parametrize("predictor", CLASSICAL_NAMES)
@pytest.mark.parametrize("fraction", FRACTIONS)
def test_absolute_bound_holds_on_every_sample_of_every_signal(
    signals: dict[str, np.ndarray], predictor: str, fraction: float
) -> None:
    """mode='abs': every sample of every signal, every predictor, four bounds."""
    checked = 0
    for name, values in signals.items():
        span = float(values.max() - values.min())
        bound = fraction * span
        result = encode(values, "abs", bound, predictor=predictor)
        assert result.certificate.bound_holds, f"{name} {predictor} {bound}"
        assert result.certificate.violations == 0
        checked += _assert_every_sample(values, result.reconstruction, "abs", bound)
        # The decoded array, not only the encoder's own reconstruction.
        checked += _assert_every_sample(values, decode(result.blob), "abs", bound)
    assert checked == 2 * sum(values.size for values in signals.values())


@pytest.mark.parametrize("predictor", CLASSICAL_NAMES)
@pytest.mark.parametrize("bound", RELATIVE_BOUNDS)
def test_pointwise_relative_bound_holds_on_every_sample(
    positive_signals: dict[str, np.ndarray], predictor: str, bound: float
) -> None:
    """mode='rel': the per-sample relative bound, on every sample individually.

    This is the mode where the guarantee is least obvious -- it comes out of
    log-magnitude quantisation and a floating-point ``exp`` -- so it is the
    one where sample-by-sample assertion earns its keep.
    """
    for name, values in positive_signals.items():
        result = encode(values, "rel", bound, predictor=predictor)
        assert result.certificate.bound_holds, f"{name} {predictor} {bound}"
        _assert_every_sample(values, result.reconstruction, "rel", bound)
        _assert_every_sample(values, decode(result.blob), "rel", bound)


@pytest.mark.parametrize("fraction", (1e-2, 1e-4))
def test_range_relative_bound_holds_on_every_sample(
    signals: dict[str, np.ndarray], fraction: float
) -> None:
    """mode='range_rel': the derived absolute bound, on every sample."""
    for values in signals.values():
        span = float(values.max() - values.min())
        result = encode(values, "range_rel", fraction, predictor="linear")
        assert result.certificate.bound_holds
        _assert_every_sample(values, result.reconstruction, "abs", fraction * span)


def test_learned_predictor_bound_holds_on_every_sample(
    signals: dict[str, np.ndarray], learned_linear, learned_mlp
) -> None:
    """The learned predictor cannot widen the bound, and that is asserted, not argued.

    A predictor in a closed-loop DPCM coder affects the rate and nothing else,
    because the residual stream is coded losslessly. The claim is structural;
    the test is here because a structural claim with no test behind it is a
    comment.
    """
    for model in (learned_linear, learned_mlp):
        for values in signals.values():
            span = float(values.max() - values.min())
            bound = 1e-3 * span
            result = encode(values, "abs", bound, predictor="learned", model=model)
            assert result.certificate.bound_holds
            _assert_every_sample(values, result.reconstruction, "abs", bound)
            _assert_every_sample(values, decode(result.blob, model=model), "abs", bound)


def test_certificate_agrees_with_the_independent_sample_sweep(
    signals: dict[str, np.ndarray]
) -> None:
    """The certificate's maximum equals the maximum computed outside the package."""
    for values in signals.values():
        span = float(values.max() - values.min())
        bound = 1e-3 * span
        result = encode(values, "abs", bound, predictor="linear")
        independent = float(np.max(np.abs(values - result.reconstruction)))
        assert result.certificate.max_absolute_error == pytest.approx(
            independent, rel=0.0, abs=0.0
        )
        assert result.certificate.max_absolute_error <= bound
        again = certify(values, result.reconstruction, "abs", bound)
        assert again.bound_holds
        assert again.max_absolute_error == result.certificate.max_absolute_error
        assert again.argmax_index == result.certificate.argmax_index


def test_samples_checked_is_substantial(signals: dict[str, np.ndarray]) -> None:
    """Guard against a configuration that collects the tests but checks nothing.

    A suite that silently stops exercising arrays still passes. The measured
    count for this module at ``TEST_SAMPLES = 1024`` is 360,448 individual
    sample-versus-bound comparisons; the threshold is set at 350,000, close
    enough to catch a parametrisation that quietly shrank and far enough below
    to survive a deliberate change to the signal set.

    This test depends on running after the others in the module, which it does
    under pytest's file-order collection. The dependency is stated rather than
    hidden, and it is why the assertion is a floor and not an equality.
    """
    total = SAMPLES_CHECKED["total"]
    assert total > 350_000, f"only {total} samples were compared to a bound"
    assert len(signals) == len(dataset.SIGNAL_NAMES)
