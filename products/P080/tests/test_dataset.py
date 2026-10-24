"""Dataset generator tests. Determinism, shapes, units and the documented properties."""

from __future__ import annotations

import numpy as np
import pytest

from boundcompress import dataset


@pytest.mark.parametrize("name", dataset.SIGNAL_NAMES)
def test_generation_is_deterministic(name: str) -> None:
    first = dataset.generate(name, samples=256, seed=4)
    second = dataset.generate(name, samples=256, seed=4)
    assert np.array_equal(first.values, second.values)
    assert first.values.dtype == np.float64
    assert first.samples == 256
    assert first.unit == dataset.UNITS[name]


@pytest.mark.parametrize("name", dataset.SIGNAL_NAMES)
def test_values_are_finite(name: str) -> None:
    assert np.all(np.isfinite(dataset.generate(name, samples=512).values))


@pytest.mark.parametrize("name", dataset.SIGNAL_NAMES)
def test_different_seeds_give_different_arrays(name: str) -> None:
    """Every signal except the noise-free counter depends on the seed."""
    first = dataset.generate(name, samples=256, seed=0).values
    second = dataset.generate(name, samples=256, seed=1).values
    if name == "packet_counter":
        assert np.array_equal(first, second)
    else:
        assert not np.array_equal(first, second)


@pytest.mark.parametrize("name", dataset.POSITIVE_SIGNALS)
def test_positive_signals_are_strictly_positive(name: str) -> None:
    """These are the signals ``mode='rel'`` accepts, so the property is load-bearing."""
    assert np.all(dataset.generate(name, samples=4096, seed=0).values > 0.0)


def test_wheel_rate_crosses_zero() -> None:
    """Documented as the reason ``mode='rel'`` refuses it."""
    values = dataset.generate("wheel_rate", samples=4096, seed=2).values
    assert values.min() < 0.0 < values.max()


def test_packet_counter_is_a_ramp_with_resets() -> None:
    """Known answer: stride 1024, so the first 1024 samples are 0..1023."""
    values = dataset.generate("packet_counter", samples=2048).values
    assert values[:1024].tolist() == list(range(1024))
    assert values[1024] == 0.0


def test_step_transient_has_twelve_levels() -> None:
    values = dataset.generate("step_transient", samples=2400, seed=7).values
    changes = int(np.count_nonzero(np.abs(np.diff(values)) > 0.1))
    assert changes >= 10


def test_generate_all_covers_every_name_with_offset_seeds() -> None:
    signals = dataset.generate_all(samples=128, seed=50)
    assert set(signals) == set(dataset.SIGNAL_NAMES)
    for index, name in enumerate(dataset.SIGNAL_NAMES):
        assert signals[name].seed == 50 + index


def test_training_signals_are_disjoint_from_the_evaluation_set() -> None:
    """The seed offset of 1000 is what makes the split real; asserted, not assumed."""
    evaluation = dataset.generate_all(samples=256, seed=0)
    training = dataset.training_signals(samples=256)
    for name in dataset.SIGNAL_NAMES:
        if name == "packet_counter":
            continue  # noise-free by construction, so the two realisations coincide
        assert not np.array_equal(evaluation[name].values, training[name].values)


def test_unknown_signal_is_refused() -> None:
    with pytest.raises(dataset.DatasetError, match="unknown signal"):
        dataset.generate("star_tracker_quaternion")


def test_too_few_samples_is_refused() -> None:
    with pytest.raises(dataset.DatasetError, match="at least 8 samples"):
        dataset.generate("gyro_rate", samples=4)


def test_non_positive_rate_is_refused() -> None:
    with pytest.raises(dataset.DatasetError, match="rate_hz must be positive"):
        dataset.generate("gyro_rate", samples=64, rate_hz=0.0)


def test_eclipse_indicator_fraction_matches_the_declared_constant() -> None:
    """Over a whole orbit the indicator should be on for the declared fraction."""
    times = np.arange(int(dataset.ORBIT_PERIOD_S), dtype=np.float64)
    fraction = float(np.mean(dataset._eclipse(times)))
    assert fraction == pytest.approx(dataset.ECLIPSE_FRACTION, abs=1e-3)
