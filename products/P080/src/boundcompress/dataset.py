"""Seeded synthetic telemetry generator. No flight data is used anywhere.

Every number in this repository is a statement about the signals defined here
and about nothing else. They are shapes chosen to resemble spacecraft
housekeeping channels -- smooth where real housekeeping is smooth, noisy where
real housekeeping is noisy, discontinuous where a mode change makes it
discontinuous -- and they are not calibrated against any flight dataset. A
compression ratio measured on a sine wave is a statement about sine waves.

Equations
---------

Time is :math:`t_k = k/f_s` seconds for :math:`k = 0 \\ldots n-1`. The orbital
period :math:`T` defaults to 5400 s, a round number in the low-Earth-orbit
range, and the eclipse fraction to 0.35. :math:`g_k` denotes independent
standard normal draws from a seeded ``numpy.random.Generator``.

``battery_voltage`` (volts, strictly positive)
    .. math:: v_k = 27.0 + 1.6 \\sin(2\\pi t_k/T) - 0.9\\,e_k + 0.004 g_k

    where :math:`e_k \\in \\{0,1\\}` is the eclipse indicator. A smooth
    orbital ripple with a load step at eclipse entry and exit, plus a small
    sensor noise floor.

``panel_temperature`` (kelvin, strictly positive)
    First-order thermal relaxation toward an eclipse-driven setpoint,
    integrated with an explicit Euler step:

    .. math:: T_{k+1} = T_k + \\frac{\\Delta t}{\\tau}\\,(S_k - T_k) + 0.01 g_k

    with :math:`S_k = 318` K in sunlight and 243 K in eclipse and
    :math:`\\tau = 600` s. Deliberately the easiest signal in the set: a
    low-pass filter output is close to piecewise linear.

``wheel_rate`` (rad/s, crosses zero)
    Piecewise-linear momentum ramps between random setpoints in
    :math:`[-250, 250]` rad/s with a slew limit of 0.8 rad/s per second, plus
    :math:`0.02 g_k`. Crosses zero, so ``mode="rel"`` refuses it; that
    refusal is itself tested.

``gyro_rate`` (rad/s)
    .. math:: \\omega_k = 0.01\\sin(2\\pi t_k/T) + 0.003\\sin(2\\pi t_k/97) + 0.0012 g_k

    A smooth two-tone with a noise floor at a tenth of the smaller tone. The
    hard-but-realistic case: the noise is what the predictor cannot predict
    and it sets the achievable rate.

``magnetometer`` (microtesla)
    .. math:: b_k = 24\\sin(2\\pi t_k/T) + 7\\sin(4\\pi t_k/T + 1.1) + 0.05 g_k

``packet_counter`` (counts, integer-valued floats)
    A monotone ramp of 1 per sample with a reset to zero at a fixed stride.
    Near-perfectly predicted by ``previous`` and exactly predicted by
    ``linear`` away from the resets, which is why it is in the set: it shows
    what the floor of this coder looks like when prediction works.

``white_noise`` (dimensionless)
    :math:`g_k`. The incompressible control. Any coder that claims a large
    ratio on this is measuring its own header.

``step_transient`` (dimensionless)
    A piecewise-constant signal with eleven level changes and
    :math:`0.001 g_k`. The adversarial case for an extrapolating predictor:
    every step costs one large residual, and a higher-order predictor pays for
    it :math:`k` times over.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

#: Default number of samples per signal. 4096 keeps every measurement in this
#: repository under the three-minute compute budget on two cores.
DEFAULT_SAMPLES = 4096

#: Default sample rate in hertz: one sample per second of mission time.
DEFAULT_RATE_HZ = 1.0

#: Orbital period in seconds, low-Earth-orbit order of magnitude.
ORBIT_PERIOD_S = 5400.0

#: Fraction of the orbit spent in eclipse.
ECLIPSE_FRACTION = 0.35

SIGNAL_NAMES: tuple[str, ...] = (
    "battery_voltage",
    "panel_temperature",
    "wheel_rate",
    "gyro_rate",
    "magnetometer",
    "packet_counter",
    "white_noise",
    "step_transient",
)

#: Signals whose samples are strictly positive, so ``mode="rel"`` applies.
POSITIVE_SIGNALS: tuple[str, ...] = ("battery_voltage", "panel_temperature")

UNITS: dict[str, str] = {
    "battery_voltage": "V",
    "panel_temperature": "K",
    "wheel_rate": "rad/s",
    "gyro_rate": "rad/s",
    "magnetometer": "uT",
    "packet_counter": "counts",
    "white_noise": "1",
    "step_transient": "1",
}


class DatasetError(ValueError):
    """Raised for an unknown signal name or an invalid sample count."""


@dataclass(frozen=True)
class Signal:
    """One generated channel: name, samples, unit, and the seed that made it."""

    name: str
    values: np.ndarray
    unit: str
    seed: int

    @property
    def samples(self) -> int:
        """Number of samples."""
        return int(self.values.size)


def _times(samples: int, rate_hz: float) -> np.ndarray:
    return np.arange(samples, dtype=np.float64) / rate_hz


def _eclipse(times: np.ndarray) -> np.ndarray:
    phase = np.mod(times, ORBIT_PERIOD_S) / ORBIT_PERIOD_S
    return (phase > (1.0 - ECLIPSE_FRACTION)).astype(np.float64)


def generate(
    name: str, *, samples: int = DEFAULT_SAMPLES, seed: int = 0, rate_hz: float = DEFAULT_RATE_HZ
) -> Signal:
    """Generate one named signal deterministically from ``seed``.

    Parameters
    ----------
    name:
        One of :data:`SIGNAL_NAMES`.
    samples:
        Number of samples; must be at least 8 so that every predictor's
        warm-up fits.
    seed:
        Seed for ``numpy.random.default_rng``. The same seed gives the same
        array bit-for-bit, which ``tests/test_dataset.py`` asserts.
    rate_hz:
        Sample rate in hertz, used only to build the time vector.

    Raises
    ------
    DatasetError
        For an unknown name or fewer than 8 samples.
    """
    if name not in SIGNAL_NAMES:
        raise DatasetError(f"unknown signal {name!r}; available: {SIGNAL_NAMES}")
    if samples < 8:
        raise DatasetError(f"need at least 8 samples for the longest warm-up, got {samples}")
    if rate_hz <= 0.0:
        raise DatasetError(f"rate_hz must be positive, got {rate_hz}")

    rng = np.random.default_rng(seed)
    times = _times(samples, rate_hz)
    eclipse = _eclipse(times)

    if name == "battery_voltage":
        values = (
            27.0
            + 1.6 * np.sin(2.0 * np.pi * times / ORBIT_PERIOD_S)
            - 0.9 * eclipse
            + 0.004 * rng.standard_normal(samples)
        )
    elif name == "panel_temperature":
        setpoint = np.where(eclipse > 0.5, 243.0, 318.0)
        noise = 0.01 * rng.standard_normal(samples)
        values = np.empty(samples, dtype=np.float64)
        values[0] = 300.0
        tau = 600.0
        step = 1.0 / rate_hz
        for index in range(1, samples):
            values[index] = (
                values[index - 1]
                + (step / tau) * (setpoint[index - 1] - values[index - 1])
                + noise[index]
            )
    elif name == "wheel_rate":
        values = np.empty(samples, dtype=np.float64)
        values[0] = 0.0
        slew = 0.8 / rate_hz
        target = 0.0
        hold = 0
        noise = 0.02 * rng.standard_normal(samples)
        for index in range(1, samples):
            if hold <= 0:
                target = float(rng.uniform(-250.0, 250.0))
                hold = int(rng.integers(300, 900))
            hold -= 1
            delta = float(np.clip(target - values[index - 1], -slew, slew))
            values[index] = values[index - 1] + delta + noise[index]
    elif name == "gyro_rate":
        values = (
            0.01 * np.sin(2.0 * np.pi * times / ORBIT_PERIOD_S)
            + 0.003 * np.sin(2.0 * np.pi * times / 97.0)
            + 0.0012 * rng.standard_normal(samples)
        )
    elif name == "magnetometer":
        values = (
            24.0 * np.sin(2.0 * np.pi * times / ORBIT_PERIOD_S)
            + 7.0 * np.sin(4.0 * np.pi * times / ORBIT_PERIOD_S + 1.1)
            + 0.05 * rng.standard_normal(samples)
        )
    elif name == "packet_counter":
        stride = 1024
        values = np.mod(np.arange(samples, dtype=np.float64), float(stride))
    elif name == "white_noise":
        values = rng.standard_normal(samples)
    else:  # step_transient
        levels = rng.uniform(-5.0, 5.0, size=12)
        edges = np.linspace(0, samples, 13).astype(int)
        values = np.empty(samples, dtype=np.float64)
        for index in range(12):
            values[edges[index] : edges[index + 1]] = levels[index]
        values = values + 0.001 * rng.standard_normal(samples)

    return Signal(name, np.ascontiguousarray(values, dtype=np.float64), UNITS[name], seed)


def generate_all(
    *, samples: int = DEFAULT_SAMPLES, seed: int = 0, rate_hz: float = DEFAULT_RATE_HZ
) -> dict[str, Signal]:
    """Every signal in :data:`SIGNAL_NAMES`, each from ``seed + its index``.

    Offsetting the seed per signal keeps the noise realisations independent
    while leaving the whole set reproducible from one number.
    """
    return {
        name: generate(name, samples=samples, seed=seed + index, rate_hz=rate_hz)
        for index, name in enumerate(SIGNAL_NAMES)
    }


def training_signals(*, samples: int = 2048, seed: int = 1000) -> dict[str, Signal]:
    """A separate realisation of every signal, for fitting the learned predictor.

    The seed offset of 1000 is what keeps the training and evaluation sets
    disjoint: nothing fitted on these arrays has seen the arrays
    :func:`generate_all` produces at the default seed. The split is by noise
    realisation and not by signal, so the learned predictor is evaluated on
    signal *shapes* it has seen, which is the easier of the two possible
    splits and is stated as such in ``MODEL_CARD.md``.
    """
    return generate_all(samples=samples, seed=seed)
