"""Seeded synthetic generator for inventories, contact plans and training tables.

Everything in this module is synthetic and the whole product's learned
components are therefore measured against a generator whose equations are
printed below, not against flight data. This is the central limitation of the
AI claims in this repository and it is stated in MODEL_CARD.md,
DATASET_CARD.md and README.md as well as here. No real spacecraft recorder
inventory is used, and nothing in these equations was fitted to one.

What the generator asserts, and what it therefore cannot test:

* **The value of a data product decays with age at an instrument-dependent
  rate.** Real value decays in ways that depend on what the data is for -- a
  weather product has a hard assimilation deadline and a calibration frame does
  not decay at all -- and the exponential used here is a convenience.
* **A planner's declared value is either a fresh estimate with multiplicative
  noise, or a stale estimate made at acquisition that ignores the decay since.**
  This is the mechanism by which a declaration becomes wrong, and it is the one
  the learned value model is scored against. A real declaration is wrong in
  other ways too: inconsistent units between teams, political weighting, and a
  value scale that is not additive across instruments.
* **Compressibility sits below the memoryless entropy floor by a factor set by a
  spatial-correlation feature.** A real compressor's ratio depends on the codec,
  the tiling and the data, and no codec is run anywhere in this repository.

Generator equations
-------------------
For a product with instrument ``k``, novelty ``nu`` in ``[0, 1]``, coverage
``chi`` in ``[0, 1]``, age ``a`` seconds and latent site weight ``omega``:

    realised value   V = A_k * nu**0.7 * chi * exp(-a / tau_k) * omega          (3)
    fresh declaration  D_fresh = V * (1 + eta),   eta ~ N(0, sigma_d)           (4)
    stale declaration  D_stale = A_k * nu**0.7 * chi * omega * (1 + eta)        (5)

Equation (5) is equation (3) evaluated at ``a = 0``: a declaration written when
the product was recorded and never revisited. ``omega`` is latent -- it is not a
feature of any model here -- so it is the irreducible error floor of every value
estimator in this package.

For compressibility, with first-order entropy ``H`` bits per sample, stored bit
depth ``B`` and lag-one autocorrelation proxy ``rho`` in ``[0, 1]``:

    memoryless floor  f = min(H / B, 1)                                         (6)
    realised ratio    r = clip(f * g_k * (1 - 0.55 * rho) * (1 + zeta), 0.02, 1)(7)

with ``zeta ~ N(0, 0.06)`` and ``g_k`` a per-instrument codec factor. Equation
(6) is the Shannon source-coding bound for a memoryless source (Shannon 1948);
equation (7) deliberately goes *below* it, because a real codec exploits the
correlation that the memoryless bound ignores, which is why the floor is a
biased estimator and why measuring that bias is worth doing.

References
----------
Shannon, C. E. (1948), A mathematical theory of communication, Bell System
Technical Journal 27(3), 379-423. The source-coding bound of equation (6).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from .contacts import ContactPlan, ContactWindow, rate_profile_from_elevation
from .inventory import DataProduct, Inventory

#: Feature names consumed by the learned value model, in column order.
VALUE_FEATURES: tuple[str, ...] = (
    "log10_raw_bytes",
    "age_s",
    "time_to_expiry_s",
    "novelty",
    "coverage_fraction",
    "entropy_bits_per_sample",
    "bit_depth",
    "channels",
    "n_dependents",
    "instrument_code",
)

#: Feature names consumed by the learned compressibility predictor.
COMPRESSION_FEATURES: tuple[str, ...] = (
    "entropy_bits_per_sample",
    "bit_depth",
    "entropy_ratio",
    "lag1_autocorr",
    "log10_raw_bytes",
    "channels",
    "instrument_code",
)


@dataclass(frozen=True)
class InstrumentClass:
    """Generator parameters for one instrument class.

    Attributes
    ----------
    name:
        Label used as :attr:`downlinkpick.inventory.DataProduct.instrument`.
    amplitude:
        ``A_k`` of equation (3), in value units.
    decay_tau_s:
        ``tau_k`` of equation (3), in seconds.
    codec_factor:
        ``g_k`` of equation (7), dimensionless.
    bit_depth, channels:
        Storage format.
    entropy_range:
        Range of ``H`` in bits per sample, sampled uniformly.
    size_range_bytes:
        Range of raw sizes in bytes, sampled log-uniformly.
    """

    name: str
    amplitude: float
    decay_tau_s: float
    codec_factor: float
    bit_depth: int
    channels: int
    entropy_range: tuple[float, float]
    size_range_bytes: tuple[float, float]


#: The five instrument classes the generator draws from. The numbers are chosen
#: to span three orders of magnitude in size and a factor of twelve in decay
#: time, so that an instance has genuine tension between value and size, and
#: between value and deadline. They are not measured from any instrument.
INSTRUMENTS: tuple[InstrumentClass, ...] = (
    InstrumentClass("msi", 9.0, 7.2e4, 0.95, 12, 6, (6.5, 10.5), (4.0e7, 9.0e8)),
    InstrumentClass("sar", 14.0, 2.6e5, 1.05, 16, 2, (11.0, 15.0), (2.0e8, 4.0e9)),
    InstrumentClass("hsi", 20.0, 1.4e5, 0.88, 14, 48, (7.0, 12.0), (6.0e8, 8.0e9)),
    InstrumentClass("hkt", 3.0, 8.6e5, 0.70, 16, 1, (2.0, 6.0), (1.0e5, 4.0e6)),
    InstrumentClass("cal", 6.0, 6.0e6, 0.80, 12, 1, (4.0, 8.0), (2.0e6, 3.0e7)),
)

INSTRUMENT_CODE: dict[str, int] = {
    instrument.name: index for index, instrument in enumerate(INSTRUMENTS)
}


@dataclass(frozen=True)
class GeneratedInstance:
    """One generated selection instance with its ground truth.

    Attributes
    ----------
    inventory:
        The inventory, whose ``declared_value`` and ``compressibility`` are
        ``None`` exactly where the generator withheld them.
    plan:
        The contact plan.
    true_values:
        Realised value per product from equation (3), in value units.
    true_ratios:
        Realised compression ratio per product from equation (7).
    autocorrelation:
        Lag-one autocorrelation proxy per product, the feature the learned
        compressibility predictor uses and the entropy floor cannot.
    stale:
        Names whose declaration was generated by equation (5).
    seed:
        The seed that produced this instance.
    """

    inventory: Inventory
    plan: ContactPlan
    true_values: dict[str, float]
    true_ratios: dict[str, float]
    autocorrelation: dict[str, float]
    stale: tuple[str, ...]
    seed: int


def _sample_products(
    rng: np.random.Generator,
    n_products: int,
    horizon_s: float,
) -> list[dict[str, float | int | str]]:
    rows: list[dict[str, float | int | str]] = []
    for index in range(n_products):
        instrument = INSTRUMENTS[int(rng.integers(len(INSTRUMENTS)))]
        low, high = instrument.size_range_bytes
        raw_bytes = float(np.exp(rng.uniform(np.log(low), np.log(high))))
        entropy = float(rng.uniform(*instrument.entropy_range))
        entropy = min(entropy, float(instrument.bit_depth))
        rows.append(
            {
                "name": f"{instrument.name}-{index:03d}",
                "instrument": instrument.name,
                "raw_bytes": raw_bytes,
                "entropy_bits_per_sample": entropy,
                "bit_depth": instrument.bit_depth,
                "channels": instrument.channels,
                "age_s": float(rng.uniform(0.0, 4.0 * horizon_s)),
                "expiry_s": float(rng.uniform(0.12, 1.6) * horizon_s),
                "novelty": float(rng.beta(2.0, 2.0)),
                "coverage_fraction": float(np.clip(rng.beta(7.0, 1.6), 0.0, 1.0)),
                "lag1_autocorr": float(np.clip(rng.beta(3.0, 2.0), 0.0, 1.0)),
                "site_weight": float(np.exp(rng.normal(0.0, 0.33))),
            }
        )
    return rows


def _true_value(row: Mapping[str, float | int | str]) -> float:
    instrument = INSTRUMENTS[INSTRUMENT_CODE[str(row["instrument"])]]
    return (
        instrument.amplitude
        * float(row["novelty"]) ** 0.7
        * float(row["coverage_fraction"])
        * float(np.exp(-float(row["age_s"]) / instrument.decay_tau_s))
        * float(row["site_weight"])
    )


def _value_at_acquisition(row: Mapping[str, float | int | str]) -> float:
    instrument = INSTRUMENTS[INSTRUMENT_CODE[str(row["instrument"])]]
    return (
        instrument.amplitude
        * float(row["novelty"]) ** 0.7
        * float(row["coverage_fraction"])
        * float(row["site_weight"])
    )


def _true_ratio(row: Mapping[str, float | int | str], rng: np.random.Generator) -> float:
    instrument = INSTRUMENTS[INSTRUMENT_CODE[str(row["instrument"])]]
    floor = min(float(row["entropy_bits_per_sample"]) / float(row["bit_depth"]), 1.0)
    jitter = float(rng.normal(0.0, 0.06))
    ratio = floor * instrument.codec_factor * (1.0 - 0.55 * float(row["lag1_autocorr"]))
    return float(np.clip(ratio * (1.0 + jitter), 0.02, 1.0))


def generate_plan(
    seed: int,
    *,
    n_windows: int = 3,
    horizon_s: float = 4.32e4,
    peak_rate_bps: float = 1.2e8,
) -> ContactPlan:
    """A contact plan of non-overlapping passes with sine rate profiles.

    Windows are placed at evenly spaced starts within ``horizon_s`` with
    jittered durations between 240 and 600 seconds, peak rates jittered around
    ``peak_rate_bps`` by a factor in ``[0.6, 1.4]``, and reliabilities drawn
    uniformly from ``[0.72, 1.0]``. The sine profile is
    :func:`downlinkpick.contacts.rate_profile_from_elevation` and is a stand-in
    for an elevation profile, not a link budget.
    """
    if n_windows < 1:
        raise ValueError(f"n_windows must be >= 1, got {n_windows!r}")
    rng = np.random.default_rng(np.random.SeedSequence([seed, 0xC0FFEE]))
    windows: list[ContactWindow] = []
    stride = horizon_s / n_windows
    for index in range(n_windows):
        duration = float(rng.uniform(240.0, 600.0))
        start = float(index * stride + rng.uniform(0.05, 0.45) * stride)
        peak = peak_rate_bps * float(rng.uniform(0.6, 1.4))
        windows.append(
            ContactWindow(
                name=f"pass{index + 1}",
                start_s=start,
                end_s=start + duration,
                rate_breakpoints=rate_profile_from_elevation(
                    start, start + duration, peak, floor_fraction=0.22, points=9
                ),
                reliability=float(rng.uniform(0.72, 1.0)),
                station=f"gs{index % 3 + 1}",
            )
        )
    return ContactPlan(windows)


def generate_instance(
    seed: int,
    *,
    n_products: int = 24,
    n_windows: int = 3,
    horizon_s: float = 4.32e4,
    peak_rate_bps: float = 1.2e8,
    missing_declared_fraction: float = 0.0,
    stale_declared_fraction: float = 0.0,
    missing_ratio_fraction: float = 0.0,
    declared_noise: float = 0.08,
    dependency_fraction: float = 0.18,
    capacity_fraction: float | None = 0.45,
) -> GeneratedInstance:
    """Generate one instance and its ground truth.

    Parameters
    ----------
    seed:
        Any integer. The same seed and the same keyword arguments always produce
        the identical instance; the generator draws from
        :class:`numpy.random.Generator` seeded through a
        :class:`numpy.random.SeedSequence` so that the product stream and the
        plan stream are independent.
    n_products, n_windows:
        Instance size.
    horizon_s:
        Planning horizon in seconds. Window starts fall inside it and expiries
        are drawn as a fraction of it.
    peak_rate_bps:
        Nominal peak downlink rate in bits per second. Overridden when
        ``capacity_fraction`` is not ``None``.
    capacity_fraction:
        When not ``None``, the peak rate is rescaled so that the plan's total
        capacity equals this fraction of the queue's total compressed size under
        the ground-truth ratios. ``0.45`` means the contact plan can carry about
        45 per cent of what is on the recorder, which is the regime this product
        exists for: the contact is shorter than the queue and something has to be
        left behind. Capacity is linear in the peak rate for the sine profile, so
        the rescaling is exact and not iterative.
    missing_declared_fraction:
        Fraction of products whose ``declared_value`` is withheld as ``None``.
    stale_declared_fraction:
        Fraction of the remaining declarations generated by equation (5)
        instead of equation (4).
    missing_ratio_fraction:
        Fraction of products whose ``compressibility`` is withheld as ``None``.
    declared_noise:
        ``sigma_d`` of equations (4) and (5).
    dependency_fraction:
        Fraction of products given one prerequisite, drawn from the products
        already generated so the dependency graph is acyclic by construction.

    Returns
    -------
    GeneratedInstance
    """
    for label, fraction in (
        ("missing_declared_fraction", missing_declared_fraction),
        ("stale_declared_fraction", stale_declared_fraction),
        ("missing_ratio_fraction", missing_ratio_fraction),
        ("dependency_fraction", dependency_fraction),
    ):
        if not (0.0 <= fraction <= 1.0):
            raise ValueError(f"{label} must lie in [0, 1], got {fraction!r}")
    if n_products < 1:
        raise ValueError(f"n_products must be >= 1, got {n_products!r}")
    if declared_noise < 0.0:
        raise ValueError(f"declared_noise must be non-negative, got {declared_noise!r}")

    rng = np.random.default_rng(np.random.SeedSequence([seed, 0xDEC0DE]))
    rows = _sample_products(rng, n_products, horizon_s)

    true_values = {str(row["name"]): _true_value(row) for row in rows}
    true_ratios = {str(row["name"]): _true_ratio(row, rng) for row in rows}

    order = rng.permutation(n_products)
    n_missing = int(round(missing_declared_fraction * n_products))
    missing = {str(rows[i]["name"]) for i in order[:n_missing]}
    remaining = [i for i in order[n_missing:]]
    n_stale = int(round(stale_declared_fraction * len(remaining)))
    stale = {str(rows[i]["name"]) for i in remaining[:n_stale]}

    order_ratio = rng.permutation(n_products)
    n_missing_ratio = int(round(missing_ratio_fraction * n_products))
    missing_ratio = {str(rows[i]["name"]) for i in order_ratio[:n_missing_ratio]}

    products: list[DataProduct] = []
    for index, row in enumerate(rows):
        name = str(row["name"])
        if name in missing:
            declared: float | None = None
        else:
            base = _value_at_acquisition(row) if name in stale else true_values[name]
            declared = float(max(base * (1.0 + rng.normal(0.0, declared_noise)), 0.0))
        ratio: float | None = None if name in missing_ratio else true_ratios[name]
        depends: tuple[str, ...] = ()
        if index > 0 and rng.random() < dependency_fraction:
            donor = int(rng.integers(index))
            depends = (str(rows[donor]["name"]),)
        products.append(
            DataProduct(
                name=name,
                raw_bytes=float(row["raw_bytes"]),
                age_s=float(row["age_s"]),
                expiry_s=float(row["expiry_s"]),
                declared_value=declared,
                compressibility=ratio,
                instrument=str(row["instrument"]),
                entropy_bits_per_sample=float(row["entropy_bits_per_sample"]),
                bit_depth=int(row["bit_depth"]),
                channels=int(row["channels"]),
                coverage_fraction=float(row["coverage_fraction"]),
                novelty=float(row["novelty"]),
                depends_on=depends,
            )
        )

    inventory = Inventory(products)
    plan = generate_plan(
        seed, n_windows=n_windows, horizon_s=horizon_s, peak_rate_bps=peak_rate_bps
    )
    if capacity_fraction is not None:
        if not (capacity_fraction > 0.0):
            raise ValueError(
                f"capacity_fraction must be strictly positive or None, got {capacity_fraction!r}"
            )
        queue_bits = sum(
            product.raw_bits * true_ratios[product.name] for product in inventory
        )
        scale = capacity_fraction * queue_bits / plan.total_capacity_bits()
        plan = generate_plan(
            seed,
            n_windows=n_windows,
            horizon_s=horizon_s,
            peak_rate_bps=peak_rate_bps * scale,
        )

    return GeneratedInstance(
        inventory=inventory,
        plan=plan,
        true_values=true_values,
        true_ratios=true_ratios,
        autocorrelation={
            str(row["name"]): float(row["lag1_autocorr"]) for row in rows
        },
        stale=tuple(sorted(stale)),
        seed=seed,
    )


def instance_family(
    seeds: Sequence[int],
    **kwargs: object,
) -> tuple[GeneratedInstance, ...]:
    """Generate one instance per seed with identical keyword arguments."""
    return tuple(generate_instance(int(seed), **kwargs) for seed in seeds)  # type: ignore[arg-type]


def value_feature_row(
    product: DataProduct, inventory: Inventory
) -> tuple[float, ...]:
    """Feature row for the value model, in :data:`VALUE_FEATURES` order."""
    return (
        float(np.log10(product.raw_bytes)),
        product.age_s,
        product.expiry_s,
        product.novelty,
        product.coverage_fraction,
        product.entropy_bits_per_sample,
        float(product.bit_depth),
        float(product.channels),
        float(len(inventory.dependents(product.name))),
        float(INSTRUMENT_CODE.get(product.instrument, -1)),
    )


def compression_feature_row(product: DataProduct, lag1_autocorr: float) -> tuple[float, ...]:
    """Feature row for the compressibility model, in :data:`COMPRESSION_FEATURES` order."""
    return (
        product.entropy_bits_per_sample,
        float(product.bit_depth),
        min(product.entropy_bits_per_sample / product.bit_depth, 1.0),
        lag1_autocorr,
        float(np.log10(product.raw_bytes)),
        float(product.channels),
        float(INSTRUMENT_CODE.get(product.instrument, -1)),
    )


@dataclass(frozen=True)
class TrainingTable:
    """Feature matrices and targets for the two learned components.

    Attributes
    ----------
    value_features:
        ``(n, len(VALUE_FEATURES))`` array.
    value_target:
        Realised value per row, equation (3), in value units.
    compression_features:
        ``(n, len(COMPRESSION_FEATURES))`` array.
    compression_target:
        Realised compression ratio per row, equation (7), dimensionless.
    entropy_floor:
        ``f`` of equation (6) per row, the analytic baseline for the ratio.
    instrument:
        Instrument label per row.
    names:
        Product names, unique across the table.
    """

    value_features: np.ndarray
    value_target: np.ndarray
    compression_features: np.ndarray
    compression_target: np.ndarray
    entropy_floor: np.ndarray
    instrument: tuple[str, ...]
    names: tuple[str, ...]

    def __len__(self) -> int:
        return int(self.value_features.shape[0])


def generate_training_table(
    n_rows: int, seed: int, *, horizon_s: float = 4.32e4
) -> TrainingTable:
    """A flat training table drawn from the same equations as the instances.

    Rows are independent -- no dependency graph and no contact plan -- because
    the learned components are per-product regressors and nothing about the
    selection enters their training. This matters for the test-split strategy:
    the models are split by *product*, and the end-to-end benchmark is then run
    on instances built from unseen seeds, so no instance in the benchmark shares
    a product with the training table.
    """
    if n_rows < 1:
        raise ValueError(f"n_rows must be >= 1, got {n_rows!r}")
    rng = np.random.default_rng(np.random.SeedSequence([seed, 0x7AB1E]))
    rows = _sample_products(rng, n_rows, horizon_s)
    value_features = np.empty((n_rows, len(VALUE_FEATURES)))
    compression_features = np.empty((n_rows, len(COMPRESSION_FEATURES)))
    value_target = np.empty(n_rows)
    compression_target = np.empty(n_rows)
    floor = np.empty(n_rows)
    for index, row in enumerate(rows):
        instrument = INSTRUMENTS[INSTRUMENT_CODE[str(row["instrument"])]]
        value_features[index] = (
            float(np.log10(float(row["raw_bytes"]))),
            float(row["age_s"]),
            float(row["expiry_s"]),
            float(row["novelty"]),
            float(row["coverage_fraction"]),
            float(row["entropy_bits_per_sample"]),
            float(row["bit_depth"]),
            float(row["channels"]),
            0.0,
            float(INSTRUMENT_CODE[str(row["instrument"])]),
        )
        entropy_ratio = min(
            float(row["entropy_bits_per_sample"]) / float(row["bit_depth"]), 1.0
        )
        compression_features[index] = (
            float(row["entropy_bits_per_sample"]),
            float(row["bit_depth"]),
            entropy_ratio,
            float(row["lag1_autocorr"]),
            float(np.log10(float(row["raw_bytes"]))),
            float(row["channels"]),
            float(INSTRUMENT_CODE[str(row["instrument"])]),
        )
        value_target[index] = _true_value(row)
        compression_target[index] = _true_ratio(row, rng)
        floor[index] = entropy_ratio
        del instrument
    return TrainingTable(
        value_features=value_features,
        value_target=value_target,
        compression_features=compression_features,
        compression_target=compression_target,
        entropy_floor=floor,
        instrument=tuple(str(row["instrument"]) for row in rows),
        names=tuple(str(row["name"]) for row in rows),
    )
