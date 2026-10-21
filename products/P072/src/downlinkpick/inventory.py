"""Recorded-data inventory: what is on the recorder and what it is worth.

Units are stated on every field because the two that are easy to confuse --
bytes on the recorder and bits through the link -- differ by a factor of eight
and the whole product is an accounting exercise.

The value model here is deliberately thin. A data product carries a *declared*
value, which is the number an operator or a science planner wrote down, in
dimensionless value units. Nothing in this package knows what a value unit is;
it only knows that they are additive across products and comparable across
instruments, which is the assumption the whole selection rests on and is stated
in README.md under Limitations.

Expiry is a hard feasibility constraint, not a decay: a product whose expiry has
passed before a contact ends contributes nothing if downlinked in that contact.
This is the behaviour of a product with a downstream deadline (a weather
assimilation cutoff, a tasking-loop deadline, a recorder overwrite), and it is
the reason a selection cannot be made from value density alone.

Dependency is a logical prerequisite: a radiometrically uncalibrated image is
worth nothing without the calibration frame it references, so selecting the
image without the frame yields the image's value and delivers nothing useful.
The constraint implemented is set membership -- if a product is selected, every
product in its transitive dependency closure is selected too. Arrival *order*
is not modelled; see README.md, Limitations.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass, field

BITS_PER_BYTE = 8

#: Lower clamp on an estimated compression ratio, dimensionless. No committed
#: estimator may return a ratio below this: a ratio of zero would make a product
#: free to downlink and would silently break the capacity accounting.
MIN_COMPRESSION_RATIO = 1.0e-3


@dataclass(frozen=True)
class DataProduct:
    """One recorded data product on the spacecraft recorder.

    Parameters
    ----------
    name:
        Unique identifier within an :class:`Inventory`.
    raw_bytes:
        Size on the recorder before compression, in bytes. Strictly positive.
    age_s:
        Seconds between acquisition and the plan epoch. Non-negative.
    expiry_s:
        Seconds after the plan epoch at which the product stops being useful.
        A contact that ends after this time delivers the product too late and it
        contributes zero value. Strictly positive.
    declared_value:
        Value in dimensionless value units as declared by the planner, or
        ``None`` when no declaration exists. ``None`` is the case the learned
        value model of :mod:`downlinkpick.valuemodel` exists to cover.
    compressibility:
        Measured compression ratio, compressed bytes divided by raw bytes, in
        ``(0, 1]``. ``None`` when unmeasured, which is the case the learned
        predictor of :mod:`downlinkpick.compress` exists to cover.
    instrument:
        Instrument or product-class label. Used as a categorical feature and as
        the key of the per-instrument constant-ratio baseline.
    entropy_bits_per_sample:
        First-order entropy of the sample stream in bits per sample, as measured
        onboard by a histogram pass. Non-negative.
    bit_depth:
        Bits per stored sample. Positive integer.
    channels:
        Number of co-registered channels or bands. Positive integer.
    coverage_fraction:
        Fraction of the commanded footprint actually acquired, in ``[0, 1]``.
    novelty:
        Dimensionless novelty score in ``[0, 1]``: 0 for a repeat observation of
        a site already downlinked, 1 for a first look. A feature of the learned
        value model, never of the solvers.
    depends_on:
        Names of products that must also be downlinked for this one to be
        useful. Must not contain this product's own name.
    """

    name: str
    raw_bytes: float
    age_s: float
    expiry_s: float
    declared_value: float | None = None
    compressibility: float | None = None
    instrument: str = "unknown"
    entropy_bits_per_sample: float = 8.0
    bit_depth: int = 16
    channels: int = 1
    coverage_fraction: float = 1.0
    novelty: float = 0.5
    depends_on: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.name or not self.name.strip():
            raise ValueError("DataProduct.name must be a non-empty string")
        if not (self.raw_bytes > 0.0):
            raise ValueError(
                f"{self.name}: raw_bytes must be strictly positive bytes, got {self.raw_bytes!r}"
            )
        if self.age_s < 0.0:
            raise ValueError(f"{self.name}: age_s must be non-negative seconds, got {self.age_s!r}")
        if not (self.expiry_s > 0.0):
            raise ValueError(
                f"{self.name}: expiry_s must be strictly positive seconds after the plan epoch, "
                f"got {self.expiry_s!r}"
            )
        if self.declared_value is not None and self.declared_value < 0.0:
            raise ValueError(
                f"{self.name}: declared_value must be non-negative value units or None, "
                f"got {self.declared_value!r}"
            )
        if self.compressibility is not None and not (
            MIN_COMPRESSION_RATIO <= self.compressibility <= 1.0
        ):
            raise ValueError(
                f"{self.name}: compressibility must lie in "
                f"[{MIN_COMPRESSION_RATIO}, 1.0] (compressed/raw) or be None, "
                f"got {self.compressibility!r}"
            )
        if self.entropy_bits_per_sample < 0.0:
            raise ValueError(
                f"{self.name}: entropy_bits_per_sample must be non-negative, "
                f"got {self.entropy_bits_per_sample!r}"
            )
        if self.bit_depth < 1:
            raise ValueError(f"{self.name}: bit_depth must be >= 1 bit, got {self.bit_depth!r}")
        if self.channels < 1:
            raise ValueError(f"{self.name}: channels must be >= 1, got {self.channels!r}")
        if not (0.0 <= self.coverage_fraction <= 1.0):
            raise ValueError(
                f"{self.name}: coverage_fraction must lie in [0, 1], "
                f"got {self.coverage_fraction!r}"
            )
        if not (0.0 <= self.novelty <= 1.0):
            raise ValueError(f"{self.name}: novelty must lie in [0, 1], got {self.novelty!r}")
        if self.name in self.depends_on:
            raise ValueError(f"{self.name}: a product may not depend on itself")
        if len(set(self.depends_on)) != len(self.depends_on):
            raise ValueError(f"{self.name}: depends_on contains a repeated name")

    @property
    def raw_bits(self) -> float:
        """Recorder size in bits, ``raw_bytes * 8``."""
        return self.raw_bytes * BITS_PER_BYTE

    def downlink_bits(self, ratio: float | None = None) -> float:
        """Bits on the link for a compression ratio, in bits.

        Parameters
        ----------
        ratio:
            Compression ratio (compressed/raw), dimensionless. ``None`` uses
            :attr:`compressibility`, and raises if that is also ``None``.

        Returns
        -------
        float
            ``raw_bytes * 8 * ratio`` in bits.
        """
        effective = self.compressibility if ratio is None else ratio
        if effective is None:
            raise ValueError(
                f"{self.name}: no compression ratio available; pass one explicitly or use "
                "downlinkpick.compress to estimate it"
            )
        if not (MIN_COMPRESSION_RATIO <= effective <= 1.0):
            raise ValueError(
                f"{self.name}: compression ratio must lie in "
                f"[{MIN_COMPRESSION_RATIO}, 1.0], got {effective!r}"
            )
        return self.raw_bits * effective

    @property
    def samples(self) -> float:
        """Sample count implied by ``raw_bytes``, ``bit_depth`` and ``channels``."""
        return self.raw_bits / (self.bit_depth * self.channels)


@dataclass(frozen=True)
class Inventory:
    """An ordered, name-unique, acyclic collection of :class:`DataProduct`.

    The constructor rejects a duplicate name, an unresolved dependency and a
    dependency cycle. All three are silent-wrong-answer defects rather than
    crashes if allowed through: a cycle makes the dependency closure infinite,
    an unresolved name makes the MILP dependency row reference a column that
    does not exist, and a duplicate name makes the no-double-downlink row
    ambiguous.
    """

    products: tuple[DataProduct, ...]
    _index: dict[str, int] = field(init=False, repr=False, compare=False)

    def __init__(self, products: Iterable[DataProduct]) -> None:
        items = tuple(products)
        if not items:
            raise ValueError("Inventory must contain at least one DataProduct")
        names: dict[str, int] = {}
        for position, product in enumerate(items):
            if not isinstance(product, DataProduct):
                raise TypeError(
                    f"Inventory element {position} is {type(product).__name__}, not DataProduct"
                )
            if product.name in names:
                raise ValueError(f"duplicate product name in inventory: {product.name!r}")
            names[product.name] = position
        for product in items:
            for dependency in product.depends_on:
                if dependency not in names:
                    raise ValueError(
                        f"{product.name}: depends_on names {dependency!r}, which is not in the "
                        "inventory"
                    )
        object.__setattr__(self, "products", items)
        object.__setattr__(self, "_index", names)
        self._reject_cycles()

    def _reject_cycles(self) -> None:
        colour: dict[str, int] = {product.name: 0 for product in self.products}

        def visit(name: str, trail: tuple[str, ...]) -> None:
            if colour[name] == 1:
                cycle = " -> ".join((*trail, name))
                raise ValueError(f"dependency cycle in inventory: {cycle}")
            if colour[name] == 2:
                return
            colour[name] = 1
            for dependency in self[name].depends_on:
                visit(dependency, (*trail, name))
            colour[name] = 2

        for product in self.products:
            visit(product.name, ())

    def __len__(self) -> int:
        return len(self.products)

    def __iter__(self) -> Iterator[DataProduct]:
        return iter(self.products)

    def __getitem__(self, name: str) -> DataProduct:
        try:
            return self.products[self._index[name]]
        except KeyError as error:
            raise KeyError(f"no product named {name!r} in inventory") from error

    def __contains__(self, name: object) -> bool:
        return name in self._index

    @property
    def names(self) -> tuple[str, ...]:
        """Product names in inventory order."""
        return tuple(product.name for product in self.products)

    def position(self, name: str) -> int:
        """Index of ``name`` in inventory order."""
        if name not in self._index:
            raise KeyError(f"no product named {name!r} in inventory")
        return self._index[name]

    def dependency_closure(self, name: str) -> frozenset[str]:
        """Transitive dependencies of ``name``, excluding ``name`` itself."""
        seen: set[str] = set()
        stack = list(self[name].depends_on)
        while stack:
            current = stack.pop()
            if current in seen:
                continue
            seen.add(current)
            stack.extend(self[current].depends_on)
        return frozenset(seen)

    def dependents(self, name: str) -> frozenset[str]:
        """Products whose transitive closure contains ``name``."""
        return frozenset(
            product.name
            for product in self.products
            if name in self.dependency_closure(product.name)
        )

    def dependency_pairs(self) -> tuple[tuple[str, str], ...]:
        """Direct ``(product, prerequisite)`` pairs, in inventory order."""
        return tuple(
            (product.name, dependency)
            for product in self.products
            for dependency in product.depends_on
        )

    def topological_order(self) -> tuple[str, ...]:
        """Names ordered so that every prerequisite precedes its dependent."""
        ordered: list[str] = []
        placed: set[str] = set()

        def emit(name: str) -> None:
            if name in placed:
                return
            for dependency in self[name].depends_on:
                emit(dependency)
            placed.add(name)
            ordered.append(name)

        for product in self.products:
            emit(product.name)
        return tuple(ordered)

    def declared_values(self) -> dict[str, float | None]:
        """Declared value per product, ``None`` where undeclared."""
        return {product.name: product.declared_value for product in self.products}

    def missing_declarations(self) -> tuple[str, ...]:
        """Names whose :attr:`DataProduct.declared_value` is ``None``."""
        return tuple(p.name for p in self.products if p.declared_value is None)

    def missing_compressibility(self) -> tuple[str, ...]:
        """Names whose :attr:`DataProduct.compressibility` is ``None``."""
        return tuple(p.name for p in self.products if p.compressibility is None)

    def total_raw_bits(self) -> float:
        """Sum of recorder sizes in bits."""
        return sum(product.raw_bits for product in self.products)

    def total_downlink_bits(self, ratios: Mapping[str, float] | None = None) -> float:
        """Sum of compressed sizes in bits for the given ratios."""
        if ratios is None:
            return sum(product.downlink_bits() for product in self.products)
        return sum(product.downlink_bits(ratios[product.name]) for product in self.products)
