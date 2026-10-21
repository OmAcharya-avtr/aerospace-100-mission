"""YAML case files: one inventory and one contact plan in one document.

A case file is the input a user actually has: a list of what is on the recorder
and a list of the passes. The schema is flat and every key maps to a field of
:class:`downlinkpick.inventory.DataProduct` or
:class:`downlinkpick.contacts.ContactWindow`, so there is nothing to learn
beyond those two docstrings.

Paths are resolved by the caller. Nothing in this module touches the working
directory, and nothing in this package anchors a path on it: the examples and
validation scripts anchor on ``__file__`` and change to the repository root
before they load anything, which is why they run identically from any directory.

Example
-------
.. code-block:: yaml

    name: tight-pass
    description: one short pass, three products, two of them expiring
    products:
      - name: msi-001
        raw_bytes: 4.0e8
        age_s: 5400.0
        expiry_s: 30000.0
        declared_value: 8.4
        compressibility: 0.31
        instrument: msi
        entropy_bits_per_sample: 8.1
        bit_depth: 12
        channels: 6
        coverage_fraction: 0.94
        novelty: 0.71
    windows:
      - name: pass1
        station: gs1
        start_s: 1200.0
        end_s: 1680.0
        reliability: 0.93
        rate_breakpoints:
          - [1200.0, 2.4e7]
          - [1440.0, 1.1e8]
          - [1680.0, 2.4e7]
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .contacts import ContactPlan, ContactWindow
from .inventory import DataProduct, Inventory

PRODUCT_KEYS = frozenset(
    {
        "name",
        "raw_bytes",
        "age_s",
        "expiry_s",
        "declared_value",
        "compressibility",
        "instrument",
        "entropy_bits_per_sample",
        "bit_depth",
        "channels",
        "coverage_fraction",
        "novelty",
        "depends_on",
    }
)

WINDOW_KEYS = frozenset(
    {"name", "start_s", "end_s", "rate_breakpoints", "reliability", "station"}
)

#: Lag-one autocorrelation key accepted per product. Optional: it is a feature
#: of the learned compressibility predictor and not a field of DataProduct,
#: because a product that has the statistic and a product that does not are
#: different situations.
AUTOCORRELATION_KEY = "lag1_autocorr"


@dataclass(frozen=True)
class Case:
    """A loaded case file.

    Attributes
    ----------
    name:
        Case identifier from the file.
    description:
        Free text from the file, or the empty string.
    inventory:
        The recorded-data inventory.
    plan:
        The contact plan.
    autocorrelation:
        Lag-one autocorrelation proxy per product for the products that
        declared one. Products that did not are absent from the mapping.
    source:
        The path the case was read from, as given by the caller.
    """

    name: str
    description: str
    inventory: Inventory
    plan: ContactPlan
    autocorrelation: dict[str, float]
    source: str


def _require_mapping(node: Any, what: str) -> dict[str, Any]:
    if not isinstance(node, dict):
        raise ValueError(f"{what} must be a mapping, got {type(node).__name__}")
    return node


def product_from_mapping(node: Any) -> tuple[DataProduct, float | None]:
    """A :class:`DataProduct` and its optional autocorrelation from one mapping."""
    mapping = _require_mapping(node, "a product entry")
    unknown = sorted(set(mapping) - PRODUCT_KEYS - {AUTOCORRELATION_KEY})
    if unknown:
        raise ValueError(
            "unknown product key(s): "
            + ", ".join(unknown)
            + "; expected a subset of "
            + ", ".join(sorted(PRODUCT_KEYS | {AUTOCORRELATION_KEY}))
        )
    for key in ("name", "raw_bytes", "age_s", "expiry_s"):
        if key not in mapping:
            raise ValueError(f"product entry is missing the required key {key!r}")
    depends = mapping.get("depends_on", ())
    if isinstance(depends, str):
        depends = (depends,)
    product = DataProduct(
        name=str(mapping["name"]),
        raw_bytes=float(mapping["raw_bytes"]),
        age_s=float(mapping["age_s"]),
        expiry_s=float(mapping["expiry_s"]),
        declared_value=(
            None if mapping.get("declared_value") is None else float(mapping["declared_value"])
        ),
        compressibility=(
            None if mapping.get("compressibility") is None else float(mapping["compressibility"])
        ),
        instrument=str(mapping.get("instrument", "unknown")),
        entropy_bits_per_sample=float(mapping.get("entropy_bits_per_sample", 8.0)),
        bit_depth=int(mapping.get("bit_depth", 16)),
        channels=int(mapping.get("channels", 1)),
        coverage_fraction=float(mapping.get("coverage_fraction", 1.0)),
        novelty=float(mapping.get("novelty", 0.5)),
        depends_on=tuple(str(name) for name in depends),
    )
    autocorrelation = mapping.get(AUTOCORRELATION_KEY)
    return product, (None if autocorrelation is None else float(autocorrelation))


def window_from_mapping(node: Any) -> ContactWindow:
    """A :class:`ContactWindow` from one mapping."""
    mapping = _require_mapping(node, "a window entry")
    unknown = sorted(set(mapping) - WINDOW_KEYS)
    if unknown:
        raise ValueError(
            "unknown window key(s): "
            + ", ".join(unknown)
            + "; expected a subset of "
            + ", ".join(sorted(WINDOW_KEYS))
        )
    for key in ("name", "start_s", "end_s", "rate_breakpoints"):
        if key not in mapping:
            raise ValueError(f"window entry is missing the required key {key!r}")
    breakpoints = mapping["rate_breakpoints"]
    if not isinstance(breakpoints, (list, tuple)):
        raise ValueError("rate_breakpoints must be a list of [time_s, rate_bps] pairs")
    points: list[tuple[float, float]] = []
    for entry in breakpoints:
        if not isinstance(entry, (list, tuple)) or len(entry) != 2:
            raise ValueError(
                f"each rate breakpoint must be a [time_s, rate_bps] pair, got {entry!r}"
            )
        points.append((float(entry[0]), float(entry[1])))
    return ContactWindow(
        name=str(mapping["name"]),
        start_s=float(mapping["start_s"]),
        end_s=float(mapping["end_s"]),
        rate_breakpoints=tuple(points),
        reliability=float(mapping.get("reliability", 1.0)),
        station=str(mapping.get("station", "unknown")),
    )


def load_case(path: str | Path) -> Case:
    """Load a case file.

    Parameters
    ----------
    path:
        Path to the YAML document, relative or absolute. It is used as given;
        this function never consults the working directory itself.

    Raises
    ------
    FileNotFoundError
        If the file does not exist. The message quotes the path exactly as the
        caller supplied it, so a relative path stays relative in the error.
    ValueError
        On a missing or unknown key, on a value the dataclasses reject, or on a
        file that is not parseable YAML. A parser error is translated rather
        than propagated: a caller handed a path has no reason to know that this
        loader is built on PyYAML, and the CLI turns a ValueError into an exit
        status 2 with a message instead of a traceback.
    """
    import yaml

    target = Path(path)
    if not target.exists():
        raise FileNotFoundError(f"no case file at {path}")
    try:
        document = yaml.safe_load(target.read_text())
    except yaml.YAMLError as error:
        raise ValueError(
            f"{path} is not parseable as YAML: {str(error).splitlines()[0]}"
        ) from error
    mapping = _require_mapping(document, f"the case document in {path}")
    for key in ("products", "windows"):
        if key not in mapping:
            raise ValueError(f"case {path} is missing the required key {key!r}")
    products: list[DataProduct] = []
    autocorrelation: dict[str, float] = {}
    for entry in mapping["products"]:
        product, rho = product_from_mapping(entry)
        products.append(product)
        if rho is not None:
            autocorrelation[product.name] = rho
    return Case(
        name=str(mapping.get("name", target.stem)),
        description=str(mapping.get("description", "")),
        inventory=Inventory(products),
        plan=ContactPlan(window_from_mapping(entry) for entry in mapping["windows"]),
        autocorrelation=autocorrelation,
        source=str(path),
    )


def case_to_mapping(case: Case) -> dict[str, Any]:
    """Round-trip a case back to a plain mapping suitable for YAML."""
    products: list[dict[str, Any]] = []
    for product in case.inventory:
        entry: dict[str, Any] = {
            "name": product.name,
            "raw_bytes": product.raw_bytes,
            "age_s": product.age_s,
            "expiry_s": product.expiry_s,
            "declared_value": product.declared_value,
            "compressibility": product.compressibility,
            "instrument": product.instrument,
            "entropy_bits_per_sample": product.entropy_bits_per_sample,
            "bit_depth": product.bit_depth,
            "channels": product.channels,
            "coverage_fraction": product.coverage_fraction,
            "novelty": product.novelty,
        }
        if product.depends_on:
            entry["depends_on"] = list(product.depends_on)
        if product.name in case.autocorrelation:
            entry[AUTOCORRELATION_KEY] = case.autocorrelation[product.name]
        products.append(entry)
    windows = [
        {
            "name": window.name,
            "station": window.station,
            "start_s": window.start_s,
            "end_s": window.end_s,
            "reliability": window.reliability,
            "rate_breakpoints": [list(point) for point in window.rate_breakpoints],
        }
        for window in case.plan
    ]
    return {
        "name": case.name,
        "description": case.description,
        "products": products,
        "windows": windows,
    }
