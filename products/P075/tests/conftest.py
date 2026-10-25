"""Shared fixtures, and the tolerance convention this suite uses.

The ``pytest.approx`` default absolute tolerance
------------------------------------------------

``pytest.approx(expected, rel=r)`` carries a **default absolute tolerance of
1e-12** and passes if *either* bound is met. For this package that default is
acute rather than theoretical: reliability complements and small failure
probabilities here are routinely of order 1e-5 and go down to 1e-18, so any
relative tolerance tighter than about 2e-8 on an unreliability figure of order
1e-5 is silently replaced by the absolute default, and the assertion no longer
checks what it says it checks.

Every ``approx`` call in this suite therefore passes ``abs=0.0`` explicitly
when it means a relative tolerance, through :func:`rel`. Where an absolute
tolerance is genuinely what is wanted, it is passed as ``abs=`` with ``rel=0``
through :func:`abs_`. ``validation/audit_approx_tolerances.py`` re-runs the
whole suite with every ``rel=`` call forced to ``abs=0.0`` and reports how many
assertions depended on the default; the committed answer is in its output file.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from reliablock.diagram import (  # noqa: E402
    Block,
    Diagram,
    KOfN,
    Network,
    Parallel,
    Series,
    SharedLoad,
    Standby,
)
from reliablock.distributions import (  # noqa: E402
    Exponential,
    FixedProbability,
    Lognormal,
    Weibull,
)


def rel(expected: float, tolerance: float) -> object:
    """``pytest.approx`` with a relative tolerance and **no** absolute fallback.

    This is the only form this suite uses for a relative comparison. Writing
    ``pytest.approx(x, rel=1e-12)`` would also pass on an absolute difference
    of 1e-12, which for a quantity of order 1e-18 is every value in the
    neighbourhood.
    """
    return pytest.approx(expected, rel=tolerance, abs=0.0)


def abs_(expected: float, tolerance: float) -> object:
    """``pytest.approx`` with an absolute tolerance and no relative fallback."""
    return pytest.approx(expected, rel=0.0, abs=tolerance)


def exponential_block(name: str, rate_per_hour: float) -> Block:
    """A block with a constant hazard rate in 1/hour."""
    return Block(name=name, distribution=Exponential(rate_per_hour))


def fixed_block(name: str, q: float) -> Block:
    """A one-shot block with a declared per-actuation failure probability."""
    return Block(name=name, distribution=FixedProbability(q))


def blocks_at_unreliability(names: list[str], q: float) -> list[Block]:
    """Blocks whose unreliability at ``t = 1`` hour is exactly ``q``."""
    rate = -math.log1p(-q)
    return [exponential_block(name, rate) for name in names]


@pytest.fixture
def series_pair() -> Diagram:
    """Two blocks in series."""
    return Diagram(
        "series pair",
        Series(items=(exponential_block("a", 2.0e-4), exponential_block("b", 3.0e-4))),
    )


@pytest.fixture
def parallel_pair() -> Diagram:
    """Two identical blocks in parallel, no declared beta."""
    return Diagram(
        "parallel pair",
        Parallel(
            items=(exponential_block("a", 3.0e-4), exponential_block("b", 3.0e-4)),
            label="pair",
        ),
    )


@pytest.fixture
def beta_pair() -> Diagram:
    """Two identical blocks in parallel with a declared beta of 0.1."""
    return Diagram(
        "beta pair",
        Parallel(
            items=(exponential_block("a", 3.0e-4), exponential_block("b", 3.0e-4)),
            beta_factor=0.1,
            label="pair",
        ),
    )


@pytest.fixture
def two_of_three() -> Diagram:
    """A 2-of-3 group of identical blocks."""
    return Diagram(
        "two of three",
        KOfN(
            k=2,
            items=(
                exponential_block("a", 3.0e-4),
                exponential_block("b", 3.0e-4),
                exponential_block("c", 3.0e-4),
            ),
            label="triple",
        ),
    )


@pytest.fixture
def cold_standby() -> Diagram:
    """A cold standby pair with a declared switch reliability of 0.98."""
    return Diagram(
        "cold standby",
        Standby(
            primary=exponential_block("prime", 3.0e-4),
            standbys=(exponential_block("spare", 3.0e-4),),
            switch_reliability=0.98,
            label="pair",
        ),
    )


@pytest.fixture
def warm_standby() -> Diagram:
    """A warm standby pair at hazard fraction 0.4 and switch reliability 0.97."""
    return Diagram(
        "warm standby",
        Standby(
            primary=exponential_block("prime", 3.0e-4),
            standbys=(exponential_block("spare", 3.0e-4),),
            switch_reliability=0.97,
            standby_hazard_fraction=0.4,
            label="pair",
        ),
    )


@pytest.fixture
def shared_load() -> Diagram:
    """Two load-sharing units at a sharing factor of 0.5."""
    return Diagram(
        "shared load",
        SharedLoad(
            units=(exponential_block("u1", 4.0e-4), exponential_block("u2", 5.0e-4)),
            shared_rate_factor=0.5,
            label="pumps",
        ),
    )


@pytest.fixture
def bridge_network() -> Diagram:
    """The five-edge bridge at equal edge reliability 0.9, as one-shot blocks."""
    layout = (
        ("s", "a", "e1"),
        ("s", "b", "e2"),
        ("a", "b", "e3"),
        ("a", "t", "e4"),
        ("b", "t", "e5"),
    )
    return Diagram(
        "bridge",
        Network(
            edges=tuple((x, y, fixed_block(name, 0.1)) for x, y, name in layout),
            source="s",
            sink="t",
            label="bridge",
        ),
    )


@pytest.fixture
def weibull_block() -> Block:
    """A Weibull block with a wear-out shape."""
    return Block(name="w", distribution=Weibull(shape=2.3, scale_hours=5.0e4))


@pytest.fixture
def lognormal_block() -> Block:
    """A lognormal block with a median life of exp(10) hours."""
    return Block(name="ln", distribution=Lognormal(log_median=10.0, log_sigma=0.6))
