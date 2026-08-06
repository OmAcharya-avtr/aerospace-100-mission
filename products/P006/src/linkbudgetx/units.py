"""Unit conversions for optical link-budget calculations.

All conversions are exact algebraic identities (no physics assumptions).

Conventions
-----------
- dBm: decibels referenced to 1 mW, ``P_dBm = 10 log10(P_W / 1e-3)``
  (IEEE Std 100, standard decibel definition).
- dB: ``x_dB = 10 log10(x_linear)`` for power ratios.
- Angles are radians unless a suffix says otherwise.
"""

from __future__ import annotations

import math

__all__ = [
    "dbm_to_watts",
    "watts_to_dbm",
    "db_to_linear",
    "linear_to_db",
    "nm_to_m",
    "m_to_nm",
    "km_to_m",
    "m_to_km",
]


def dbm_to_watts(p_dbm: float) -> float:
    """Convert power in dBm to watts.

    Args:
        p_dbm: Power in dBm (dB referenced to 1 mW).

    Returns:
        Power in W. Always positive.
    """
    return 1e-3 * 10.0 ** (p_dbm / 10.0)


def watts_to_dbm(p_w: float) -> float:
    """Convert power in watts to dBm.

    Args:
        p_w: Power in W. Must be > 0 (dB of a non-positive power is undefined).

    Returns:
        Power in dBm.

    Raises:
        ValueError: If ``p_w <= 0``.
    """
    if p_w <= 0.0:
        raise ValueError(f"Power must be > 0 W to express in dBm, got {p_w} W.")
    return 10.0 * math.log10(p_w / 1e-3)


def db_to_linear(x_db: float) -> float:
    """Convert a power ratio in dB to a linear ratio: ``10**(x_db/10)``."""
    return 10.0 ** (x_db / 10.0)


def linear_to_db(x: float) -> float:
    """Convert a linear power ratio to dB: ``10 log10(x)``.

    Args:
        x: Linear power ratio. Must be > 0.

    Raises:
        ValueError: If ``x <= 0``.
    """
    if x <= 0.0:
        raise ValueError(f"Ratio must be > 0 to express in dB, got {x}.")
    return 10.0 * math.log10(x)


def nm_to_m(x_nm: float) -> float:
    """Nanometres to metres."""
    return x_nm * 1e-9


def m_to_nm(x_m: float) -> float:
    """Metres to nanometres."""
    return x_m * 1e9


def km_to_m(x_km: float) -> float:
    """Kilometres to metres."""
    return x_km * 1e3


def m_to_km(x_m: float) -> float:
    """Metres to kilometres."""
    return x_m * 1e-3
