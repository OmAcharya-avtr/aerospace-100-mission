"""linkbudgetx — deterministic, unit-aware free-space optical link budgets.

Educational (validation Level 1). Not flight-qualified, not certified,
not approved for operational aerospace use.
"""

from .core import BEAM_PROFILES, LinkBudget, LinkBudgetResult
from .uncertainty import MarginUncertainty, monte_carlo_margin, propagate_margin_sigma
from .units import (
    db_to_linear,
    dbm_to_watts,
    km_to_m,
    linear_to_db,
    m_to_km,
    m_to_nm,
    nm_to_m,
    watts_to_dbm,
)

__version__ = "0.1.0"

__all__ = [
    "LinkBudget",
    "LinkBudgetResult",
    "BEAM_PROFILES",
    "MarginUncertainty",
    "propagate_margin_sigma",
    "monte_carlo_margin",
    "dbm_to_watts",
    "watts_to_dbm",
    "db_to_linear",
    "linear_to_db",
    "nm_to_m",
    "m_to_nm",
    "km_to_m",
    "m_to_km",
    "__version__",
]
