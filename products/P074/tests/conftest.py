"""Shared fixtures.

``pytest.approx`` and the default absolute tolerance
----------------------------------------------------
``pytest.approx(expected, rel=r)`` carries a **default absolute tolerance of
1e-12** and passes when *either* bound is satisfied. For any quantity of order
1e-5, every relative tolerance tighter than 1e-7 is therefore silently
replaced by the absolute one, and the assertion stops testing what it says it
tests. Conservation residuals, which are small by construction, are exactly
where this bites.

Every ``pytest.approx`` call in this suite that passes ``rel=`` also passes an
explicit ``abs=``, so no assertion in this repository depends on the default.
``validation/validate_approx_audit.py`` enforces that by static scan and then
re-runs the whole suite with ``abs=0.0`` substituted into every such call, and
reports how many assertions changed verdict. The answer it reports is zero,
and that is a measurement rather than an intention.
"""

from __future__ import annotations

import pkgutil

import numpy as np
import pytest
from sgp4.api import Satrec

from orbitdrift.kepler import ClassicalElements, elements_to_rv


@pytest.fixture(scope="session")
def verification_tle_lines() -> list[str]:
    """The element lines of ``SGP4-VER.TLE`` from the installed sgp4 distribution."""
    raw = pkgutil.get_data("sgp4", "SGP4-VER.TLE")
    assert raw is not None
    return [
        line
        for line in raw.decode("ascii").replace("\r", "").splitlines()
        if line[:2] in ("1 ", "2 ")
    ]


@pytest.fixture(scope="session")
def moderate_drag_satrec(verification_tle_lines: list[str]) -> Satrec:
    """Catalogue 6251, the DELTA 1 DEB case of the published verification set.

    Perigee about 377 km, so drag is active but the object does not decay
    inside a week. The published set labels it "near earth normal drag
    equation", which is the regime this package is mostly about.
    """
    for index in range(0, len(verification_tle_lines), 2):
        if int(verification_tle_lines[index][2:7]) == 6251:
            return Satrec.twoline2rv(
                verification_tle_lines[index], verification_tle_lines[index + 1]
            )
    raise AssertionError("catalogue 6251 is not in SGP4-VER.TLE")


@pytest.fixture(scope="session")
def circular_leo_state() -> tuple[np.ndarray, np.ndarray]:
    """A 500 km, 51.6 degree, near-circular state. Eccentricity 1e-3, not zero.

    Not exactly circular on purpose: an exactly circular orbit makes the
    argument of perigee undefined and would exercise a branch of
    ``rv_to_elements`` that the eccentric cases do not, which is covered
    separately in ``test_kepler.py``.
    """
    elements = ClassicalElements(
        semi_major_axis_km=6878.135,
        eccentricity=1e-3,
        inclination_rad=float(np.deg2rad(51.6)),
        raan_rad=0.3,
        arg_perigee_rad=0.7,
        true_anomaly_rad=0.2,
    )
    return elements_to_rv(elements)


@pytest.fixture(scope="session")
def committed_table():
    """The committed divergence table.

    Loaded, never regenerated: ``validation/regenerate_dataset.py`` takes
    about three minutes and the suite must stay well inside its budget. A test
    that regenerated it would also be testing a different population from the
    one in the repository.
    """
    from pathlib import Path

    from orbitdrift import DivergenceTable

    path = Path(__file__).resolve().parent.parent / "fixtures" / "divergence_dataset.npz"
    if not path.exists():
        pytest.skip(
            "fixtures/divergence_dataset.npz is absent; run "
            "python validation/regenerate_dataset.py"
        )
    return DivergenceTable.load(path)


@pytest.fixture(scope="session")
def small_table(committed_table):
    """The first 48 objects of the committed table, for the benchmark tests.

    The benchmark fits thirteen predictors including four gradient-boosted
    ensembles, and on the full table that is about twenty-five seconds. The
    subset keeps the suite fast; every published benchmark number comes from
    the full table via ``validation/validate_predictors.py``, and no number
    measured on this subset appears in any document.
    """
    import numpy as np

    from orbitdrift import DivergenceTable

    objects = committed_table.column("object_index")
    keep = np.flatnonzero(objects < 48)
    return DivergenceTable(
        values=committed_table.values[keep],
        columns=committed_table.columns,
        tle_text=committed_table.tle_text,
        detail=committed_table.detail + " [SUBSET: first 48 objects, tests only]",
    )
