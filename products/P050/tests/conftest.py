"""Shared fixtures. The two small tables here are the ones hand-solved in
``tests/test_known_answers.py``; nothing else depends on their exact numbers.
"""

from __future__ import annotations

import numpy as np
import pytest

from coderateopt import EmpiricalFade, ModcodSet, RateProblem, illustrative_modcod_table

#: Four dB fades, so the empirical survival function takes the exact values
#: 1, 3/4, 1/2, 1/4, 0 and every availability below is a short fraction.
QUARTER_SAMPLES_DB = np.array([-6.0, -4.0, -2.0, 0.0])


@pytest.fixture
def quarter_fade() -> EmpiricalFade:
    return EmpiricalFade(QUARTER_SAMPLES_DB)


@pytest.fixture
def monotone_table() -> ModcodSet:
    """Availabilities 1, 3/4, 1/2, 1/4 and goodputs 1.0, 1.5, 2.0, 2.0 at 10 dB."""
    return ModcodSet.from_rows(
        [
            ("m1", 1.0, 4.0),
            ("m2", 2.0, 6.0),
            ("m3", 4.0, 8.0),
            ("m4", 8.0, 10.0),
        ]
    )


@pytest.fixture
def non_monotone_table() -> ModcodSet:
    """Goodputs 1.0, 1.5, 2.0, 1.5 at 10 dB -- the peak is not the top rate."""
    return ModcodSet.from_rows(
        [
            ("m1", 1.0, 4.0),
            ("m2", 2.0, 6.0),
            ("m3", 4.0, 8.0),
            ("m4", 6.0, 10.0),
        ]
    )


@pytest.fixture
def monotone_problem(monotone_table, quarter_fade) -> RateProblem:
    return RateProblem(
        modcods=monotone_table,
        fade=quarter_fade,
        margin_db=10.0,
        availability_target=0.7,
        max_entries=1,
    )


@pytest.fixture
def reference_table() -> ModcodSet:
    return illustrative_modcod_table()
