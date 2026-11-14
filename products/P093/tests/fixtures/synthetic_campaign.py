"""SYNTHETIC campaign fixtures, with the closed forms the tests check against.

The generator itself is ``src/powerenvelope/synthetic.py``, because the CLI's
``--simulate`` path needs it too. This module adds what only the tests need: the
closed-form value of every quantity the generator's declared model implies,
written out here from the declared constants and NOT by calling the product's
own integrator.

That is the point of writing them twice. A test that computes the expected
energy by calling :func:`powerenvelope.energy.trapezoid_energy_mj` checks
nothing. The formulas below are derived from the generator's model on paper and
are reproduced in the comments so a reader can check the algebra:

For a cell of duty ``d`` and ``c`` cycles, with period ``h = 0.05`` s, 40 slots
per 2.0 s cycle, ``m = 40 d`` active slots, ``r = min(3, m)`` ramp slots, floor
power ``F``, active excess ``D`` and ramp excess ``A``:

The generator BRACKETS the run: the sampler starts half a second before
``t0_mono`` and stops half a second after ``t1_mono``, exactly as the campaign
does, and the pipeline is off outside the run so those samples read the floor.
``E_total`` is the integral over the RUN, ``[t0_mono, t1_mono]``, which is slots
``0 .. N`` inclusive where ``N = 40 c``; the bracket samples are not in it. A
fixture whose sampler span equalled its run could not catch code that
integrated the span and called it the run, which is the defect the measured
campaign exposed.

.. code-block:: text

    N        = 40 c                              slots in the run
    sum P    = (N + 1) F + (c m) D + (c r) A     over slots 0 .. N
    P_first  = F + D + A        (slot 0, active and ramping, when m > 0)
    P_last   = F               (slot N is the first instant after the run)
    E_total  = h (sum P - (P_first + P_last) / 2)

and, for ``d < 1``, where each cycle's merged active window is
``[2k, 2k + 2d]`` and the sample at its right edge reads the floor:

.. code-block:: text

    per cycle:  v_0 = F + D + A,  v_m = F
                sum v = r (F + D + A) + (m - r) (F + D) + F
    E_active = c h (sum v - (v_0 + v_m) / 2)

For ``d = 1`` the active windows of consecutive cycles are contiguous, so they
merge into one window spanning the whole run and ``E_active == E_total``. The
helpers below refuse the ``d = 1`` case for the per-cycle formula and the tests
use the identity instead.

**Why the agreement is to 1e-9 and not exact.** The campaign writes its power
column with ``%.3f``, exactly as ``data/campaign.py`` does, so the value the
loader reads is the three-decimal rounding of ``V * I / 1000``. The closed forms
above evaluate ``V * I / 1000`` in floating point without that rounding. The two
therefore differ per sample at about the 1e-13 relative level, and the
difference accumulates across the few hundred samples of a window: measured on
the ``w96_d075`` cell, the active energy agrees to 1.6e-11 relative and not
better. :data:`CLOSED_FORM_REL` is set to 1e-9, which is two orders above the
observed disagreement and ten orders below anything a real mistake in the
integrator would produce. Nothing in this suite asserts bit-exact equality over
a floating-point pipeline.
"""

from __future__ import annotations

from pathlib import Path

from powerenvelope.provenance import Origin
from powerenvelope.schema import CURRENT_LSB_MA
from powerenvelope.synthetic import (
    SYNTHETIC_CYCLE_S,
    SYNTHETIC_I_ACTIVE_LSB,
    SYNTHETIC_I_FLOOR_LSB,
    SYNTHETIC_I_RAMP_LSB,
    SYNTHETIC_LATENCY_S,
    SYNTHETIC_RAMP_S,
    SYNTHETIC_SAMPLE_PERIOD_S,
    SYNTHETIC_TAU_PERMISSIVE,
    SYNTHETIC_TAU_STRICT,
    SYNTHETIC_V_MV,
    gate_outcome,
    write_campaign,
)
from powerenvelope.trace import Campaign, load_campaign

#: Slots per cycle on the synthetic sample grid.
SLOTS_PER_CYCLE = int(round(SYNTHETIC_CYCLE_S / SYNTHETIC_SAMPLE_PERIOD_S))

#: Ramp slots on that grid.
RAMP_SLOTS = int(round(SYNTHETIC_RAMP_S / SYNTHETIC_SAMPLE_PERIOD_S))


def _power_mw(lsb: int) -> float:
    return SYNTHETIC_V_MV * (lsb * CURRENT_LSB_MA) / 1000.0


def floor_mw() -> float:
    """Closed-form synthetic floor power, mW."""
    return _power_mw(SYNTHETIC_I_FLOOR_LSB)


def active_excess_mw() -> float:
    """Closed-form synthetic active excess above the floor, mW."""
    return _power_mw(SYNTHETIC_I_FLOOR_LSB + SYNTHETIC_I_ACTIVE_LSB) - floor_mw()


def ramp_excess_mw() -> float:
    """Closed-form synthetic ramp excess above the active level, mW."""
    return (
        _power_mw(SYNTHETIC_I_FLOOR_LSB + SYNTHETIC_I_ACTIVE_LSB + SYNTHETIC_I_RAMP_LSB)
        - _power_mw(SYNTHETIC_I_FLOOR_LSB + SYNTHETIC_I_ACTIVE_LSB)
    )


def active_slots(duty: float) -> int:
    """Active slots per cycle on the synthetic grid."""
    return int(round(duty * SLOTS_PER_CYCLE))


def total_energy_mj(duty: float, cycles: int) -> float:
    """Closed-form energy over the RUN of a synthetic cell, mJ.

    Over ``[t0_mono, t1_mono]``, which is slots 0 to N inclusive. The sampler's
    bracketing samples on either side are deliberately NOT in it.
    """
    h = SYNTHETIC_SAMPLE_PERIOD_S
    f, d, a = floor_mw(), active_excess_mw(), ramp_excess_mw()
    m = active_slots(duty)
    r = min(RAMP_SLOTS, m)
    n = cycles * SLOTS_PER_CYCLE
    total = (n + 1) * f + cycles * m * d + cycles * r * a
    first = f + (d if m > 0 else 0.0) + (a if r > 0 else 0.0)
    last = f  # slot N is the first instant after the run; the pipeline is off
    return h * (total - (first + last) / 2.0)


def sleep_energy_mj(duty: float, cycles: int) -> float:
    """Closed-form energy over the declared SLEEP phases, mJ, for ``duty < 1``.

    The declared phases tile the run exactly, so this is the run's energy less
    the active phases'. It is written as the difference on purpose: the two
    closed forms above are derived independently, and their difference being
    what the sleep integral returns is the partition identity.
    """
    if duty >= 1.0:
        raise ValueError("a duty-1.00 cell has no sleep phase")
    return total_energy_mj(duty, cycles) - active_energy_mj(duty, cycles)


def active_energy_mj(duty: float, cycles: int) -> float:
    """Closed-form active-window energy of a synthetic cell, mJ, for ``duty < 1``.

    Raises
    ------
    ValueError
        For ``duty >= 1``, where consecutive cycles' active windows merge into
        one and the per-cycle formula does not apply. Use the identity
        ``E_active == E_total`` instead, which the tests assert directly.
    """
    if duty >= 1.0:
        raise ValueError(
            "at duty 1.0 the active windows of consecutive cycles are contiguous and "
            "merge into one window spanning the whole run, so the per-cycle closed "
            "form does not apply; E_active equals E_total there"
        )
    if duty <= 0.0:
        raise ValueError("a cell with duty 0 has no active window")
    h = SYNTHETIC_SAMPLE_PERIOD_S
    f, d, a = floor_mw(), active_excess_mw(), ramp_excess_mw()
    m = active_slots(duty)
    r = min(RAMP_SLOTS, m)
    total = r * (f + d + a) + (m - r) * (f + d) + f
    v_first = f + d + a
    v_last = f
    return cycles * h * (total - (v_first + v_last) / 2.0)


def inferences(duty: float, dim: int, cycles: int) -> int:
    """Closed-form inference count of a synthetic cell."""
    per_cycle = int(round(duty * SYNTHETIC_CYCLE_S / SYNTHETIC_LATENCY_S[dim]))
    return per_cycle * cycles


def tau_for(dim: int, permissive: bool = False) -> float:
    """The declared synthetic gate threshold for a frame dimension."""
    return SYNTHETIC_TAU_PERMISSIVE if permissive else SYNTHETIC_TAU_STRICT[dim]


def accepted(duty: float, dim: int, cycles: int, tau: float | None = None) -> int:
    """Accepted count, by applying the declared gate rule to the frame indices.

    This re-applies :func:`powerenvelope.synthetic.gate_outcome` rather than
    reading the written file, so a test comparing it with what the loader
    counted checks the whole write-then-read path.
    """
    threshold = tau_for(dim) if tau is None else tau
    return sum(
        gate_outcome(index, threshold)[0]
        for index in range(inferences(duty, dim, cycles))
    )


def acceptance_rate(duty: float, dim: int, cycles: int, tau: float | None = None) -> float:
    """Acceptance rate of a synthetic cell, from the declared gate rule."""
    n = inferences(duty, dim, cycles)
    if n == 0:
        raise ValueError("a cell with no inference has no acceptance rate")
    return accepted(duty, dim, cycles, tau) / n


def nominal_factor() -> float:
    """``V_measured / V_nominal`` of the synthetic model, dimensionless.

    The nominal-rail variant substitutes 5000 mV for the measured bus voltage,
    so the rail-voltage factor of the decomposition must come out as exactly
    this ratio in exact arithmetic.
    """
    return SYNTHETIC_V_MV / 5000.0


#: Cycles per run for every fixture here.
CYCLES = 10

#: Relative tolerance for a closed-form comparison. See this module's docstring:
#: the CSV's ``%.3f`` power column and the exact closed form cannot agree better
#: than about 1e-11 relative, and this is set two orders above that.
CLOSED_FORM_REL = 1e-9


def write_campaign_dir(directory: Path | str, **kwargs) -> Path:
    """Write a synthetic campaign with the fixture defaults, return its path."""
    kwargs.setdefault("cycles", CYCLES)
    return write_campaign(directory, **kwargs)


def load_dir(directory: Path | str) -> Campaign:
    """Load an already-written synthetic campaign directory as SYNTHETIC."""
    root = Path(directory)
    return load_campaign(root, origin=Origin.SYNTHETIC, directory_label=root.name)


def campaign(directory: Path | str, **kwargs) -> Campaign:
    """Write a synthetic campaign into ``directory`` and load it as SYNTHETIC."""
    return load_dir(write_campaign_dir(directory, **kwargs))
