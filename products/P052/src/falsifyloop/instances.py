r"""The seeded benchmark suite: eight falsification instances of stated difficulty.

An *instance* is a simulator, a declared search box, and one requirement. Its
**difficulty** is the probability that a single uniform draw from the box
violates the requirement -- the quantity that decides how many simulations a
uniform-random search needs, since the number of draws to the first violation is
geometric with that parameter and has mean ``1/p``.

How the bounds were chosen, and why that is not circular
--------------------------------------------------------
Each requirement bound was set from a **pilot** run of 40000 uniform draws
(seed 13, 2026-10-08) so that the suite spans four orders of difficulty rather
than clustering. The number recorded here is therefore the *design target*, and
it is not evidence of anything: the figure the README and VALIDATION.md quote is
the independently re-measured probability with a Clopper-Pearson interval from
``validation/validate_difficulty.py``, which uses a different seed and commits
its raw output. If the two disagree beyond the interval, the measured one is the
one that counts.

Choosing a bound to hit a target difficulty is legitimate *because the search
strategies are never told the bound's provenance*. It would be illegitimate to
choose the bound after seeing which strategy won, and no bound here was.

Difficulty tiers
----------------
========== ================== ==========================================
tier       target ``p``       uniform-random expectation at 100 draws
========== ================== ==========================================
easy       ``>= 0.1``         finds a violation essentially always
moderate   ``0.01 - 0.1``     usually finds one
hard       ``0.002 - 0.01``   finds one in roughly a fifth to two thirds
very hard  ``< 0.002``        usually does not
========== ================== ==========================================

The search box
--------------
Six decision variables, in :data:`falsifyloop.systems.LoopInput.FIELDS` order:

===================== ============== ==========================================
variable              range          unit / meaning
===================== ============== ==========================================
``step_amplitude``    1.0 - 6.0      deg, commanded attitude step
``kp_factor``         0.6 - 2.0      dimensionless, multiplies nominal ``Kp``
``kd_factor``         0.25 - 1.2     dimensionless, multiplies nominal ``Kd``
``tau_factor``        0.6 - 3.0      dimensionless, multiplies actuator ``tau``
``gust_amplitude``    0.0 - 15.0     deg/s^2, sinusoidal gust amplitude
``gust_frequency``    0.1 - 3.0      Hz, gust frequency
===================== ============== ==========================================

The box is the same for every instance so that a strategy comparison is not
confounded by the dimension or the scaling of the space.

An honesty note about the physics at the violating settings
-----------------------------------------------------------
The violating corners of this box drive the attitude well past the small-angle
regime in which the declared second-order model is even nominally valid:
``validation/validate_simulator.py`` check 3 observed a worst ``|theta|`` of
24.1727 deg over 4000 uniform draws. **That is a property of the benchmark, not
a prediction about any vehicle.** The suite exists to measure
how many simulations a search needs to find a corner of a declared box, and the
simulator's job is to be a cheap, deterministic, non-smooth, non-convex function
of six variables, which it is. Nothing here licenses a statement about an
aircraft.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .requirements import (
    Abs,
    Always,
    And,
    Difference,
    Eventually,
    Formula,
    Predicate,
    Signal,
    check_horizon,
    robustness,
)
from .systems import (
    DEFAULT_DT,
    DEFAULT_HORIZON,
    LoopInput,
    LoopParameters,
    simulate,
)
from .traces import Trace

#: Declared search box, shape ``(6, 2)``, rows in ``LoopInput.FIELDS`` order.
SEARCH_BOX = np.array(
    [
        [1.0, 6.0],
        [0.6, 2.0],
        [0.25, 1.2],
        [0.6, 3.0],
        [0.0, 15.0],
        [0.1, 3.0],
    ]
)

#: Difficulty tier labels, easiest first.
TIERS = ("easy", "moderate", "hard", "very hard")


@dataclass(frozen=True)
class Instance:
    """One falsification problem: simulator, box, requirement.

    Attributes
    ----------
    identifier:
        Short stable name used by the CLI and every report.
    requirement:
        The :class:`~falsifyloop.requirements.Formula` to falsify.
    rationale:
        What engineering question the requirement stands in for, in one sentence.
    tier:
        One of :data:`TIERS`.
    design_target_probability:
        The uniform-violation probability the bound was chosen to hit, from the
        pilot run described in this module's docstring. **Not** evidence; see
        ``validation/validate_difficulty.py`` for the measured figure.
    box:
        Search box, shape ``(6, 2)``.
    dt, horizon:
        Simulation step and duration in seconds.
    parameters:
        Declared loop constants.
    """

    identifier: str
    requirement: Formula
    rationale: str
    tier: str
    design_target_probability: float
    box: np.ndarray = field(default_factory=lambda: SEARCH_BOX.copy())
    dt: float = DEFAULT_DT
    horizon: float = DEFAULT_HORIZON
    parameters: LoopParameters = field(default_factory=LoopParameters)

    def __post_init__(self) -> None:
        if not isinstance(self.identifier, str) or not self.identifier:
            raise ValueError(f"identifier must be a non-empty string, got {self.identifier!r}")
        if self.tier not in TIERS:
            raise ValueError(f"tier must be one of {TIERS}, got {self.tier!r}")
        if not 0.0 < self.design_target_probability < 1.0:
            raise ValueError(
                "design_target_probability must lie in (0, 1), got "
                f"{self.design_target_probability}"
            )
        box = np.asarray(self.box, dtype=float)
        if box.shape != (len(LoopInput.FIELDS), 2):
            raise ValueError(f"box must have shape ({len(LoopInput.FIELDS)}, 2), got {box.shape}")
        if not np.all(np.isfinite(box)):
            raise ValueError("box contains a non-finite bound")
        if not np.all(box[:, 1] > box[:, 0]):
            bad = np.nonzero(box[:, 1] <= box[:, 0])[0].tolist()
            raise ValueError(f"box rows {bad} have an upper bound at or below the lower bound")
        # The horizon must actually cover the requirement's time windows, or the
        # verdict at t = 0 is taken over a clipped window and means less than it
        # looks like it means. Checked once, here, on a representative trace.
        probe = self.simulate(self.centre())
        check_horizon(self.requirement, probe)

    @property
    def dimension(self) -> int:
        """Number of decision variables."""
        return int(self.box.shape[0])

    def centre(self) -> np.ndarray:
        """Box centre, shape ``(6,)``."""
        return 0.5 * (self.box[:, 0] + self.box[:, 1])

    def widths(self) -> np.ndarray:
        """Box edge lengths, shape ``(6,)``."""
        return self.box[:, 1] - self.box[:, 0]

    def clip(self, vector: np.ndarray) -> np.ndarray:
        """Clip a decision vector into the box."""
        return np.clip(np.asarray(vector, dtype=float), self.box[:, 0], self.box[:, 1])

    def sample(self, rng: np.random.Generator, size: int = 1) -> np.ndarray:
        """Draw ``size`` uniform points from the box, shape ``(size, 6)``."""
        if size < 1:
            raise ValueError(f"size must be at least 1, got {size}")
        return rng.uniform(self.box[:, 0], self.box[:, 1], size=(size, self.dimension))

    def simulate(self, vector: np.ndarray) -> Trace:
        """Simulate the loop at decision vector ``vector``."""
        return simulate(
            LoopInput.from_array(vector),
            parameters=self.parameters,
            dt=self.dt,
            horizon=self.horizon,
        )

    def evaluate(self, vector: np.ndarray) -> float:
        """Requirement robustness at ``vector``. Negative exactly when violated.

        One call is one simulation. This is the unit the sample-efficiency curve
        counts, so nothing in this package evaluates the requirement without
        going through here or through :meth:`simulate` plus
        :func:`~falsifyloop.requirements.robustness`.
        """
        return robustness(self.requirement, self.simulate(vector))

    def describe(self) -> str:
        """Human-readable one-block description. Returns a string; prints nothing."""
        lines = [
            f"{self.identifier}  [{self.tier}]",
            f"  requirement : {self.requirement}",
            f"  rationale   : {self.rationale}",
            f"  design target p(violation | uniform draw) = {self.design_target_probability:.4f}",
            f"  horizon {self.horizon:g} s at dt {self.dt:g} s"
            f"  ({int(self.horizon / self.dt) + 1} samples)",
        ]
        for name, (lo, hi) in zip(LoopInput.FIELDS, self.box, strict=True):
            lines.append(f"  box {name:<16s} [{lo:g}, {hi:g}]")
        return "\n".join(lines)


def _build_suite() -> dict[str, Instance]:
    """Construct the eight shipped instances. Called once at import."""
    over = Signal("over")
    err = Abs(Signal("error"))
    theta = Abs(Signal("theta"))
    rate = Abs(Signal("q"))
    cmd_rate = Abs(Difference("cmd"))

    suite = [
        Instance(
            identifier="overshoot-loose",
            requirement=Always(Predicate(over, "<=", 2.0, scale=2.0), 0.0, 2.0),
            rationale=(
                "attitude must not exceed the commanded step by more than 2 deg at any "
                "time in the first 2 s: a loose overshoot allowance"
            ),
            tier="easy",
            design_target_probability=0.31,
        ),
        Instance(
            identifier="settling-band",
            requirement=Always(Predicate(err, "<=", 3.0, scale=3.0), 1.2, 2.0),
            rationale=(
                "tracking error must stay inside a 3 deg band from 1.2 s to 2 s: a "
                "settling requirement"
            ),
            tier="easy",
            design_target_probability=0.10,
        ),
        Instance(
            identifier="command-rate",
            requirement=Always(Predicate(cmd_rate, "<=", 100.0, scale=100.0), 0.0, 2.0),
            rationale=(
                "the commanded deflection rate must stay under 100 deg/s, so that the "
                "actuator's own slew limit is not the only thing holding the loop together"
            ),
            tier="moderate",
            design_target_probability=0.030,
        ),
        Instance(
            identifier="multi-requirement",
            requirement=And(
                Always(Predicate(over, "<=", 6.0, scale=6.0), 0.0, 2.0),
                Always(Predicate(rate, "<=", 40.0, scale=40.0), 0.0, 2.0),
                Always(Predicate(err, "<=", 4.0, scale=4.0), 1.2, 2.0),
            ),
            rationale=(
                "three requirements at once -- overshoot, attitude rate and settling -- "
                "each normalised by its own tolerance so the conjunction's min is taken "
                "over dimensionless margins"
            ),
            tier="moderate",
            design_target_probability=0.063,
        ),
        Instance(
            identifier="overshoot-tight",
            requirement=Always(Predicate(over, "<=", 9.0, scale=9.0), 0.0, 2.0),
            rationale=(
                "the same overshoot requirement at a bound the loop meets almost "
                "everywhere in the box: the violating set is a corner"
            ),
            tier="hard",
            design_target_probability=0.010,
        ),
        Instance(
            identifier="nested-capture",
            requirement=Eventually(
                Always(Predicate(err, "<=", 5.5, scale=5.5), 0.0, 0.5), 0.0, 1.0
            ),
            rationale=(
                "the error must stay inside a 5.5 deg band for some half-second window "
                "beginning in the first second: a capture requirement, and the suite's "
                "only nested temporal operator"
            ),
            tier="hard",
            design_target_probability=0.0035,
        ),
        Instance(
            identifier="attitude-envelope",
            requirement=Always(Predicate(theta, "<=", 17.0, scale=17.0), 0.0, 2.0),
            rationale=(
                "attitude must stay inside a 17 deg envelope: an envelope-protection "
                "requirement whose violating set is a thin corner of the box"
            ),
            tier="hard",
            design_target_probability=0.0040,
        ),
        Instance(
            identifier="rate-envelope",
            requirement=Always(Predicate(rate, "<=", 75.0, scale=75.0), 0.0, 2.0),
            rationale=(
                "attitude rate must stay under 75 deg/s: the hardest instance in the "
                "suite, violated by roughly one uniform draw in seven hundred"
            ),
            tier="very hard",
            design_target_probability=0.0014,
        ),
    ]
    return {inst.identifier: inst for inst in suite}


_SUITE = _build_suite()

#: Instance identifiers in increasing order of design difficulty.
SUITE_ORDER: tuple[str, ...] = tuple(
    sorted(_SUITE, key=lambda k: -_SUITE[k].design_target_probability)
)


def instance(identifier: str) -> Instance:
    """Return the shipped instance named ``identifier``.

    Raises
    ------
    KeyError
        If no such instance exists; the message lists the ones that do.
    """
    try:
        return _SUITE[identifier]
    except KeyError:
        raise KeyError(
            f"unknown instance {identifier!r}; the suite is {list(SUITE_ORDER)}"
        ) from None


def suite() -> tuple[Instance, ...]:
    """All shipped instances, easiest design target first."""
    return tuple(_SUITE[k] for k in SUITE_ORDER)
