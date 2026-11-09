"""Deterministic synthetic cases: what reality did, and which telemetry saw it.

Every case here is produced from a declared integer seed by a generator
constructed inside the function that uses it, so the whole dataset is a
function of the seed list and nothing else. No data file is committed: the
generator is, and it is cheaper to re-run than to store (one case is one
integration of equation (1), measured at about 2 ms for the ``full``
campaign).

The four declared case kinds, and why each exists
-------------------------------------------------
``single``
    Exactly one parameter moved, by a relative amount drawn from
    ``[MIN_MAGNITUDE, MAX_MAGNITUDE]`` with a random sign. The ordinary
    attribution problem, and the only kind the learned classifier is fitted
    on. Label: the parameter index.

``scale``
    All five parameters moved by the SAME relative amount. By equation (6) the
    noiseless rate telemetry is then **identical** to the twin's, so the
    residual is pure sensor noise, and no attribution to any single parameter
    is correct. Label: :data:`UNIDENTIFIABLE`. This is the inside of the
    unidentifiable subspace in its purest form, and the analytic path's correct
    answer is its chi-square detection test failing to reject.

``equivalent``
    The partner of a ``single`` case under equation (6): for a single case
    ``u * e_j``, the perturbation ``u * (e_j - 1)``, which leaves parameter
    ``j`` at nominal and moves the other four by ``-u``. The two differ by
    exactly ``u`` times the scale direction, so they produce **identical
    noiseless telemetry** -- verified numerically in
    ``validation/validate_attribution.py`` -- while their true causes are
    different and only one of them is a single-parameter fault. Generated with
    the SAME noise seed as its partner, so the two observations are identical
    to the last bit and any classifier necessarily returns the same answer for
    both. Label: :data:`MULTI`. This is the sharp version of the
    unidentifiable-subspace test: the residual is large and easily detected,
    and it is the *attribution* that is impossible.

``weak``
    Exactly one parameter moved, by a relative amount drawn from
    ``[WEAK_MIN_MAGNITUDE, WEAK_MAX_MAGNITUDE]`` -- a tenth of the ordinary
    range, and entirely BELOW it. The learned classifier is never fitted on
    this range, so this population is an out-of-distribution test and is
    reported as one: it asks whether the learned component degrades faster
    than the analytic attributor as the fault shrinks towards the noise floor.
    Label: the parameter index.

``none``
    Nothing moved. The control that says what the detection test does on a
    residual that really is only noise. Label: :data:`UNIDENTIFIABLE`, because
    "no attributable change" is the same answer.

Magnitude range
---------------
``MIN_MAGNITUDE`` is 0.02 and ``MAX_MAGNITUDE`` 0.15 in logarithmic parameters,
so between 2.02 and 16.18 per cent. The floor is above the level at which the
rate channel could not detect the change at all for the weakest parameter and
the ceiling is inside the plant's declared validity window for every one of the
four case kinds; both are verified by
``validation/validate_learned_model.py``, which reports the measured detection
rate at the floor.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import numpy as np

from .plant import NOMINAL, PARAMETER_NAMES, WheelParameters, perturb
from .telemetry import TelemetrySet, measure

__all__ = [
    "KINDS",
    "LABEL_NAMES",
    "MAX_MAGNITUDE",
    "MIN_MAGNITUDE",
    "MULTI",
    "UNIDENTIFIABLE",
    "WEAK_MAX_MAGNITUDE",
    "WEAK_MIN_MAGNITUDE",
    "Case",
    "generate_cases",
    "observe",
]

#: Label for a case whose cause lies in the unidentifiable subspace, or which
#: has no cause at all. Negative so it can never be confused with a parameter
#: index.
UNIDENTIFIABLE: Final[int] = -1

#: Label for a case in which several parameters moved together, so that no
#: single-parameter answer is correct.
MULTI: Final[int] = -2

#: Declared minimum and maximum relative perturbation magnitude, in logarithmic
#: parameter units.
MIN_MAGNITUDE: Final[float] = 0.02
MAX_MAGNITUDE: Final[float] = 0.15

#: Declared magnitude range of the ``weak`` kind, a tenth of the ordinary one
#: and wholly below it, so that it is out of the learned component's training
#: distribution by construction.
WEAK_MIN_MAGNITUDE: Final[float] = 0.002
WEAK_MAX_MAGNITUDE: Final[float] = 0.02

#: The declared case kinds.
KINDS: Final[tuple[str, ...]] = ("single", "weak", "scale", "equivalent", "none")

#: Human-readable label names, for report tables.
LABEL_NAMES: Final[dict[int, str]] = {
    UNIDENTIFIABLE: "UNIDENTIFIABLE",
    MULTI: "MULTI",
    **{index: name for index, name in enumerate(PARAMETER_NAMES)},
}


@dataclass(frozen=True)
class Case:
    """One synthetic reality: the true perturbation, its label and its noise seed."""

    seed: int
    kind: str
    label: int
    delta_log: np.ndarray
    partner_seed: int | None = None

    @property
    def parameters(self) -> WheelParameters:
        """The true parameters of this case."""
        return perturb(NOMINAL, self.delta_log)

    @property
    def magnitude(self) -> float:
        """Euclidean norm of the perturbation, in logarithmic parameter units."""
        return float(np.linalg.norm(self.delta_log))

    @property
    def label_name(self) -> str:
        """The label as it is printed in reports."""
        return LABEL_NAMES[self.label]


def generate_cases(seeds: tuple[int, ...] | list[int], kind: str = "single") -> tuple[Case, ...]:
    """Deterministic cases of one kind, one per seed.

    ``equivalent`` returns the partners of the ``single`` cases built from the
    same seeds, so ``generate_cases(s, "single")[i]`` and
    ``generate_cases(s, "equivalent")[i]`` are the two members of one pair and
    share a noise seed.
    """
    if kind not in KINDS:
        raise ValueError(f"unknown case kind {kind!r}; expected one of {KINDS}")
    if not len(seeds):
        raise ValueError("at least one seed is needed")
    cases = []
    for seed in seeds:
        if seed < 0:
            raise ValueError(f"seeds must not be negative, got {seed!r}")
        generator = np.random.default_rng(seed)
        index = int(generator.integers(0, len(PARAMETER_NAMES)))
        magnitude = float(generator.uniform(MIN_MAGNITUDE, MAX_MAGNITUDE))
        weak = float(generator.uniform(WEAK_MIN_MAGNITUDE, WEAK_MAX_MAGNITUDE))
        sign = 1.0 if generator.random() < 0.5 else -1.0
        amount = sign * magnitude
        axis = np.zeros(5)
        axis[index] = 1.0
        if kind == "single":
            cases.append(Case(int(seed), kind, index, amount * axis))
        elif kind == "weak":
            cases.append(Case(int(seed), kind, index, sign * weak * axis))
        elif kind == "equivalent":
            cases.append(
                Case(int(seed), kind, MULTI, amount * (axis - 1.0), partner_seed=int(seed))
            )
        elif kind == "scale":
            cases.append(Case(int(seed), kind, UNIDENTIFIABLE, amount * np.ones(5)))
        else:
            cases.append(Case(int(seed), kind, UNIDENTIFIABLE, np.zeros(5)))
    return tuple(cases)


def observe(telemetry: TelemetrySet, case: Case) -> np.ndarray:
    """Noisy telemetry of one case. The noise is a function of the case seed alone.

    A ``single`` case and its ``equivalent`` partner therefore draw the SAME
    noise, which is what makes the pair test an exact statement about the
    classifier rather than a statistical one.
    """
    return measure(telemetry, case.parameters, seed=case.seed)[0]
