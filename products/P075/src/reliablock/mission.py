"""The mission profile: phases, duty cycles and stress factors.

A mission profile converts calendar time into an **effective age** per block,
and it does that in cumulative-hazard space because that is where the
conversion is a sum rather than a product of nearly-one numbers.

The model
---------

A phase declares a duration in hours, an optional set of blocks that are
powered during it, and a stress factor. A block's effective age over the
mission is

    ``age_eff = sum over phases of duration * duty_cycle * stress_factor``

and its cumulative hazard is then ``H(age_eff)`` from its own distribution.

This is the standard accelerated-age (or "equivalent operating hours")
treatment, and it is exact for an exponential block, because for a constant
hazard rate ``H`` is linear in age and the sum of the phase contributions is
the integral. For a **non-exponential** block it is an approximation, and a
specific one worth naming: it assumes the block ages along its own hazard curve
at a rate scaled by the stress, which is the cumulative-exposure (Nelson)
model. It is *not* the same as integrating a time-varying hazard rate in
calendar time, and the two differ whenever the shape parameter is not 1. The
difference is measured in ``validation/validate_mission_output.txt`` rather
than asserted to be small, and the model's name and the reference are:

Nelson, W. (1990), *Accelerated Testing: Statistical Models, Test Plans, and
Data Analyses*, Wiley, chapter 10, the cumulative-exposure model. No numerical
value in this repository is taken from it.

What a profile refuses to do
----------------------------

A phase cannot age a block whose distribution is
:class:`~reliablock.distributions.FixedProbability`. Such a block's figure of
merit is a probability per actuation, and multiplying it by a duty cycle would
be treating a probability as a rate. A profile that names such a block in
``powered`` with a duty cycle other than 1.0, or applies a stress factor other
than 1.0 to it, raises rather than quietly producing a number.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml

from .distributions import FixedProbability

if TYPE_CHECKING:  # pragma: no cover - import cycle only for type checking
    from .diagram import Diagram

__all__ = ["MissionPhase", "MissionProfile", "load_mission"]


@dataclass(frozen=True)
class MissionPhase:
    """One phase of a mission profile.

    Parameters
    ----------
    name
        Phase identifier, used in the per-phase breakdown.
    duration_hours
        Phase duration in hours, ``>= 0``.
    duty_cycles
        Per-block duty cycle in ``[0, 1]``, the fraction of the phase for which
        the block is powered. A block absent from the mapping takes
        ``default_duty_cycle``.
    stress_factors
        Per-block multiplier on the hazard accrual rate, ``>= 0``,
        dimensionless. A block absent from the mapping takes 1.0. A factor of
        2.0 means the block ages at twice its nominal rate during this phase.
    default_duty_cycle
        Duty cycle for a block not named in ``duty_cycles``. Defaults to 1.0,
        so a profile that names nothing is a plain elapsed-time profile.
    """

    name: str
    duration_hours: float
    duty_cycles: dict[str, float] | None = None
    stress_factors: dict[str, float] | None = None
    default_duty_cycle: float = 1.0

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError(f"phase name must be a non-empty string, got {self.name!r}")
        duration = float(self.duration_hours)
        if duration < 0.0:
            raise ValueError(f"phase {self.name!r}: duration_hours must be >= 0, got {duration}")
        if not 0.0 <= float(self.default_duty_cycle) <= 1.0:
            raise ValueError(
                f"phase {self.name!r}: default_duty_cycle must lie in [0, 1], "
                f"got {self.default_duty_cycle!r}"
            )
        for block, value in (self.duty_cycles or {}).items():
            if not 0.0 <= float(value) <= 1.0:
                raise ValueError(
                    f"phase {self.name!r}: duty cycle for {block!r} must lie in "
                    f"[0, 1], got {value!r}"
                )
        for block, value in (self.stress_factors or {}).items():
            if float(value) < 0.0:
                raise ValueError(
                    f"phase {self.name!r}: stress factor for {block!r} must be "
                    f">= 0, got {value!r}"
                )

    def duty(self, block_name: str) -> float:
        """Duty cycle for ``block_name`` in this phase, dimensionless."""
        return float((self.duty_cycles or {}).get(block_name, self.default_duty_cycle))

    def stress(self, block_name: str) -> float:
        """Stress factor for ``block_name`` in this phase, dimensionless."""
        return float((self.stress_factors or {}).get(block_name, 1.0))

    def contribution(self, block_name: str) -> float:
        """Effective age added by this phase, in hours."""
        return float(self.duration_hours) * self.duty(block_name) * self.stress(block_name)

    def to_mapping(self) -> dict[str, Any]:
        """Round-trippable declaration of this phase."""
        body: dict[str, Any] = {"name": self.name, "duration_hours": float(self.duration_hours)}
        if self.duty_cycles:
            body["duty_cycles"] = {k: float(v) for k, v in self.duty_cycles.items()}
        if self.stress_factors:
            body["stress_factors"] = {k: float(v) for k, v in self.stress_factors.items()}
        if self.default_duty_cycle != 1.0:
            body["default_duty_cycle"] = float(self.default_duty_cycle)
        return body


@dataclass(frozen=True)
class MissionProfile:
    """An ordered sequence of mission phases."""

    name: str = "mission"
    phases: tuple[MissionPhase, ...] = ()

    def __post_init__(self) -> None:
        if not self.phases:
            raise ValueError(f"mission profile {self.name!r} declares no phases")

    @classmethod
    def from_hours(cls, hours: float) -> MissionProfile:
        """A single-phase profile of ``hours`` at full duty and unit stress."""
        duration = float(hours)
        if duration < 0.0:
            raise ValueError(f"mission duration must be >= 0, got {duration}")
        return cls(name=f"{duration:g} h", phases=(MissionPhase("mission", duration),))

    @property
    def total_hours(self) -> float:
        """Total elapsed mission duration in hours."""
        return float(sum(p.duration_hours for p in self.phases))

    def effective_ages(self, diagram: Diagram) -> dict[str, float]:
        """Effective age in hours for every block of ``diagram``.

        Raises ``ValueError`` if a phase would age a one-shot block, whose
        figure of merit is a probability per actuation rather than a rate.
        """
        ages: dict[str, float] = {}
        for block in diagram.blocks():
            if not block.distribution.ages_with_time:
                offenders = [
                    phase.name
                    for phase in self.phases
                    if phase.duty(block.name) != 1.0 or phase.stress(block.name) != 1.0
                ]
                if offenders:
                    raise ValueError(
                        f"block {block.name!r} has a {type(block.distribution).__name__} "
                        f"distribution, whose figure of merit is a probability per "
                        f"actuation and not a rate, but phases {offenders} declare a "
                        "duty cycle or stress factor for it. Multiplying a "
                        "probability by a duty cycle is not a model of anything. "
                        "Remove the entry, or declare the block with a rate."
                    )
                ages[block.name] = self.total_hours
                continue
            ages[block.name] = float(sum(p.contribution(block.name) for p in self.phases))
        return ages

    def breakdown(self, diagram: Diagram) -> list[dict[str, Any]]:
        """Per-phase, per-block effective-age contributions, for reporting."""
        rows: list[dict[str, Any]] = []
        for phase in self.phases:
            for block in diagram.blocks():
                rows.append(
                    {
                        "phase": phase.name,
                        "block": block.name,
                        "duration_hours": float(phase.duration_hours),
                        "duty_cycle": phase.duty(block.name),
                        "stress_factor": phase.stress(block.name),
                        "effective_hours": phase.contribution(block.name),
                    }
                )
        return rows

    def to_mapping(self) -> dict[str, Any]:
        """Round-trippable declaration of the profile."""
        return {"name": self.name, "phases": [p.to_mapping() for p in self.phases]}

    def to_yaml(self) -> str:
        """The declaration as YAML, sorted so that two profiles diff cleanly."""
        return yaml.safe_dump(self.to_mapping(), sort_keys=True, default_flow_style=False)


_PHASE_KEYS = {
    "name",
    "duration_hours",
    "duty_cycles",
    "stress_factors",
    "default_duty_cycle",
}


def load_mission(path: str | Path) -> MissionProfile:
    """Load a mission profile from a YAML file, rejecting unknown keys."""
    body = yaml.safe_load(Path(path).read_text())
    if not isinstance(body, dict):
        raise ValueError(f"{path}: a mission profile must be a mapping at the top level")
    unknown = set(body) - {"name", "phases"}
    if unknown:
        raise ValueError(
            f"{path}: mission profile does not accept {sorted(unknown)}; "
            "accepted keys are ['name', 'phases']"
        )
    if "phases" not in body:
        raise ValueError(f"{path}: mission profile requires 'phases'")
    phases: list[MissionPhase] = []
    for entry in body["phases"]:
        if not isinstance(entry, dict):
            raise ValueError(f"{path}: each phase must be a mapping, got {entry!r}")
        extra = set(entry) - _PHASE_KEYS
        if extra:
            raise ValueError(
                f"{path}: phase does not accept {sorted(extra)}; "
                f"accepted keys are {sorted(_PHASE_KEYS)}"
            )
        for required in ("name", "duration_hours"):
            if required not in entry:
                raise ValueError(f"{path}: phase requires {required!r}")
        phases.append(
            MissionPhase(
                name=str(entry["name"]),
                duration_hours=float(entry["duration_hours"]),
                duty_cycles=dict(entry.get("duty_cycles") or {}),
                stress_factors=dict(entry.get("stress_factors") or {}),
                default_duty_cycle=float(entry.get("default_duty_cycle", 1.0)),
            )
        )
    return MissionProfile(name=str(body.get("name", "mission")), phases=tuple(phases))


def _fixed_probability_note() -> str:
    """One line naming the distribution a profile refuses to age."""
    return f"{FixedProbability.__name__} blocks are not aged by a mission profile"
