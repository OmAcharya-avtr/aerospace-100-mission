"""YAML case files: a crosslink, a beam and a declared budget configuration.

A case is the unit of input for the command-line interface. It exists so that
the geometry and the declared uncertainties travel together: a point-ahead
number without the velocity knowledge it assumed, the ephemeris age it assumed
and the beam it is judged against is not reproducible, and three of the four
are the kind of thing that gets dropped when a number is pasted into a slide.

Schema, with units in the key names where they are not SI-obvious::

    label: leo_crossing
    frame: unspecified Earth-centred inertial
    trajectory_model: kepler          # or linear
    mu_m3_s2: 3.986004418e14          # optional, defaults to WGS-84 Earth
    local:
      position_km: [6978.137, 0.0, 0.0]
      velocity_m_s: [0.0, 7557.9, 0.0]
    remote:
      position_km: [...]
      velocity_m_s: [...]
    beam:
      wavelength_nm: 1550.0
      waist_mm: 25.0
    budget:
      sigma_velocity_per_axis_m_s: 0.05
      ephemeris_age_s: 30.0
      light_time_iterations: 1
      aberration_route: momentum_boost
      angle_method: closed_form
      loss_budget_db: 1.0

Positions are given in kilometres and velocities in metres per second because
that is the form a state vector arrives in; both are converted to SI at load.
Every key is required except ``mu_m3_s2`` and the ``frame`` and ``label``
strings. An unknown key is an error, not a warning: a silently ignored
``sigma_velocity_m_s`` that should have been
``sigma_velocity_per_axis_m_s`` would change the headline number without
changing anything a reader could see.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from .budget import BudgetInputs
from .kepler import MU_EARTH
from .terminal import BeamContext, Crosslink, TerminalState

_TOP_KEYS = {"label", "frame", "trajectory_model", "mu_m3_s2", "local", "remote", "beam", "budget"}
_STATE_KEYS = {"position_km", "velocity_m_s"}
_BEAM_KEYS = {"wavelength_nm", "waist_mm"}
_BUDGET_KEYS = {
    "sigma_velocity_per_axis_m_s",
    "ephemeris_age_s",
    "light_time_iterations",
    "aberration_route",
    "angle_method",
    "loss_budget_db",
}


@dataclass(frozen=True)
class Case:
    """A loaded case: the link, the beam and the declared budget inputs."""

    name: str
    link: Crosslink
    inputs: BudgetInputs
    source: str


def _require(mapping: object, keys: set[str], where: str) -> dict[str, Any]:
    if not isinstance(mapping, dict):
        raise ValueError(f"{where} must be a mapping, got {type(mapping).__name__}")
    unknown = set(mapping) - keys
    if unknown:
        raise ValueError(
            f"{where} has unknown key(s) {sorted(unknown)}; allowed keys are {sorted(keys)}"
        )
    return mapping


def _state(mapping: object, name: str, where: str) -> TerminalState:
    data = _require(mapping, _STATE_KEYS, where)
    missing = _STATE_KEYS - set(data)
    if missing:
        raise ValueError(f"{where} is missing {sorted(missing)}")
    position_km = data["position_km"]
    try:
        position_m = [float(value) * 1.0e3 for value in position_km]
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{where}.position_km must be three numbers, got {position_km!r}") from exc
    return TerminalState(position_m, data["velocity_m_s"], 0.0, name)


def load_case(path: str | Path) -> Case:
    """Load one case from a YAML file.

    Raises
    ------
    FileNotFoundError
        If the path does not exist.
    ValueError
        On any schema violation, with the offending key named.
    """
    location = Path(path)
    text = location.read_text()
    document = yaml.safe_load(text)
    data = _require(document, _TOP_KEYS, f"case {location.name}")
    for key in ("local", "remote", "beam", "budget"):
        if key not in data:
            raise ValueError(f"case {location.name} is missing the {key!r} section")

    beam_data = _require(data["beam"], _BEAM_KEYS, f"{location.name}.beam")
    missing = _BEAM_KEYS - set(beam_data)
    if missing:
        raise ValueError(f"{location.name}.beam is missing {sorted(missing)}")
    beam = BeamContext(
        wavelength_m=float(beam_data["wavelength_nm"]) * 1.0e-9,
        waist_m=float(beam_data["waist_mm"]) * 1.0e-3,
        label=f"{beam_data['wavelength_nm']} nm, {beam_data['waist_mm']} mm waist",
    )

    budget_data = _require(data["budget"], _BUDGET_KEYS, f"{location.name}.budget")
    required = {"sigma_velocity_per_axis_m_s", "ephemeris_age_s", "light_time_iterations"}
    missing = required - set(budget_data)
    if missing:
        raise ValueError(f"{location.name}.budget is missing {sorted(missing)}")
    inputs = BudgetInputs(
        sigma_velocity_per_axis_mps=float(budget_data["sigma_velocity_per_axis_m_s"]),
        ephemeris_age_s=float(budget_data["ephemeris_age_s"]),
        light_time_iterations=int(budget_data["light_time_iterations"]),
        beam=beam,
        aberration_route=str(budget_data.get("aberration_route", "momentum_boost")),
        angle_method=str(budget_data.get("angle_method", "closed_form")),
        loss_budget_db=float(budget_data.get("loss_budget_db", 1.0)),
    )

    label = str(data.get("label", location.stem))
    link = Crosslink(
        local=_state(data["local"], "local", f"{location.name}.local"),
        remote=_state(data["remote"], "remote", f"{location.name}.remote"),
        trajectory_model=str(data.get("trajectory_model", "kepler")),
        mu=float(data.get("mu_m3_s2", MU_EARTH)),
        label=label,
        frame=str(data.get("frame", "unspecified inertial")),
    )
    return Case(name=label, link=link, inputs=inputs, source=location.name)


def case_to_mapping(case: Case) -> dict[str, Any]:
    """Round-trip a :class:`Case` back to the YAML mapping it came from.

    Used by ``validation/regenerate_cases.py`` to show that the shipped case
    files are what the loader reproduces, so a reader editing one can tell
    whether the loader agrees with their edit.
    """
    beam = case.inputs.beam
    return {
        "label": case.name,
        "frame": case.link.frame,
        "trajectory_model": case.link.trajectory_model,
        "mu_m3_s2": float(case.link.mu),
        "local": {
            "position_km": [float(value) / 1.0e3 for value in case.link.local.position_m],
            "velocity_m_s": [float(value) for value in case.link.local.velocity_mps],
        },
        "remote": {
            "position_km": [float(value) / 1.0e3 for value in case.link.remote.position_m],
            "velocity_m_s": [float(value) for value in case.link.remote.velocity_mps],
        },
        "beam": {
            "wavelength_nm": float(beam.wavelength_m) * 1.0e9,
            "waist_mm": float(beam.waist_m) * 1.0e3,
        },
        "budget": {
            "sigma_velocity_per_axis_m_s": float(case.inputs.sigma_velocity_per_axis_mps),
            "ephemeris_age_s": float(case.inputs.ephemeris_age_s),
            "light_time_iterations": int(case.inputs.light_time_iterations),
            "aberration_route": case.inputs.aberration_route,
            "angle_method": case.inputs.angle_method,
            "loss_budget_db": float(case.inputs.loss_budget_db),
        },
    }


def dump_case(case: Case) -> str:
    """Serialise a case to YAML text, keys in schema order."""
    return yaml.safe_dump(case_to_mapping(case), sort_keys=False, default_flow_style=False)
