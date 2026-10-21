"""Declared conjunction cases in YAML, loaded without inventing anything.

A case file declares two states, their position covariances and a combined
hard-body radius. Covariances may be given either as a full 3x3 inertial matrix
(``covariance_m2``) or as radial / along-track / cross-track standard
deviations (``sigma_rtn_m``), which is how they are usually quoted.

Every value in a shipped case file under ``examples/cases/`` is a declared
input of a synthetic scenario. None was identified from a catalogue, an
operator's screening output or any vehicle.

Units in the file are SI and the keys say so: ``position_m``, ``velocity_mps``,
``covariance_m2``, ``sigma_rtn_m``, ``hard_body_radius_m``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from .frames import covariance_from_rtn_sigmas
from .geometry import ObjectState

_REQUIRED_TOP = ("name", "hard_body_radius_m", "primary", "secondary")


@dataclass(frozen=True)
class ConjunctionCase:
    """One declared close approach.

    Attributes
    ----------
    name
        Case identifier from the file.
    primary, secondary
        States with covariance, metres and metres per second.
    radius_m
        Combined hard-body radius, metres.
    description
        Free text from the file, or the empty string.
    source_path
        Path the case was read from, as given by the caller.
    """

    name: str
    primary: ObjectState
    secondary: ObjectState
    radius_m: float
    description: str
    source_path: Path


def _state_from_mapping(mapping: Any, label: str) -> ObjectState:
    if not isinstance(mapping, dict):
        raise ValueError(f"{label} must be a mapping, got {type(mapping).__name__}")
    for key in ("position_m", "velocity_mps"):
        if key not in mapping:
            raise ValueError(f"{label} is missing required key {key!r}")
    position = np.asarray(mapping["position_m"], dtype=float)
    velocity = np.asarray(mapping["velocity_mps"], dtype=float)
    has_full = "covariance_m2" in mapping
    has_rtn = "sigma_rtn_m" in mapping
    if has_full == has_rtn:
        raise ValueError(
            f"{label} must declare exactly one of 'covariance_m2' or 'sigma_rtn_m'"
        )
    if has_full:
        covariance = np.asarray(mapping["covariance_m2"], dtype=float)
    else:
        covariance = covariance_from_rtn_sigmas(
            position,
            velocity,
            mapping["sigma_rtn_m"],
            correlation_ra=float(mapping.get("correlation_radial_along_track", 0.0)),
        )
    return ObjectState(
        position_m=position,
        velocity_mps=velocity,
        covariance_m2=covariance,
        name=str(mapping.get("name", label)),
    )


def load_case(path: str | Path) -> ConjunctionCase:
    """Load a declared conjunction case from a YAML file.

    Parameters
    ----------
    path
        Path to the YAML file. Relative paths resolve against the current
        working directory, which is the caller's business; every validation
        script in this repository changes directory to the repository root
        before using a relative path so that it works from anywhere.

    Returns
    -------
    ConjunctionCase

    Raises
    ------
    FileNotFoundError
        If the file does not exist. The message quotes the path as given.
    ValueError
        If a required key is missing, a covariance is declared twice or not at
        all, or the hard-body radius is not positive.
    """
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"conjunction case file not found: {source}")
    with source.open("r", encoding="utf-8") as handle:
        document = yaml.safe_load(handle)
    if not isinstance(document, dict):
        raise ValueError(f"case file {source} must contain a YAML mapping at the top level")
    missing = [key for key in _REQUIRED_TOP if key not in document]
    if missing:
        raise ValueError(f"case file {source} is missing required keys: {missing}")
    radius = float(document["hard_body_radius_m"])
    if not radius > 0.0:
        raise ValueError(f"hard_body_radius_m must be positive metres, got {radius!r}")
    return ConjunctionCase(
        name=str(document["name"]),
        primary=_state_from_mapping(document["primary"], "primary"),
        secondary=_state_from_mapping(document["secondary"], "secondary"),
        radius_m=radius,
        description=str(document.get("description", "")),
        source_path=source,
    )
