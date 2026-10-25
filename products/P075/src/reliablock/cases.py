"""Built-in declared diagrams, and the mission profiles that go with them.

These exist so that the CLI, the examples and the validation scripts all drive
the same declarations, and so that a reader can run something without writing
YAML first. Each is also written out to ``examples/diagrams/`` by
``validation/regenerate_cases.py``, and the committed YAML is checked to
round-trip, so the two can never drift.

Every rate below is a **declared input chosen to be representative in order of
magnitude**, not a figure from a parts database or a published handbook. No
number in this module is a measurement, and the README says so. A real
assessment needs the programme's own rates, and the point of this package is
that the diagram carrying them is a reviewable artifact.
"""

from __future__ import annotations

from .diagram import Block, Diagram, KOfN, Parallel, Series, SharedLoad, Standby
from .distributions import Exponential, FixedProbability, Weibull
from .mission import MissionPhase, MissionProfile

__all__ = ["CASES", "MISSIONS", "case", "mission"]


def _exponential_block(name: str, rate_per_hour: float, description: str = "") -> Block:
    """A block with a constant hazard rate in 1/hour."""
    return Block(name=name, distribution=Exponential(rate_per_hour), description=description)


def _smallsat_bus(beta_reaction_wheels: float = 0.08, beta_receivers: float = 0.05) -> Diagram:
    """A small-satellite bus: the diagram the headline comparison runs on.

    Structure, from the top: a series chain of the subsystems that have no
    redundancy, with three redundant groups.

    - **power**: two strings in parallel, a declared beta factor over them
      because they share a harness, a connector type and one qualification
      campaign.
    - **reaction wheels**: 3-of-4, a declared beta factor because four wheels
      from one lot share a bearing lubricant and a lot-level workmanship risk.
    - **receivers**: two in parallel with a declared beta factor, because both
      sit behind one antenna and one thermal environment.
    - **transmitter**: a cold standby pair behind a changeover switch whose
      reliability is declared rather than assumed.

    The deployment mechanism is a one-shot item with a declared per-actuation
    probability, so a mission profile cannot age it.
    """
    power = Parallel(
        items=(
            _exponential_block("power_string_a", 1.2e-6, "PCDU string A"),
            _exponential_block("power_string_b", 1.2e-6, "PCDU string B"),
        ),
        beta_factor=0.12,
        label="power",
    )
    wheels = KOfN(
        k=3,
        items=tuple(
            _exponential_block(f"wheel_{suffix}", 1.0e-6, f"reaction wheel {suffix.upper()}")
            for suffix in ("a", "b", "c", "d")
        ),
        beta_factor=beta_reaction_wheels,
        label="reaction_wheels",
    )
    receivers = Parallel(
        items=(
            _exponential_block("receiver_a", 2.0e-6, "S-band receiver A"),
            _exponential_block("receiver_b", 2.0e-6, "S-band receiver B"),
        ),
        beta_factor=beta_receivers,
        label="receivers",
    )
    transmitter = Standby(
        primary=_exponential_block("transmitter_main", 4.0e-6, "X-band TWTA, prime"),
        standbys=(_exponential_block("transmitter_spare", 4.0e-6, "X-band TWTA, cold spare"),),
        switch_reliability=0.995,
        standby_hazard_fraction=0.0,
        label="transmitter",
    )
    return Diagram(
        name="smallsat bus",
        root=Series(
            items=(
                Block(
                    name="solar_array_deployment",
                    distribution=FixedProbability(1.0e-3),
                    description="one-shot deployment; probability per actuation, not a rate",
                ),
                power,
                Block(
                    name="obc",
                    distribution=Weibull(shape=0.85, scale_hours=5.0e6),
                    description="on-board computer, single string, infant-mortality shape",
                ),
                wheels,
                receivers,
                transmitter,
                _exponential_block("thermal_control", 2.0e-7, "heater control, single string"),
            ),
            label="bus",
        ),
        description=(
            "A small-satellite bus with four redundant groups and three declared "
            "common-cause beta factors. Every rate is a declared input chosen to be "
            "representative in order of magnitude; none is a measurement."
        ),
        notes=(
            "The beta factors are declared inputs, not data. A real assessment "
            "needs the programme's own common-cause parameters.",
            "The transmitter's changeover switch reliability is declared, not "
            "assumed to be 1.0.",
        ),
    )


def _redundant_pair(beta: float = 0.1) -> Diagram:
    """The smallest diagram that shows the point: one redundant pair, one beta."""
    return Diagram(
        name="redundant pair",
        root=Parallel(
            items=(
                _exponential_block("unit_a", 2.0e-6),
                _exponential_block("unit_b", 2.0e-6),
            ),
            beta_factor=beta,
            label="pair",
        ),
        description=(
            "Two identical units in parallel with a declared beta factor. The "
            "smallest diagram on which the headline comparison says something."
        ),
    )


def _voting_triple() -> Diagram:
    """A 2-of-3 voting group behind a single voter: the voter is the cut set of one."""
    return Diagram(
        name="voting triple",
        root=Series(
            items=(
                KOfN(
                    k=2,
                    items=tuple(
                        _exponential_block(f"channel_{s}", 3.0e-6) for s in ("a", "b", "c")
                    ),
                    beta_factor=0.06,
                    label="channels",
                ),
                _exponential_block("voter", 5.0e-7, "majority voter, single string"),
            ),
            label="triple_modular_redundancy",
        ),
        description=(
            "Triple modular redundancy with a single voter. The voter is a cut set "
            "of order one whatever the channels do, which is the standard objection "
            "to the architecture and is visible in the cut-set listing."
        ),
    )


def _load_sharing_pair() -> Diagram:
    """Two load-sharing units: dependent by construction, no beta factor needed."""
    return Diagram(
        name="load sharing pair",
        root=SharedLoad(
            units=(
                _exponential_block("pump_a", 5.0e-5, "full-load rate"),
                _exponential_block("pump_b", 5.0e-5, "full-load rate"),
            ),
            shared_rate_factor=0.45,
            label="pumps",
        ),
        description=(
            "Two units sharing a load at 45 percent of their full-load hazard rate "
            "each. Dependent by construction: no beta factor is declared or "
            "accepted, because the dependence is already in the model."
        ),
        notes=(
            "Importance measures are refused for this diagram: the quantification "
            "is not the independent-event arithmetic over its structure function.",
        ),
    )


def _warm_standby() -> Diagram:
    """A warm standby pair: the hazard fraction and the switch are both declared."""
    return Diagram(
        name="warm standby",
        root=Standby(
            primary=_exponential_block("prime", 3.0e-5),
            standbys=(_exponential_block("spare", 3.0e-5),),
            switch_reliability=0.97,
            standby_hazard_fraction=0.35,
            label="warm_pair",
        ),
        description=(
            "A spare that is powered and therefore ageing at 35 percent of its "
            "operating hazard rate while it waits, behind a changeover that "
            "succeeds with probability 0.97."
        ),
    )


#: Built-in diagrams, by name.
CASES: dict[str, Diagram] = {
    "smallsat_bus": _smallsat_bus(),
    "redundant_pair": _redundant_pair(),
    "voting_triple": _voting_triple(),
    "load_sharing_pair": _load_sharing_pair(),
    "warm_standby": _warm_standby(),
}


#: Built-in mission profiles, by name.
MISSIONS: dict[str, MissionProfile] = {
    "one_year": MissionProfile.from_hours(8766.0),
    "three_year": MissionProfile.from_hours(3.0 * 8766.0),
    "leo_phased": MissionProfile(
        name="LEO phased, 3 years",
        phases=(
            MissionPhase(
                name="launch_and_leop",
                duration_hours=72.0,
                stress_factors={
                    "power_string_a": 3.0,
                    "power_string_b": 3.0,
                    "obc": 2.0,
                    "thermal_control": 2.5,
                },
                duty_cycles={
                    "wheel_a": 0.1,
                    "wheel_b": 0.1,
                    "wheel_c": 0.1,
                    "wheel_d": 0.1,
                    "transmitter_main": 0.3,
                    "transmitter_spare": 0.3,
                },
            ),
            MissionPhase(
                name="commissioning",
                duration_hours=720.0,
                duty_cycles={"transmitter_main": 0.5, "transmitter_spare": 0.5},
            ),
            MissionPhase(
                name="routine_operations",
                duration_hours=3.0 * 8766.0 - 792.0,
                duty_cycles={
                    "transmitter_main": 0.12,
                    "transmitter_spare": 0.12,
                    "receiver_a": 1.0,
                    "receiver_b": 1.0,
                },
            ),
        ),
    ),
}


def case(name: str) -> Diagram:
    """Return the built-in diagram called ``name``."""
    if name not in CASES:
        raise KeyError(f"unknown case {name!r}; known cases are {sorted(CASES)}")
    return CASES[name]


def mission(name: str) -> MissionProfile:
    """Return the built-in mission profile called ``name``."""
    if name not in MISSIONS:
        raise KeyError(f"unknown mission {name!r}; known missions are {sorted(MISSIONS)}")
    return MISSIONS[name]
