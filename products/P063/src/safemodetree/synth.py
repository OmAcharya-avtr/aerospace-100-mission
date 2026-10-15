"""Random declaration generator, for cross-checking the checker.

This module exists so the deterministic core can be compared against brute
force on many small declarations rather than on a handful of hand-built ones.
It generates structurally valid declarations whose CONTENT is meaningless: the
modes, faults and responses are named ``m0``, ``f0``, ``r0`` and carry random
costs. Nothing here is a model of anything.

Used by ``tests/test_properties.py`` and
``validation/validate_reachability.py``. Kept in the package rather than in the
test tree so both can import it without path manipulation.
"""

from __future__ import annotations

import numpy as np

from .spec import (
    Condition,
    Fault,
    Literal,
    Mode,
    ObservabilityModel,
    ObservationChannel,
    Precondition,
    Response,
    SafeModeSpec,
)

__all__ = ["random_declaration"]

_SEVERITIES = ("I", "II", "III", "IV")


def _channel(index: int) -> ObservationChannel:
    return ObservationChannel(
        id=f"c{index}",
        unit="dimensionless",
        nominal=0.0,
        threshold=1.0,
        faulted=2.0,
        direction="above",
        noise_sd=0.5,
    )


def random_declaration(
    seed: int,
    *,
    n_modes: int = 4,
    n_faults: int = 3,
    n_responses: int = 6,
    n_channels: int = 3,
    absorbing_probability: float = 0.25,
    always_false_probability: float = 0.3,
) -> SafeModeSpec:
    """Generate a structurally valid random declaration.

    Args:
        seed: seed for ``numpy.random.default_rng``; the result is a
            deterministic function of the seed and the arguments.
        n_modes: number of modes, >= 2. Exactly one is forced safe; later ones
            are safe with probability 1/3.
        n_faults: number of faults, >= 1.
        n_responses: number of responses, >= 0.
        n_channels: number of observation channels, >= 1. Signatures are drawn
            uniformly from the non-zero bit vectors of this width, so
            collisions -- ambiguity classes -- occur at the rate the width
            implies.
        absorbing_probability: chance that a non-safe mode after the first is
            declared absorbing.
        always_false_probability: chance that the second declared condition is
            declared always_false.

    Returns:
        A :class:`safemodetree.spec.SafeModeSpec`.
    """
    if n_modes < 2:
        raise ValueError(f"n_modes must be >= 2, got {n_modes}")
    if n_faults < 1:
        raise ValueError(f"n_faults must be >= 1, got {n_faults}")
    if n_responses < 0:
        raise ValueError(f"n_responses must be >= 0, got {n_responses}")
    if n_channels < 1:
        raise ValueError(f"n_channels must be >= 1, got {n_channels}")

    rng = np.random.default_rng(seed)
    safe_index = int(rng.integers(n_modes))
    modes: list[Mode] = []
    for index in range(n_modes):
        safe = index == safe_index or bool(rng.random() < 1.0 / 3.0)
        absorbing = (not safe) and index != 0 and bool(rng.random() < absorbing_probability)
        modes.append(Mode(id=f"m{index}", safe=safe, absorbing=absorbing))
    mode_ids = [m.id for m in modes]
    operational = [m.id for m in modes if not m.absorbing] or [mode_ids[0]]

    conditions = [
        Condition(
            id="k0",
            available_in_modes=tuple(
                sorted(
                    {
                        mode_ids[int(i)]
                        for i in rng.choice(n_modes, size=max(1, n_modes // 2), replace=True)
                    }
                )
            ),
        ),
        Condition(id="k1", always_false=bool(rng.random() < always_false_probability)),
    ]

    faults = [
        Fault(
            id=f"f{index}",
            effect="random",
            severity=_SEVERITIES[int(rng.integers(len(_SEVERITIES)))],
            detectable=bool(rng.random() < 0.85),
            detection_latency_s=float(rng.integers(0, 200)),
            occurs_in_modes=tuple(operational),
        )
        for index in range(n_faults)
    ]

    responses: list[Response] = []
    for index in range(n_responses):
        fault = None if rng.random() < 0.35 else f"f{int(rng.integers(n_faults))}"
        from_mode = None if rng.random() < 0.4 else mode_ids[int(rng.integers(n_modes))]
        literals: list[Literal] = []
        if rng.random() < 0.5:
            literals.append(
                Literal(f"k{int(rng.integers(2))}", positive=bool(rng.random() < 0.8))
            )
        responses.append(
            Response(
                id=f"r{index}",
                fault=fault,
                from_mode=from_mode,
                to_mode=mode_ids[int(rng.integers(n_modes))],
                cost_min=float(rng.integers(0, 100)),
                clears_fault=bool(rng.random() < 0.4),
                precondition=Precondition(tuple(literals)),
            )
        )

    channels = tuple(_channel(index) for index in range(n_channels))
    signatures: dict[str, tuple[int, ...]] = {}
    for fault in faults:
        while True:
            bits = tuple(int(b) for b in rng.integers(0, 2, size=n_channels))
            if any(bits):
                signatures[fault.id] = bits
                break

    return SafeModeSpec(
        name=f"random_{seed}",
        modes=tuple(modes),
        faults=tuple(faults),
        responses=tuple(responses),
        conditions=tuple(conditions),
        observability=ObservabilityModel(channels=channels, signatures=signatures),
        initial_mode=operational[0],
    )
