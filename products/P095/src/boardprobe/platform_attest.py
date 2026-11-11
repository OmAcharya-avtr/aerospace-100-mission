"""Which machine a reading was taken on, as a separate fact from its provenance.

Provenance and platform are two independent questions and this package keeps
them apart, because conflating them is how a container timing figure ends up
quoted as a board measurement.

- **Provenance** answers *is this a reading of a device, a replay, or
  synthetic?* That is the lattice in ``boardprobe.provenance``.
- **Platform** answers *which device?* That is this module.

A live reading of the build host's monotonic clock is a genuine live reading of
a real clock. It is not a reading of a Jetson Orin Nano, and a figure derived
from it is not evidence about a Jetson Orin Nano. So every
:class:`~boardprobe.certify.Certificate` carries a
:class:`PlatformAttestation`, and a certificate whose platform does not match
the declared target **cannot be rendered without saying so**: the label on it
reads ``MEASURED OFF-TARGET`` and names both models. There is no code path that
produces a bare ``MEASURED``.

The attestation reads ``proc/device-tree/model``, which is where a device-tree
platform publishes its own name (Linux kernel ``Documentation/ABI/testing/sysfs-firmware-ofw``
and the device-tree specification's ``model`` property; the Jetson Linux
Developer Guide, "Platform Adaptation and Bring-Up", gives the Jetson strings).
An x86_64 machine has no such node, so the attestation falls back to
``platform.machine()`` and ``platform.system()`` and reports a non-match. That
is exactly what happened in the environment this package was built in, and it
is why every figure in this repository is off-target.
"""

from __future__ import annotations

import platform as _platform
from dataclasses import dataclass
from pathlib import Path

from .orin_nano import TARGET_MODEL_PREFIX

#: The device-tree node a platform publishes its own model string in, relative
#: to the filesystem root. Relative so that error text and committed output
#: never quote an absolute host path.
MODEL_LOCATOR = "proc/device-tree/model"


@dataclass(frozen=True, slots=True)
class PlatformAttestation:
    """What machine this process is running on, and whether it is the target."""

    target_model: str
    observed_model: str
    machine: str
    system: str
    source: str
    matches_target: bool

    def describe(self) -> str:
        """One line naming the machine, suitable for a certificate headline."""
        return f"{self.observed_model} ({self.machine} {self.system}, via {self.source})"

    def describe_mismatch(self) -> str:
        """Why this is not the target, or the empty string when it is."""
        if self.matches_target:
            return ""
        return (
            f"the declared target is {self.target_model!r} and this is "
            f"{self.observed_model!r}"
        )


def attest_platform(
    root: Path | str = Path("/"), target_model_prefix: str = TARGET_MODEL_PREFIX
) -> PlatformAttestation:
    """Identify this machine. Never raises; a missing node is a non-match, not an error.

    Parameters
    ----------
    root:
        Filesystem root to read ``proc/device-tree/model`` under. The default
        is the real root; a test or a dry run passes a temporary directory.
    target_model_prefix:
        Prefix the observed model must start with to count as the target.
        A prefix, not an equality, because the trailing text of a Jetson model
        string varies between carrier boards and L4T releases.

    Returns
    -------
    PlatformAttestation
        With ``matches_target`` false on anything that is not the declared
        board, including the case where the model node does not exist.
    """
    node = Path(root) / MODEL_LOCATOR
    observed = ""
    source = "unavailable"
    try:
        # Device tree strings are NUL-terminated; strip the terminator and any
        # surrounding whitespace before comparing.
        observed = node.read_bytes().decode("utf-8", "replace").strip("\x00").strip()
        source = MODEL_LOCATOR
    except OSError:
        observed = ""
    machine = _platform.machine()
    system = _platform.system()
    if not observed:
        observed = f"unidentified {machine} {system}"
        source = "platform.machine() fallback; no device-tree model node"
    return PlatformAttestation(
        target_model=target_model_prefix,
        observed_model=observed,
        machine=machine,
        system=system,
        source=source,
        matches_target=observed.startswith(target_model_prefix),
    )
