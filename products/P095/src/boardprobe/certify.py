"""The choke point. One function may label a figure, and this is it.

Everything in this package that produces a labelled result -- the benchmark
harness, the statistics layer, the report writer and the command line --
reaches a label by calling :func:`certify` and by no other route. There is no
second path, and the structure rather than a convention is what makes that
true:

1. :class:`Certificate` cannot be constructed. Its ``__post_init__`` refuses
   any call that does not carry a module-private token, and the token is
   passed in exactly one place: inside :func:`certify`. ``Certificate(...)``
   from outside this module raises.
2. :func:`certify` takes an :class:`~boardprobe.readings.Attribution`, which
   can only be built by ``boardprobe.readings.attribute`` (from verified
   readings) or ``combine`` (from other attributions). Both verify exact types
   and seals. A caller cannot hand :func:`certify` a provenance of their own
   choosing.
3. The label is not an argument that is checked against the provenance; it is
   **derived** from it by :func:`label_for`, a total function on the lattice.
   The ``label`` argument exists only so the caller can state what they
   BELIEVE, and a disagreement raises :class:`LabelViolation` naming the
   channels responsible. A caller who wants to know rather than assert passes
   ``label_for(attribution.provenance)``.
4. :class:`Label` has no member that means "measured" for anything but
   ``LIVE``. There is nothing to pass.

The label mapping, in full
--------------------------
=================  ==========  =============================================
provenance         label       what the reader is entitled to conclude
=================  ==========  =============================================
``LIVE``           MEASURED    a device responded, on the platform named
``RECORDED``       REPLAYED    a device responded once, earlier, elsewhere
``SIMULATED``      SYNTHETIC   nothing was measured; a generator produced it
=================  ==========  =============================================

The platform is a second, independent fact
------------------------------------------
``MEASURED`` says a device responded. It does not say WHICH device, and the
difference is the whole reason none of the numbers in this repository is a
board measurement. Every certificate carries a
:class:`~boardprobe.platform_attest.PlatformAttestation`, and
:attr:`Certificate.headline` -- the only rendering of a label this package
produces -- is built so that an off-target measurement reads
``MEASURED OFF-TARGET on <machine>; the declared target is <board> and this is
<machine>``. A bare ``MEASURED`` is unreachable by construction, not by
discipline, and ``validation/validate_adversarial_label.py`` attacks that too.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass
from typing import Any

from .platform_attest import PlatformAttestation
from .provenance import Provenance
from .readings import Attribution


class Label(enum.Enum):
    """What a figure may be called. One member per lattice element, no others."""

    MEASURED = "MEASURED"
    REPLAYED = "REPLAYED"
    SYNTHETIC = "SYNTHETIC"

    def __str__(self) -> str:
        return self.value


#: The total function from provenance to label. There is no entry mapping
#: anything but ``LIVE`` to ``MEASURED``, and no other mapping exists in the
#: package.
_LABEL_FOR: dict[Provenance, Label] = {
    Provenance.LIVE: Label.MEASURED,
    Provenance.RECORDED: Label.REPLAYED,
    Provenance.SIMULATED: Label.SYNTHETIC,
}

#: The inverse, used only to phrase error messages.
_PROVENANCE_FOR: dict[Label, Provenance] = {
    label: provenance for provenance, label in _LABEL_FOR.items()
}


class LabelViolation(ValueError):
    """Raised when a label does not follow from the provenance of its inputs.

    The message names the offending channels, because an unactionable refusal
    gets worked around.
    """


class _Token:
    """Private construction token for :class:`Certificate`. Never exported."""

    __slots__ = ()


_TOKEN = _Token()


def label_for(provenance: Provenance) -> Label:
    """The one label a figure with this provenance may carry."""
    if type(provenance) is not Provenance:
        raise TypeError(
            f"provenance must be a boardprobe.Provenance member, got "
            f"{type(provenance).__name__}"
        )
    return _LABEL_FOR[provenance]


@dataclass(frozen=True, slots=True)
class Certificate:
    """A label, the attribution that earned it, and the platform it was earned on.

    Constructed only by :func:`certify`. A direct call raises ``TypeError``.
    """

    token: Any
    subject: str
    label: Label
    attribution: Attribution
    platform: PlatformAttestation

    def __post_init__(self) -> None:
        if self.token is not _TOKEN:
            raise TypeError(
                "boardprobe.Certificate cannot be constructed directly; it is produced "
                "only by boardprobe.certify(), which is the single point at which a "
                "figure in this package acquires a label. Call certify(subject, "
                "attribution, label, platform)."
            )
        if type(self.label) is not Label:
            raise TypeError(
                f"label must be a boardprobe.Label member, got {type(self.label).__name__}"
            )
        if type(self.attribution) is not Attribution:
            raise TypeError(
                "attribution must be a boardprobe.readings.Attribution, got "
                f"{type(self.attribution).__name__}"
            )
        if type(self.platform) is not PlatformAttestation:
            raise TypeError(
                "platform must be a boardprobe.PlatformAttestation, got "
                f"{type(self.platform).__name__}"
            )

    @property
    def provenance(self) -> Provenance:
        """Provenance of the attribution this certificate rests on."""
        return self.attribution.provenance

    @property
    def on_declared_target(self) -> bool:
        """``True`` only when the label was earned on the declared target board."""
        return self.platform.matches_target

    @property
    def is_hardware_measurement(self) -> bool:
        """``True`` only for a live reading taken on the declared target board.

        The conjunction, in one place, so that no caller has to remember to
        check both. ``False`` for every certificate this repository produces.
        """
        return self.label is Label.MEASURED and self.platform.matches_target

    @property
    def headline(self) -> str:
        """The only rendering of a label this package produces.

        A ``MEASURED`` certificate earned off the declared target renders as
        ``MEASURED OFF-TARGET`` with both model strings. There is no branch
        that yields a bare ``MEASURED`` on a non-matching platform.
        """
        if self.label is Label.MEASURED and not self.platform.matches_target:
            return (
                f"MEASURED OFF-TARGET on {self.platform.describe()}; "
                f"{self.platform.describe_mismatch()}"
            )
        if self.label is Label.MEASURED:
            return f"MEASURED on {self.platform.describe()}"
        return f"{self.label.value} ({self.attribution.describe()})"

    def to_mapping(self) -> dict[str, Any]:
        """A JSON-ready mapping. Carries the headline, never a bare label."""
        return {
            "subject": self.subject,
            "label": self.label.value,
            "headline": self.headline,
            "provenance": self.provenance.value,
            "on_declared_target": self.on_declared_target,
            "is_hardware_measurement": self.is_hardware_measurement,
            "reading_count": self.attribution.reading_count,
            "channels": list(self.attribution.channels),
            "weakest_channels": list(self.attribution.weakest),
            "source_ids": list(self.attribution.source_ids),
            "observed_platform": self.platform.observed_model,
            "declared_target": self.platform.target_model,
        }


def certify(
    subject: str,
    attribution: Attribution,
    label: Label,
    platform: PlatformAttestation,
) -> Certificate:
    """Label ``subject``, or refuse. The single choke point of this package.

    Parameters
    ----------
    subject:
        What is being labelled, e.g. ``"benchmark fft_64k mean duration"``.
    attribution:
        From ``boardprobe.readings.attribute`` or ``combine``. Its provenance
        decides the outcome; nothing else can.
    label:
        What the caller asserts. Must equal ``label_for(attribution.provenance)``.
        Pass ``label_for(...)`` to ask rather than assert.
    platform:
        From ``boardprobe.attest_platform``.

    Raises
    ------
    LabelViolation
        When ``label`` does not follow from the provenance. The message names
        the channels at the weakest provenance and the label that would have
        been correct.
    TypeError
        When any argument is of the wrong type, including a subclass.
    """
    if not isinstance(subject, str) or not subject.strip():
        raise TypeError("subject must be a non-empty string naming what is being labelled")
    if type(attribution) is not Attribution:
        raise TypeError(
            "attribution must be a boardprobe.readings.Attribution built by attribute() "
            f"or combine(), got {type(attribution).__module__}.{type(attribution).__name__}. "
            "A subclass is refused deliberately: it could override provenance."
        )
    if type(label) is not Label:
        raise TypeError(
            f"label must be a boardprobe.Label member, got {type(label).__name__}"
        )
    if type(platform) is not PlatformAttestation:
        raise TypeError(
            "platform must be a boardprobe.PlatformAttestation from attest_platform(), got "
            f"{type(platform).__name__}"
        )
    required = label_for(attribution.provenance)
    if label is not required:
        raise LabelViolation(
            f"{subject!r} cannot be labelled {label.value}: its inputs are "
            f"{attribution.provenance.value}, which may only be labelled {required.value}. "
            f"{label.value} requires every contributing reading to be "
            f"{_PROVENANCE_FOR[label].value}. "
            f"Offending channel(s) at {attribution.provenance.value}: "
            f"{', '.join(attribution.weakest)} "
            f"(from source(s) {', '.join(attribution.weakest_sources)}). "
            "Re-run the affected channel(s) against a live source, or label the figure "
            f"{required.value}."
        )
    return Certificate(
        token=_TOKEN,
        subject=subject,
        label=label,
        attribution=attribution,
        platform=platform,
    )
