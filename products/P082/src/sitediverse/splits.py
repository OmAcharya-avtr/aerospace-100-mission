"""Hold-out splits: held-out SITES and held-out TIME, kept separate throughout.

The specification requires two scorings reported separately, because a
spatio-temporal model can pass one and fail the other. This module defines them
and nothing else, so that there is exactly one place where the question "what
was this model allowed to see" is answered.

**(29) The two-way split.** The record is cut in half by TIME at its midpoint
and the stations are cut into a training set and a held-out set. A row of the
pair-regime table of ``sitediverse.features`` is identified by
``(station i, station j, regime, time half)``, and the three disjoint sets are

    TRAIN            : both stations trained,  first time half
    held-out SITES   : at least one station held out, first time half
    held-out TIME    : both stations trained,  second time half

The fourth cell -- held-out stations in the second half -- is deliberately NOT
scored and not reported as a third number. It is a joint extrapolation in both
directions and a score on it cannot be attributed to either, which is exactly
the confusion the specification asks to avoid. It is computed and reported in
``validation/validate_learned_model.py`` as a fourth, clearly labelled cell, so
that a reader can see it exists and see that it is not mixed into either headline
score.

**(30) What the split does and does not prevent.** Held-out SITES is a genuine
spatial extrapolation: the fitted model has never seen either of a held-out
pair's stations, so it has never seen their separation, their orientation or
their marginal rates. Held-out TIME is a genuine temporal extrapolation for the
TARGET -- the implied correlation of equation (11) is recomputed on the second
half -- but it is NOT independent of the training data in one respect worth
naming: the marginal climatology of ``sitediverse.climatology`` is estimated
once over the WHOLE record and is used in both halves. Estimating it twice would
make the two halves' targets incomparable, since each would be inverted against
a different marginal model. The leak is the climatology only, it is the same in
every model being compared including the baselines, and it is stated here and in
README.md rather than left implicit.

**(31) Which stations are held out.** One station from each of the five declared
regional groups, the fourth of each. Not five from one group: a held-out set
drawn from one group would contain only short baselines, and a model scored on
it would be scored on a quarter of the separation range. The declared held-out
set spans separations from 115 km to 7300 km, which is the full range the
training set spans, and the figure is reported in
``validation/validate_splits_output.txt``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

#: Indices of the declared held-out stations, equation (31): the fourth station
#: of each of the five declared regional groups.
HELD_OUT_SITES: tuple[int, ...] = (3, 7, 11, 15, 19)

#: Names of the three scored cells of equation (29), in report order.
CELL_NAMES: tuple[str, ...] = ("train", "held-out SITES", "held-out TIME")

#: The unscored fourth cell of equation (29), named so that it is visible.
UNSCORED_CELL: str = "held-out SITES and TIME (not a headline score)"


@dataclass(frozen=True)
class Split:
    """The equation (29) split for one record.

    Attributes
    ----------
    epochs:
        Record length.
    midpoint:
        First epoch of the second time half.
    held_out_sites:
        Station indices held out.
    site_count:
        Total number of stations.
    """

    epochs: int
    midpoint: int
    held_out_sites: tuple[int, ...]
    site_count: int

    @property
    def trained_sites(self) -> tuple[int, ...]:
        """Station indices the models are allowed to fit on."""
        return tuple(
            site for site in range(self.site_count) if site not in self.held_out_sites
        )

    def time_mask(self, half: int) -> np.ndarray:
        """Boolean epoch mask of time half ``0`` or ``1``."""
        if half not in (0, 1):
            raise ValueError(f"half must be 0 or 1, got {half!r}")
        epoch = np.arange(self.epochs)
        return epoch < self.midpoint if half == 0 else epoch >= self.midpoint

    def site_held_out(self) -> np.ndarray:
        """``(site_count,)`` boolean: whether each station is held out."""
        flag = np.zeros(self.site_count, dtype=bool)
        flag[list(self.held_out_sites)] = True
        return flag

    def cell_of(
        self, site_i: np.ndarray, site_j: np.ndarray, half: np.ndarray
    ) -> np.ndarray:
        """Cell index of each row: 0 train, 1 held-out SITES, 2 held-out TIME, 3 both.

        The returned codes line up with ``CELL_NAMES`` for 0 to 2, and 3 is the
        unscored cell of ``UNSCORED_CELL``.
        """
        held = self.site_held_out()
        left = np.asarray(site_i, dtype=int)
        right = np.asarray(site_j, dtype=int)
        when = np.asarray(half, dtype=int)
        if not (left.shape == right.shape == when.shape):
            raise ValueError(
                "site_i, site_j and half must have the same shape, got "
                f"{left.shape}, {right.shape}, {when.shape}"
            )
        site_out = held[left] | held[right]
        code = np.where(
            ~site_out & (when == 0),
            0,
            np.where(
                site_out & (when == 0), 1, np.where(~site_out & (when == 1), 2, 3)
            ),
        )
        return code.astype(int)


def make_split(
    epochs: int,
    site_count: int,
    held_out_sites: tuple[int, ...] = HELD_OUT_SITES,
) -> Split:
    """Build the equation (29) split. The midpoint is the record's own halfway point."""
    if epochs < 4:
        raise ValueError(f"epochs must be at least 4 to split in two, got {epochs!r}")
    if site_count < 4:
        raise ValueError(
            f"site_count must be at least 4 to hold stations out, got {site_count!r}"
        )
    held = tuple(int(site) for site in held_out_sites)
    if len(set(held)) != len(held):
        raise ValueError(f"held_out_sites must be distinct, got {held}")
    for site in held:
        if not 0 <= site < site_count:
            raise ValueError(
                f"held-out station {site} is outside [0, {site_count})"
            )
    if len(held) >= site_count - 1:
        raise ValueError(
            f"{len(held)} of {site_count} stations held out leaves fewer than 2 "
            "trained stations, so no training pair exists"
        )
    return Split(
        epochs=int(epochs),
        midpoint=int(epochs) // 2,
        held_out_sites=held,
        site_count=int(site_count),
    )
