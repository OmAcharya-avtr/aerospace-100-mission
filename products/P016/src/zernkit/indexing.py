"""Zernike single-index conventions and conversions between them.

Two single-index orderings of the doubly-indexed Zernike set ``Z_n^m`` are in
common use, and confusing them is the single most frequent source of silent
error in wavefront software. Both are implemented here explicitly.

Noll (1976) ordering
--------------------
Source: R. J. Noll, "Zernike polynomials and atmospheric turbulence",
*Journal of the Optical Society of America* **66** (3), 207-211 (1976).

* The index ``j`` starts at **1** (``j = 1`` is piston).
* Modes are grouped by radial degree ``n``; order ``n`` occupies
  ``n(n+1)/2 + 1 <= j <= (n+1)(n+2)/2``, i.e. ``n + 1`` modes.
* Within an order, ``|m|`` increases; each ``|m| > 0`` pair occupies two
  consecutive ``j``.
* **Sign rule:** even ``j`` carries the ``cos(m*theta)`` (``m > 0``) member,
  odd ``j`` carries the ``sin(|m|*theta)`` (``m < 0``) member.

Reproducing Noll's own list (his Table I) exactly::

    j :  1     2      3      4     5      6     7      8     9      10    11
    n :  0     1      1      2     2      2     3      3     3       3     4
    m :  0    +1     -1      0    -2     +2    -1     +1    -3      +3     0

OSA/ANSI ordering
-----------------
Source: ANSI Z80.28 "Methods for Reporting Optical Aberrations of Eyes";
equivalently L. N. Thibos, R. A. Applegate, J. T. Schwiegerling and
R. Webb, "Standards for reporting the optical aberrations of eyes",
*Journal of Refractive Surgery* **18**, S652-S660 (2002).

* The index ``j`` starts at **0** (``j = 0`` is piston).
* Closed form both ways: ``j = (n(n + 2) + m) / 2``.
* Within an order, ``m`` runs from ``-n`` to ``+n`` in steps of 2 -- a
  different order from Noll's, which is exactly why the two disagree from
  ``j = 4`` (Noll) / ``j = 3`` (OSA) onward.

Sign convention shared by both
------------------------------
``m > 0`` selects ``cos(m*theta)``, ``m < 0`` selects ``sin(|m|*theta)``,
``m = 0`` selects the rotationally symmetric mode. ``theta`` is measured
counter-clockwise from the ``+x`` axis. Note that neither standard fixes the
handedness of your pupil coordinates for you: if your optical system flips the
pupil, the sign of every ``m < 0`` coefficient flips with it.

All functions here are pure integer arithmetic; no floating point is used, so
there is no rounding failure at large ``j``.
"""

from __future__ import annotations

from math import isqrt

__all__ = [
    "validate_nm",
    "noll_to_nm",
    "nm_to_noll",
    "osa_to_nm",
    "nm_to_osa",
    "noll_to_osa",
    "osa_to_noll",
    "radial_order_from_noll",
    "mode_name",
]

#: Traditional aberration names for the low-order modes, keyed by ``(n, m)``.
#: Sources: Born & Wolf, *Principles of Optics*, 7th ed., Sec. 9.2; ANSI Z80.28.
_MODE_NAMES: dict[tuple[int, int], str] = {
    (0, 0): "piston",
    (1, -1): "vertical tilt (y)",
    (1, 1): "horizontal tilt (x)",
    (2, -2): "oblique astigmatism",
    (2, 0): "defocus",
    (2, 2): "vertical astigmatism",
    (3, -3): "oblique trefoil",
    (3, -1): "vertical coma",
    (3, 1): "horizontal coma",
    (3, 3): "vertical trefoil",
    (4, -4): "oblique quadrafoil",
    (4, -2): "oblique secondary astigmatism",
    (4, 0): "primary spherical",
    (4, 2): "vertical secondary astigmatism",
    (4, 4): "vertical quadrafoil",
}


def validate_nm(n: int, m: int) -> None:
    """Raise if ``(n, m)`` is not a legal Zernike index pair.

    Legality conditions (Born & Wolf, *Principles of Optics*, 7th ed.,
    Sec. 9.2): ``n >= 0``, ``|m| <= n``, and ``n - |m|`` even.

    Parameters
    ----------
    n : int
        Radial degree (dimensionless).
    m : int
        Azimuthal frequency (dimensionless, signed).

    Raises
    ------
    TypeError
        If ``n`` or ``m`` is not an integer.
    ValueError
        If the pair violates any legality condition.
    """
    bad_type = isinstance(n, bool) or isinstance(m, bool)
    if bad_type or not isinstance(n, int) or not isinstance(m, int):
        raise TypeError(f"n and m must be integers, got n={n!r} ({type(n).__name__}), m={m!r}")
    if n < 0:
        raise ValueError(f"radial degree n must be >= 0, got n={n}")
    if abs(m) > n:
        raise ValueError(f"azimuthal frequency must satisfy |m| <= n, got n={n}, m={m}")
    if (n - abs(m)) % 2 != 0:
        raise ValueError(
            f"n - |m| must be even (R_n^m vanishes identically otherwise), got n={n}, m={m}"
        )


def radial_order_from_noll(j: int) -> int:
    """Radial degree ``n`` of Noll index ``j`` (``j >= 1``), by exact integer math.

    ``n`` is the unique integer with ``n(n+1)/2 < j <= (n+1)(n+2)/2``, obtained
    from ``n = (isqrt(8(j-1) + 1) - 1) // 2``.
    """
    if isinstance(j, bool) or not isinstance(j, int):
        raise TypeError(f"Noll index j must be an integer, got {j!r}")
    if j < 1:
        raise ValueError(f"Noll indexing starts at j = 1 (piston); got j = {j}")
    return (isqrt(8 * (j - 1) + 1) - 1) // 2


def noll_to_nm(j: int) -> tuple[int, int]:
    """Convert a Noll index ``j >= 1`` to ``(n, m)``.

    Implements the ordering of Noll (1976), JOSA 66(3), 207-211: order-``n``
    modes occupy ``j = n(n+1)/2 + 1 ... (n+1)(n+2)/2``; within an order ``|m|``
    increases and even ``j`` takes the cosine (``m > 0``) member.

    Parameters
    ----------
    j : int
        Noll index, 1-based.

    Returns
    -------
    tuple[int, int]
        ``(n, m)`` with ``m > 0`` meaning ``cos(m*theta)`` and ``m < 0``
        meaning ``sin(|m|*theta)``.

    Examples
    --------
    >>> [noll_to_nm(j) for j in (1, 2, 3, 4, 5, 6, 11)]
    [(0, 0), (1, 1), (1, -1), (2, 0), (2, -2), (2, 2), (4, 0)]
    """
    n = radial_order_from_noll(j)
    p = j - n * (n + 1) // 2  # position within the order, 1 .. n+1
    k = n % 2
    m = ((p + k) // 2) * 2 - k
    if m != 0 and j % 2 == 1:
        m = -m
    return n, m


def nm_to_noll(n: int, m: int) -> int:
    """Convert ``(n, m)`` to the Noll index ``j >= 1``.

    Exact inverse of :func:`noll_to_nm`. Within order ``n`` the two members of
    an ``|m| > 0`` pair sit at ``j = n(n+1)/2 + |m|`` and one higher; the
    cosine member (``m > 0``) is the one with even ``j``.

    Parameters
    ----------
    n : int
        Radial degree.
    m : int
        Signed azimuthal frequency.

    Returns
    -------
    int
        Noll index (1-based).
    """
    validate_nm(n, m)
    base = n * (n + 1) // 2
    if m == 0:
        return base + 1
    j = base + abs(m)
    wants_even = m > 0
    if (j % 2 == 0) != wants_even:
        j += 1
    return j


def osa_to_nm(j: int) -> tuple[int, int]:
    """Convert an OSA/ANSI index ``j >= 0`` to ``(n, m)``.

    Uses the ANSI Z80.28 / Thibos et al. (2002) ordering, in which order ``n``
    starts at ``j = n(n+1)/2`` and ``m`` runs ``-n, -n+2, ..., +n``.

    Examples
    --------
    >>> [osa_to_nm(j) for j in (0, 1, 2, 3, 4, 5)]
    [(0, 0), (1, -1), (1, 1), (2, -2), (2, 0), (2, 2)]
    """
    if isinstance(j, bool) or not isinstance(j, int):
        raise TypeError(f"OSA/ANSI index j must be an integer, got {j!r}")
    if j < 0:
        raise ValueError(f"OSA/ANSI indexing starts at j = 0 (piston); got j = {j}")
    n = (isqrt(8 * j + 1) - 1) // 2
    m = 2 * j - n * (n + 2)
    return n, m


def nm_to_osa(n: int, m: int) -> int:
    """Convert ``(n, m)`` to the OSA/ANSI index ``j = (n(n + 2) + m) / 2``, 0-based."""
    validate_nm(n, m)
    return (n * (n + 2) + m) // 2


def noll_to_osa(j: int) -> int:
    """Convert a Noll index (1-based) to the corresponding OSA/ANSI index (0-based).

    The two orderings agree only for ``j_noll = 1, 2, 3`` (mapping to
    ``j_osa = 0, 2, 1`` -- already a swap at tip/tilt) and diverge thereafter.
    """
    return nm_to_osa(*noll_to_nm(j))


def osa_to_noll(j: int) -> int:
    """Convert an OSA/ANSI index (0-based) to the corresponding Noll index (1-based)."""
    return nm_to_noll(*osa_to_nm(j))


def mode_name(n: int, m: int) -> str:
    """Traditional aberration name for ``(n, m)``, or a generic ``Z(n, m)`` label.

    Names for ``n <= 4`` follow Born & Wolf, *Principles of Optics*, 7th ed.,
    Sec. 9.2 and ANSI Z80.28. Higher orders return ``"Z(n, m)"``.
    """
    validate_nm(n, m)
    return _MODE_NAMES.get((n, m), f"Z({n}, {m})")
