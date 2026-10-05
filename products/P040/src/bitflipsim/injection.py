"""Bit-upset sampling and injection into numpy parameter and activation blocks.

A *bit site* is an ``(element_index, bit_position)`` pair inside one contiguous
array. The population of sites for an array of ``n`` elements of width ``w``
bits is ``n * w``, and the upset model of :mod:`bitflipsim.flux` treats every
site as equally likely, because the per-bit cross-section in equation (1) of
that module is a single number with no positional dependence. Any belief that
some bits are more *likely* to be hit belongs in the cross-section, not here;
this module only decides *where*, uniformly, and :mod:`bitflipsim.criticality`
decides what the consequence is.

Sampling without replacement is the default: two particles striking the same
bit in the same exposure would cancel, and a campaign that silently cancels its
own injections reports a degradation that is too small. The
``allow_repeat=True`` path exists so that the cancellation can be measured
rather than assumed, and the test suite measures it.

Units: all indices and counts dimensionless.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .bitlayout import bits_of_array


@dataclass(frozen=True)
class BitUpset:
    """One single-bit upset at a definite site.

    Attributes
    ----------
    element_index:
        Flat index into the parameter or activation array.
    bit_position:
        Storage-bit index, 0 = LSB of the integer view.
    """

    element_index: int
    bit_position: int


def sample_upsets(
    n_elements: int,
    bits_per_element: int,
    count: int,
    rng: np.random.Generator,
    allow_repeat: bool = False,
) -> list[BitUpset]:
    """Draw ``count`` upset sites uniformly from ``n_elements * bits_per_element``.

    Parameters
    ----------
    n_elements:
        Number of array elements exposed.
    bits_per_element:
        Storage width of one element in bits.
    count:
        Number of upsets to place.
    rng:
        Seeded ``numpy.random.Generator``; the only source of randomness.
    allow_repeat:
        ``False`` (default) draws distinct sites, so ``count`` flips are
        ``count`` net flips. ``True`` draws with replacement, which lets a site
        be hit twice and cancel.

    Returns
    -------
    list[BitUpset]
        Length ``count``, in draw order.
    """
    if n_elements <= 0:
        raise ValueError(f"n_elements must be > 0, got {n_elements}")
    if bits_per_element <= 0:
        raise ValueError(f"bits_per_element must be > 0, got {bits_per_element}")
    if count < 0:
        raise ValueError(f"count must be >= 0, got {count}")
    population = int(n_elements) * int(bits_per_element)
    if not allow_repeat and count > population:
        raise ValueError(
            f"cannot draw {count} distinct sites from a population of {population}; "
            "pass allow_repeat=True to sample with replacement"
        )
    if count == 0:
        return []
    if allow_repeat:
        flat = rng.integers(0, population, size=int(count))
    else:
        flat = rng.choice(population, size=int(count), replace=False)
    return [
        BitUpset(element_index=int(i) // int(bits_per_element),
                 bit_position=int(i) % int(bits_per_element))
        for i in flat
    ]


def apply_upsets(array: np.ndarray, upsets: list[BitUpset]) -> np.ndarray:
    """Return a copy of ``array`` with every listed bit site flipped.

    The flip is performed on the unsigned-integer view, so it is exact for
    IEEE 754 floats and for two's-complement integers alike, with no rounding
    and no dtype promotion. Repeated sites flip twice and therefore cancel,
    which is why :func:`sample_upsets` draws distinct sites by default.

    Parameters
    ----------
    array:
        Any contiguous numeric array of item width 8, 16, 32 or 64 bits.
    upsets:
        Sites to flip. ``element_index`` indexes the flattened array.

    Returns
    -------
    numpy.ndarray
        New array, same shape and dtype as ``array``.
    """
    arr = np.ascontiguousarray(array)
    width = arr.dtype.itemsize * 8
    out = arr.copy()
    if not upsets:
        return out
    flat = out.reshape(-1)
    view = bits_of_array(flat)
    n = flat.size
    for upset in upsets:
        if not 0 <= upset.element_index < n:
            raise ValueError(
                f"element_index {upset.element_index} outside [0, {n}) for this array"
            )
        if not 0 <= upset.bit_position < width:
            raise ValueError(
                f"bit_position {upset.bit_position} outside [0, {width}) "
                f"for dtype {arr.dtype.name}"
            )
        mask = view.dtype.type(1) << view.dtype.type(upset.bit_position)
        view[upset.element_index] ^= mask
    return out


def upset_site_histogram(
    upsets: list[BitUpset], bits_per_element: int
) -> np.ndarray:
    """Count upsets per bit position, shape ``(bits_per_element,)``, dtype int64.

    Used to check that the sampler is uniform over bit positions.
    """
    if bits_per_element <= 0:
        raise ValueError(f"bits_per_element must be > 0, got {bits_per_element}")
    hist = np.zeros(int(bits_per_element), dtype=np.int64)
    for upset in upsets:
        hist[upset.bit_position] += 1
    return hist


def distinct_site_count(upsets: list[BitUpset]) -> int:
    """Number of distinct sites in ``upsets``, dimensionless."""
    return len({(u.element_index, u.bit_position) for u in upsets})


def net_flipped_bits(original: np.ndarray, faulty: np.ndarray) -> int:
    """Population count of ``original XOR faulty``, dimensionless.

    An independent check on :func:`apply_upsets`: with distinct sites it must
    equal ``len(upsets)``, and with repeats it must equal the number of sites
    hit an odd number of times.
    """
    a = np.ascontiguousarray(original)
    b = np.ascontiguousarray(faulty)
    if a.shape != b.shape or a.dtype != b.dtype:
        raise ValueError("arrays must share shape and dtype")
    xor = bits_of_array(a.reshape(-1)) ^ bits_of_array(b.reshape(-1))
    return int(np.unpackbits(xor.view(np.uint8)).sum())
