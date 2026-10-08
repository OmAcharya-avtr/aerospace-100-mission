"""2-D polytope plotting, Agg backend only.

Importing this module sets the Matplotlib backend to Agg, so nothing here ever
opens a window.  `show()` is never called; every function draws onto an Axes
the caller owns and the caller saves the figure.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import numpy as np  # noqa: E402  (must follow the backend selection)

from .polytope import EmptyPolytopeError, Polytope  # noqa: E402

__all__ = ["polygon_vertices", "plot_polytope"]


def polygon_vertices(P: Polytope, tol: float = 1e-9) -> np.ndarray:
    """Vertices of a bounded 2-D polytope in counter-clockwise order, (v, 2).

    Raises
    ------
    ValueError
        If `P.dim != 2`.
    EmptyPolytopeError
        If `P` is empty.
    """
    if P.dim != 2:
        raise ValueError(f"polygon_vertices needs a 2-D polytope, got dim {P.dim}")
    V = P.vertices(tol=tol)
    if V.shape[0] == 0:
        raise EmptyPolytopeError("no vertices to order")
    centre = V.mean(axis=0)
    order = np.argsort(np.arctan2(V[:, 1] - centre[1], V[:, 0] - centre[0]))
    return V[order]


def plot_polytope(ax, P: Polytope, *, closed: bool = True, **kwargs):
    """Draw the boundary of a bounded 2-D polytope on `ax`.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
    P : Polytope
        Bounded, two-dimensional, non-empty.
    closed : bool
        Repeat the first vertex so the outline closes.
    **kwargs
        Passed to `ax.plot`, or to `ax.fill` when `fill=True` is given.

    Returns
    -------
    list
        Whatever Matplotlib returned, so the caller can build a legend.
    """
    V = polygon_vertices(P)
    if closed:
        V = np.vstack([V, V[:1]])
    if kwargs.pop("fill", False):
        return ax.fill(V[:, 0], V[:, 1], **kwargs)
    return ax.plot(V[:, 0], V[:, 1], **kwargs)
