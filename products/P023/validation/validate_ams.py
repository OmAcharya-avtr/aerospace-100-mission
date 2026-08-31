"""Validation 2: the attainable moment set against its closed forms.

Three independent references:

* the **cube**: for three orthogonal unit effectors with bounds [-1, 1] the
  AMS is exactly the cube [-1, 1]^3 -- 8 vertices, volume 8, area 24, and the
  vertex coordinates themselves;
* the **zonotope volume formula**
  ``V = sum_{i<j<k} |det(b_i, b_j, b_k)| L_i L_j L_k``
  (Ziegler 1995, Lectures on Polytopes, Lecture 7), computed independently of
  the convex hull;
* the **zonotope vertex count** ``g(g-1) + 2`` for ``g`` distinct generator
  lines in general position.

Plus the internal cross-check: Durham's pairwise facet construction against
brute-force enumeration of all 2^m box vertices.

Run: ``python validation/validate_ams.py``
"""

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from alloclab.ams import (  # noqa: E402
    attainable_moment_set,
    expected_vertex_count,
    zonotope_volume,
)
from alloclab.dataset import reference_thruster_cluster  # noqa: E402
from alloclab.effectors import (  # noqa: E402
    general_effector_set,
    orthogonal_effectors,
    pyramid_reaction_wheels,
)

CONFIGS = {
    "orthogonal triad, bound 1.0": orthogonal_effectors(1.0),
    "orthogonal triad, bound 2.0": orthogonal_effectors(2.0),
    "pyramid wheels, 4 x 0.1 N*m": pyramid_reaction_wheels(0.1),
    "pyramid wheels, 5 x 0.05 N*m, 35 deg": pyramid_reaction_wheels(
        0.05, half_angle_deg=35.0, n_wheels=5
    ),
    "thruster cluster, 8 x 1 N, arm 0.5 m": reference_thruster_cluster(1.0, 0.5),
    "thruster cluster, t1 failed off": reference_thruster_cluster(1.0, 0.5).with_failures([0]),
    "thruster cluster, t1 stuck at 1 N": reference_thruster_cluster(1.0, 0.5).with_failures(
        [0], stuck_at=1.0
    ),
}


def main() -> None:
    print("=" * 78)
    print("VALIDATION 2a: the cube known answer")
    print("=" * 78)
    e = orthogonal_effectors(1.0)
    ams = attainable_moment_set(e)
    corners = np.array(
        [[sx, sy, sz] for sx in (-1.0, 1.0) for sy in (-1.0, 1.0) for sz in (-1.0, 1.0)]
    )
    got = np.sort(np.round(ams.vertices, 15), axis=0)
    want = np.sort(corners, axis=0)
    print(f"  vertices computed / expected      : {ams.n_vertices} / 8")
    print(f"  max |vertex coordinate error|     : {np.max(np.abs(got - want)):.6e}")
    print(f"  volume computed / expected        : {ams.volume:.15g} / 8")
    print(f"  |volume error|                    : {abs(ams.volume - 8.0):.6e}")
    print(f"  area computed / expected          : {ams.area:.15g} / 24")
    print(f"  |area error|                      : {abs(ams.area - 24.0):.6e}")
    print(f"  boundary scale along +x           : {ams.boundary_scale([1, 0, 0]):.15g} (exp 1)")
    diag = ams.boundary_scale([1.0, 1.0, 1.0])
    print(f"  boundary scale along (1,1,1)      : {diag:.15g} (exp {np.sqrt(3.0):.15g})")
    print(f"  |diagonal scale error|            : {abs(diag - np.sqrt(3.0)):.6e}")

    print("\n" + "=" * 78)
    print("VALIDATION 2b: hull volume against the closed-form zonotope volume")
    print("=" * 78)
    print(
        f"{'configuration':<40}{'hull volume':>18}{'closed form':>18}{'rel err':>12}"
    )
    for label, eset in CONFIGS.items():
        a = attainable_moment_set(eset)
        closed = zonotope_volume(eset)
        rel = abs(a.volume - closed) / closed if closed > 0 else abs(a.volume - closed)
        print(f"{label:<40}{a.volume:>18.12g}{closed:>18.12g}{rel:>12.2e}")

    print("\n" + "=" * 78)
    print("VALIDATION 2c: vertex count against g(g-1)+2 for g distinct generator lines")
    print("=" * 78)
    print(f"{'configuration':<40}{'computed':>12}{'formula':>12}{'match':>10}")
    for label, eset in CONFIGS.items():
        a = attainable_moment_set(eset)
        exp = expected_vertex_count(eset)
        print(f"{label:<40}{a.n_vertices:>12d}{exp:>12d}{str(a.n_vertices == exp):>10}")
    print("\nThe formula holds only in general position: no two generator lines")
    print("parallel (parallel columns are merged before counting) and no three")
    print("coplanar. Every configuration above is in general position after the")
    print("merge, so all seven match. A configuration with three coplanar")
    print("generator lines would have fewer vertices than the formula predicts,")
    print("and expected_vertex_count is then an upper bound only.")

    print("\n" + "=" * 78)
    print("VALIDATION 2d: pairwise construction against brute-force box enumeration")
    print("=" * 78)
    print(
        f"{'configuration':<40}{'v pair':>8}{'v brute':>9}{'vol rel err':>14}"
        f"{'t pair [ms]':>13}{'t brute [ms]':>14}"
    )
    for label, eset in CONFIGS.items():
        t0 = time.perf_counter()
        a = attainable_moment_set(eset, method="pairwise")
        t_a = (time.perf_counter() - t0) * 1e3
        t0 = time.perf_counter()
        b = attainable_moment_set(eset, method="bruteforce")
        t_b = (time.perf_counter() - t0) * 1e3
        rel = abs(a.volume - b.volume) / b.volume if b.volume > 0 else 0.0
        print(
            f"{label:<40}{a.n_vertices:>8d}{b.n_vertices:>9d}{rel:>14.2e}"
            f"{t_a:>13.3f}{t_b:>14.3f}"
        )

    print("\n" + "=" * 78)
    print("VALIDATION 2e: a second closed form, the 2x3x5 box from scaled axes")
    print("=" * 78)
    # B = I with asymmetric bounds: the AMS is the axis-aligned box
    # [-1,1] x [-1.5,1.5] x [-2.5,2.5], volume 2*3*5 = 30, area 2(2*3+2*5+3*5)=62.
    e = general_effector_set(
        np.eye(3), np.array([-1.0, -1.5, -2.5]), np.array([1.0, 1.5, 2.5])
    )
    ams = attainable_moment_set(e)
    print(f"  vertices                          : {ams.n_vertices} (expected 8)")
    print(f"  volume                            : {ams.volume:.15g} (expected 30)")
    print(f"  |volume error|                    : {abs(ams.volume - 30.0):.6e}")
    print(f"  area                              : {ams.area:.15g} (expected 62)")
    print(f"  |area error|                      : {abs(ams.area - 62.0):.6e}")

    print("\n" + "=" * 78)
    print("VALIDATION 2f: failure shrinks the AMS -- measured volume ratios")
    print("=" * 78)
    nominal = reference_thruster_cluster(1.0, 0.5)
    v0 = attainable_moment_set(nominal).volume
    print(f"  nominal AMS volume                : {v0:.12g} (N*m)^3")
    print(f"{'failed effectors':<26}{'volume':>16}{'ratio':>10}{'rank':>7}")
    for failed in ([], [0], [0, 1], [0, 2], [6], [6, 7], [0, 1, 2, 3], [4, 5]):
        degraded = nominal.with_failures(failed) if failed else nominal
        a = attainable_moment_set(degraded)
        print(
            f"{str(failed):<26}{a.volume:>16.10g}{a.volume / v0:>10.4f}{degraded.rank:>7d}"
        )


if __name__ == "__main__":
    main()
