"""Unit-quaternion algebra. Imports vecops."""

from sampleproj import vecops


def multiply(p, q):
    """Hamilton product of two quaternions in (w, x, y, z) order."""
    if len(p) != 4 or len(q) != 4:
        raise ValueError("multiply requires 4-element quaternions")
    pw, px, py, pz = p
    qw, qx, qy, qz = q
    return [
        pw * qw - px * qx - py * qy - pz * qz,
        pw * qx + px * qw + py * qz - pz * qy,
        pw * qy - px * qz + py * qw + pz * qx,
        pw * qz + px * qy - py * qx + pz * qw,
    ]


def conjugate(q):
    """Conjugate of a quaternion in (w, x, y, z) order."""
    if len(q) != 4:
        raise ValueError("conjugate requires a 4-element quaternion")
    return [q[0], -q[1], -q[2], -q[3]]


def unit(q):
    """Quaternion normalised to unit norm (dimensionless)."""
    return vecops.normalize(q)


def rotate(q, v):
    """Rotate a 3-vector v by unit quaternion q (same units as v)."""
    qn = unit(q)
    vq = [0.0, v[0], v[1], v[2]]
    out = multiply(multiply(qn, vq), conjugate(qn))
    return out[1:]
