"""MFV triples and angular similarity.

The five historical geometry scales are reference normalizers, not physical
maximum bond angles. They cancel under reciprocal column normalization.
They are retained for compatibility and diagnostics, not as model alternatives.
"""
from dataclasses import dataclass
from math import pi
import numpy as np
from .validation import scalar, real_array, _bounded_unit_result

_GEOMETRY_DENOMINATOR = dict(linear=pi, trigonal_planar=2*pi/3,
    tetrahedral=pi/2, trigonal_bipyramidal=2*pi/5, octahedral=pi/3)
GEOMETRIES = tuple(_GEOMETRY_DENOMINATOR)
MIN_ANGLE_FLOOR = float(np.sqrt(np.finfo(float).tiny))
ANGLE_FLOOR = 1e-6  # radians; explicit regularization for duplicates/near duplicates


def register_geometry(name, max_angle_radians):
    """Register a positive reference scale (does not change normalized NR)."""
    if not isinstance(name, str) or not name.strip():
        raise ValueError('geometry name must be a nonempty string')
    if name in _GEOMETRY_DENOMINATOR:
        raise ValueError('geometry names cannot be overwritten')
    _GEOMETRY_DENOMINATOR[name] = scalar(max_angle_radians, 'geometry scale', np.finfo(float).tiny)


def geometry_scale(name):
    if not isinstance(name, str) or name not in _GEOMETRY_DENOMINATOR:
        raise ValueError(f'Unknown geometry {name!r}')
    return _GEOMETRY_DENOMINATOR[name]


@dataclass(frozen=True)
class MFV:
    m: float
    n: float
    h: float
    tol: float = 1e-8

    def __post_init__(self):
        tol = scalar(self.tol, 'tol', 0, 1e-6, strict_low=True)
        v = np.array([scalar(x, name, 0, 1) for name, x in zip(('m','n','h'), self.as_tuple())])
        if abs(v.sum()-1) > tol:
            raise ValueError('MFV components must sum to one')
        # Canonical exact normalization removes accepted rounding drift.
        for name, x in zip(('m','n','h'), v / v.sum()):
            object.__setattr__(self, name, float(x))

    def as_tuple(self):
        return self.m, self.n, self.h

    def scale(self, weight):
        w = scalar(weight, 'weight', 0)
        return WeightedTriple(*(w*x for x in self.as_tuple()))

    @staticmethod
    def from_angle(alpha, max_alpha):
        d = scalar(max_alpha, 'max_alpha', 0, strict_low=True)
        a = scalar(alpha, 'alpha', 0, d)
        return MFV(1-a/d, a/d, 0)

    @staticmethod
    def average(values, weights=None):
        from .validation import weights as check_weights
        values = list(values)
        if not values or any(not isinstance(v, MFV) for v in values):
            raise ValueError('average requires a nonempty sequence of MFVs')
        w = np.full(len(values), 1/len(values)) if weights is None else check_weights(weights, len(values))
        v = np.sum(np.array([v.as_tuple() for v in values])*w[:,None], axis=0)
        return MFV(*_bounded_unit_result(v,len(values)))


@dataclass(frozen=True)
class WeightedTriple:
    m: float
    n: float
    h: float

    def __post_init__(self):
        for name in ('m','n','h'):
            object.__setattr__(self, name, scalar(getattr(self,name), name, 0))

    def as_tuple(self):
        return self.m, self.n, self.h


def _vector_to_array(vec):
    values = list(vec)
    if not values or any(not isinstance(v,(MFV,WeightedTriple)) for v in values):
        raise ValueError('fuzzy vector must be nonempty and contain MFV or WeightedTriple')
    return np.array([v.as_tuple() for v in values]).ravel()


def angle_between(vec_a, vec_b):
    """Stable angle using normalized-vector sum and difference; zero norm is undefined."""
    a, b = _vector_to_array(vec_a), _vector_to_array(vec_b)
    if a.shape != b.shape:
        raise ValueError('fuzzy vectors must have equal dimensions')
    if a.max() == 0 or b.max() == 0:
        raise ValueError('angle of a zero vector is undefined')
    # Scale first to avoid overflow/underflow in the norm.
    a, b = a/a.max(), b/b.max()
    a, b = a/np.linalg.norm(a), b/np.linalg.norm(b)
    return float(2*np.arctan2(np.linalg.norm(a-b), np.linalg.norm(a+b)))


def normalize_angle(alpha, geometry='linear'):
    # This API accepts an actual angle in radians, not an arbitrary magnitude.
    value = scalar(alpha, 'alpha', 0, pi) / geometry_scale(geometry)
    if not np.isfinite(value):
        raise ValueError('normalized angle is outside the finite float range')
    return value


def reciprocal_matrix(angle_matrix, epsilon=ANGLE_FLOOR):
    """Continuous bounded inverse 1/max(angle,epsilon); self edges are excluded."""
    eps = scalar(epsilon, 'epsilon', np.finfo(float).tiny)
    a = real_array(angle_matrix, 'angle_matrix')
    if a.ndim != 2 or a.shape[0] != a.shape[1] or a.size == 0 or not np.all(np.isfinite(a)) or np.any(a<0):
        raise ValueError('angle_matrix must be a nonempty finite nonnegative square matrix')
    out = 1/np.maximum(a, eps)
    np.fill_diagonal(out, 0)
    return out


def column_normalize(matrix):
    a = real_array(matrix, 'matrix')
    if a.ndim != 2 or a.size == 0 or not np.all(np.isfinite(a)) or np.any(a<0):
        raise ValueError('matrix must be finite, nonnegative and nonempty')
    # Scaling by column maximum avoids overflow before column summation.
    mx = a.max(axis=0)
    scaled = a / np.where(mx>0, mx, 1)
    sums = scaled.sum(axis=0)
    return scaled / np.where(sums>0, sums, 1)


def build_angle_matrix(rows, geometry='linear'):
    return _raw_angle_matrix(rows) / geometry_scale(geometry)


def _raw_angle_matrix(rows):
    rows = list(rows)
    if not rows:
        raise ValueError('rows must be nonempty')
    arrays = [_vector_to_array(r) for r in rows]
    if any(a.shape != arrays[0].shape or not np.any(a>0) for a in arrays):
        raise ValueError('rows must have equal dimensions and nonzero norms')
    a = np.zeros((len(rows),len(rows)))
    for i in range(len(rows)):
        for j in range(i):
            a[i,j] = a[j,i] = angle_between(rows[i],rows[j])
    return a


def nr_matrix(rows, geometry='linear', angle_floor=ANGLE_FLOOR):
    """Geometry-invariant NR; regularization occurs in raw radians.

    Calculating the equivalent canonical form avoids rounding differences among
    historical geometry scales. Call build_angle_matrix for scaled diagnostics.
    """
    geometry_scale(geometry)
    eps = scalar(angle_floor, 'angle_floor', MIN_ANGLE_FLOOR)
    raw = _raw_angle_matrix(rows)
    # Column-specific scaling preserves reciprocal ratios without overflow,
    # including subnormal floors and duplicate vectors.
    denom = np.maximum(raw, eps)
    np.fill_diagonal(denom, np.inf)
    if len(denom) == 1:
        return np.zeros((1, 1))
    similarity = denom.min(axis=0) / denom
    return column_normalize(similarity)
