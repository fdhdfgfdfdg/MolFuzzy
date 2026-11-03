"""Shared strict validation; public APIs reject malformed inputs before calculation."""
import numpy as np


def scalar(value, name, low=None, high=None, strict_low=False):
    if isinstance(value, (bool, np.bool_, complex, np.complexfloating)):
        raise ValueError(f'{name} must be a finite number')
    try:
        x = float(value)
    except (ValueError, TypeError, OverflowError) as exc:
        raise ValueError(f'{name} must be a finite number') from exc
    if not np.isfinite(x):
        raise ValueError(f'{name} must be finite')
    if low is not None and (x <= low if strict_low else x < low):
        raise ValueError(f'{name} must be {">" if strict_low else ">="} {low}')
    if high is not None and x > high:
        raise ValueError(f'{name} must be <= {high}')
    return x


def integer(value, name, minimum=1):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < minimum:
        raise ValueError(f'{name} must be an integer >= {minimum}')
    return int(value)


def labels(values, name):
    if isinstance(values, str):
        raise ValueError(f'{name} must be a sequence of unique nonempty strings')
    out = list(values)
    if not out or any(not isinstance(v, str) or not v.strip() for v in out) or len(set(out)) != len(out):
        raise ValueError(f'{name} must contain unique nonempty strings')
    return out


def real_array(values, name='values'):
    """Reject complex, boolean and textual values before any lossy conversion."""
    try:
        original = np.asarray(values, dtype=object)
        from numbers import Real
        if any(isinstance(x, (bool, np.bool_)) or not isinstance(x, Real) for x in original.flat):
            raise ValueError(f'{name} must contain real numbers, not booleans or complex values')
        return np.array(values, dtype=float, copy=True)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f'{name} must contain real numeric values') from exc


def weights(values, n, name='weights'):
    try:
        a = real_array(values, name)
    except (ValueError, TypeError) as exc:
        raise ValueError(f'{name} must be a numeric vector') from exc
    if a.shape != (n,) or not np.all(np.isfinite(a)) or np.any(a < 0):
        raise ValueError(f'{name} must have {n} finite nonnegative entries')
    if not np.isclose(a.sum(), 1, rtol=0, atol=1e-8):
        raise ValueError(f'{name} must sum to one')
    return a / a.sum()


def matrix_shape(matrix, rows, cols, name):
    try:
        good = len(matrix) == rows and all(len(row) == cols for row in matrix)
    except TypeError:
        good = False
    if not good:
        raise ValueError(f'{name} must have shape ({rows}, {cols})')


def mfv_array(matrix, name='matrix', square=False):
    from .mfv import MFV
    try:
        n = len(matrix)
        m = len(matrix[0]) if n else 0
    except TypeError as exc:
        raise ValueError(f'{name} must be a nonempty MFV matrix') from exc
    if not n or not m:
        raise ValueError(f'{name} must be nonempty')
    matrix_shape(matrix, n, n if square else m, name)
    if any(not isinstance(v, MFV) for row in matrix for v in row):
        raise ValueError(f'{name} entries must be MFV objects')
    return np.array([[v.as_tuple() for v in row] for row in matrix], dtype=float)


def mfv_stack(matrices, square=False):
    if len(matrices) == 0:
        raise ValueError('at least one expert matrix is required')
    arrays = [mfv_array(m, f'matrices[{i}]', square) for i, m in enumerate(matrices)]
    if any(a.shape != arrays[0].shape for a in arrays):
        raise ValueError('all expert matrices must have identical shapes')
    return np.stack(arrays)


def _bounded_unit_result(values, operations=1):
    """Repair only roundoff in internally computed convex results.

    External input validation must not use this helper. The allowance is
    8*max(1, operations)*binary64 epsilon, capped at 1e-8.
    """
    a=np.asarray(values,dtype=float)
    tolerance=min(1e-8,8*max(1,operations)*np.finfo(float).eps)
    if not np.all(np.isfinite(a)) or np.any(a < -tolerance) or np.any(a > 1+tolerance):
        raise ValueError('internal convex result exceeds the roundoff allowance')
    return np.clip(a,0.,1.)
