"""Analytic expert reliability normalization, replacing the redundant v1 LP.

Default reliability measures agreement with the panel, not expertise or truth.
Independent reliability dimensions can be supplied by the caller.
"""
from dataclasses import dataclass
import warnings
import numpy as np
from .validation import weights as check_weights, real_array, _bounded_unit_result

@dataclass
class ReliabilityDimensions:
    consistency: np.ndarray
    stability: np.ndarray
    suitability: np.ndarray


def consensus_reliability(number_matrices):
    a=real_array(number_matrices, 'number_matrices')
    if a.ndim!=3 or min(a.shape)<1 or a.shape[1]!=a.shape[2] or not np.all(np.isfinite(a)) or np.any(a<0) or np.any(a>1):
        raise ValueError('expected a nonempty stack of square matrices with values in [0,1]')
    if np.any(np.diagonal(a,axis1=1,axis2=2)!=0):
        raise ValueError('reliability relation matrices must have zero diagonal')
    mask=~np.eye(a.shape[1],dtype=bool)
    if not mask.any():
        return np.ones(a.shape[0])
    distance=np.mean(np.abs(a[:,mask]-a.mean(axis=0)[mask]),axis=1)
    return 1-distance


def normalize_reliability(scores):
    r=real_array(scores, 'reliability scores')
    if r.ndim!=1 or len(r)==0 or not np.all(np.isfinite(r)) or np.any(r<0) or np.any(r>1):
        raise ValueError('reliability scores must be a nonempty vector in [0,1]')
    if r.max()<=0:
        raise ValueError('total reliability must be positive')
    scaled=r/r.max()
    w=scaled/scaled.sum()
    return {'R':r.copy(),'R_bar':w.copy(),'weights':w,'objective':0.0,
            'benchmark':int(np.argmax(w)),
            'benchmark_indices':np.flatnonzero(np.isclose(w,w.max(),atol=1e-12,rtol=0)).tolist(),
            'solver':'analytic normalization'}


def lom_expert_weights(dims,beta=(1/3,1/3,1/3)):
    b=check_weights(beta,3,'beta')
    vectors=[real_array(x, 'reliability dimensions') for x in (dims.consistency,dims.stability,dims.suitability)]
    if any(x.ndim!=1 or x.size==0 or not np.all(np.isfinite(x)) or np.any(x<0) or np.any(x>1) for x in vectors) or any(x.shape!=vectors[0].shape for x in vectors):
        raise ValueError('reliability dimensions must be equal-length nonempty vectors in [0,1]')
    a=np.stack(vectors)
    active=(b[:,None]>0)&(a>0)
    if not np.any(active):
        raise ValueError('total reliability must be positive')
    # Products are formed as mantissas and exponents so even beta*dimension
    # below binary64's smallest subnormal can contribute to relative weights.
    bm,be=np.frexp(b[:,None]); am,ae=np.frexp(a)
    powers=be+ae
    if int(powers[active].min()) < -1020:
        return _exact_small_reliability(a,b)
    pivot=int(powers[active].max())
    with np.errstate(under='ignore'):
        terms=np.ldexp(bm*am,np.where(active,powers-pivot,0))
    totals=terms.sum(axis=0)
    peak=float(totals.max())
    scaled=totals/peak
    result=normalize_reliability(scaled)
    # Raw values are display diagnostics; normalization never uses them.
    with np.errstate(under='ignore'):
        raw=_bounded_unit_result(np.ldexp(totals,pivot),len(b))
    mantissa,exponent=np.frexp(peak)
    result.update(R=raw,R_scaled=scaled,
        reliability_scale={'mantissa':float(mantissa),'exponent2':int(exponent)+pivot},
        raw_reliability_underflow=((raw==0)&np.any(active,axis=0)).tolist(),
        scaling_method='common binary exponent before weighted summation')
    return result


def _exact_small_reliability(a,b):
    """Rare underflow path: exact rational arithmetic on binary64 inputs."""
    from fractions import Fraction
    coefficients=[Fraction(float(x)) for x in b]
    totals=[sum((coefficients[k]*Fraction(float(a[k,i])) for k in range(3)),Fraction(0)) for i in range(a.shape[1])]
    peak=max(totals)
    scaled=np.array([float(x/peak) for x in totals])
    result=normalize_reliability(scaled)
    total=sum(totals)
    w=np.array([float(x/total) for x in totals])
    exponent=peak.numerator.bit_length()-peak.denominator.bit_length()
    factor=Fraction(2)**exponent
    if peak>=factor:exponent+=1;factor*=2
    raw=_bounded_unit_result(np.array([float(x) for x in totals]),len(b))
    result.update(R=raw,R_scaled=scaled,R_bar=w.copy(),weights=w,
        benchmark=int(np.argmax(w)),
        benchmark_indices=np.flatnonzero(np.isclose(w,w.max(),atol=1e-12,rtol=0)).tolist(),
        reliability_scale={'mantissa':float(peak/factor),'exponent2':exponent},
        raw_reliability_underflow=[bool(raw[i]==0 and totals[i]>0) for i in range(len(totals))],
        scaling_method='exact rational fallback for subnormal weighted products')
    return result


def compute_reliability_dimensions(number_matrices):
    """Deprecated bridge; all dimensions are explicitly the same agreement score.

    No temporal stability or internal judgement consistency is inferred from
    arbitrary matrix columns. Use consensus_reliability or supply real dimensions.
    """
    warnings.warn('Use consensus_reliability or independently supplied ReliabilityDimensions',DeprecationWarning,stacklevel=2)
    r=consensus_reliability(number_matrices)
    return ReliabilityDimensions(r.copy(),r.copy(),r.copy())
