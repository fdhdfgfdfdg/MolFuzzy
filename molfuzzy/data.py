"""Validated linguistic conversions and reliability-weighted aggregation."""
import numpy as np
from .mfv import MFV
from .validation import matrix_shape, labels as check_labels, mfv_stack, weights as check_weights, real_array, _bounded_unit_result

LINGUISTIC_SCALE = {
 'Negligible': {'number':1,'mfv':(.20,.70,.10)},
 'Low': {'number':2,'mfv':(.40,.50,.10)},
 'Moderate': {'number':3,'mfv':(.60,.30,.10)},
 'Significant': {'number':4,'mfv':(.80,.15,.05)},
 'High': {'number':5,'mfv':(.95,.05,0.)}}
_ABBREV = dict(zip('NLMSH', LINGUISTIC_SCALE))


def _term(term):
    if not isinstance(term,str):
        raise ValueError(f'Unknown linguistic term {term!r}')
    key = _ABBREV.get(term,term)
    if key not in LINGUISTIC_SCALE:
        raise ValueError(f'Unknown linguistic term {term!r}; expected N/L/M/S/H or full name')
    return LINGUISTIC_SCALE[key]


def term_to_mfv(term): return MFV(*_term(term)['mfv'])
def term_to_number(term): return _term(term)['number']


def assessment_to_mfv(number):
    if isinstance(number,bool) or not isinstance(number,(int,np.integer)) or not 1 <= number <= 5:
        raise ValueError('assessment number must be an integer from 1 to 5')
    return term_to_mfv(list(LINGUISTIC_SCALE)[number-1])


def linguistic_square_matrix_to_mfv(labels,matrix):
    n = len(check_labels(labels,'criterion_labels'))
    matrix_shape(matrix,n,n,'criteria matrix')
    out=[]
    for i,row in enumerate(matrix):
        converted=[]
        for j,cell in enumerate(row):
            if i==j:
                if cell not in (None,'','-'):
                    raise ValueError('criteria diagonal must be null, empty, or -')
                converted.append(MFV(0,0,1))
            else:
                converted.append(term_to_mfv(cell))
        out.append(converted)
    return out


def linguistic_matrix_to_numbers(labels,matrix):
    linguistic_square_matrix_to_mfv(labels,matrix)
    return np.array([[0 if i==j else term_to_number(c) for j,c in enumerate(row)] for i,row in enumerate(matrix)],dtype=float)


def normalized_relation_matrix(number_matrix):
    a=real_array(number_matrix, 'number_matrix')
    if a.ndim!=2 or a.shape[0]!=a.shape[1] or not a.size or not np.all(np.isfinite(a)) or np.any(a<0):
        raise ValueError('expected a finite nonnegative square matrix')
    if np.any(np.diag(a)!=0):
        raise ValueError('relation diagonal must be zero')
    maximum=a.max(axis=1,keepdims=True)
    scaled=a/np.where(maximum>0,maximum,1)
    sums=scaled.sum(axis=1,keepdims=True)
    return scaled/np.where(sums>0,sums,1)


def linguistic_rect_matrix_to_mfv(matrix):
    if not len(matrix) or not len(matrix[0]):
        raise ValueError('decision matrix must be nonempty')
    matrix_shape(matrix,len(matrix),len(matrix[0]),'decision matrix')
    return [[term_to_mfv(c) for c in row] for row in matrix]


def aggregate_rectangular_mfv(matrices,weights=None):
    a=mfv_stack(matrices)
    w=np.full(len(a),1/len(a)) if weights is None else check_weights(weights,len(a),'expert weights')
    out=np.sum(a*w[:,None,None,None],axis=0)
    out=_bounded_unit_result(out,len(a))
    return [[MFV(*v) for v in row] for row in out]


def linguistic_matrix_to_mfv_rows(labels,matrix):
    """Full aligned rows; diagonal is a zero relation, not a shifted coordinate."""
    from .mfv import WeightedTriple
    a=linguistic_square_matrix_to_mfv(labels,matrix)
    return [[WeightedTriple(0,0,0) if i==j else c for j,c in enumerate(row)] for i,row in enumerate(a)]
