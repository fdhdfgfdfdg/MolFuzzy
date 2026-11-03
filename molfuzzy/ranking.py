"""Explicit additive utility ranking; costs reverse the membership score.

u=(m-n+1)/2 for benefits; 1-u for costs. With fixed nonnegative criterion
weights, increasing benefit utility cannot reduce the alternative score.
This model assumes an interpretable cardinal linguistic mapping and additive
compensation. It does not claim to recover an objective ground truth.
"""
from dataclasses import dataclass
import numpy as np
from .validation import mfv_array,weights as check_weights,labels as check_labels,scalar,real_array,_bounded_unit_result

@dataclass
class RankingResult:
    ranking: list
    ranking_labels: list
    scores: np.ndarray
    utilities: np.ndarray
    rank_groups: list
    method: str='additive_mfv_utility'


def rank_alternatives(decision_matrix,weights,labels,criterion_types=None,tie_tolerance=1e-10):
    a=mfv_array(decision_matrix,'decision_matrix')
    names=check_labels(labels,'alternative_labels')
    if len(names)!=len(a):
        raise ValueError('one alternative label is required for each row')
    w=check_weights(weights,a.shape[1],'criterion weights')
    tol=scalar(tie_tolerance,'tie_tolerance',0,1)
    types=np.ones(a.shape[1]) if criterion_types is None else real_array(criterion_types, 'criterion_types')
    if types.shape!=(a.shape[1],) or not np.all(np.isin(types,[-1,1])) or types.dtype==bool:
        raise ValueError('criterion_types must contain +1 (benefit) or -1 (cost) per criterion')
    u=(a[:,:,0]-a[:,:,1]+1)/2
    u=np.where(types[None,:]==1,u,1-u)
    scores=_bounded_unit_result(u@w,len(w))
    exact=sorted(range(len(names)),key=lambda i:(-scores[i],names[i]))
    groups=[]
    for i in exact:
        if not groups or scores[groups[-1][0]]-scores[i]>tol:
            groups.append([i])
        else:
            groups[-1].append(i)
    # Stable lexical presentation inside explicit tie groups; no preference implied.
    groups=[sorted(g,key=lambda i:names[i]) for g in groups]
    order=[i for g in groups for i in g]
    return RankingResult(order,[names[i] for i in order],scores,u,[[names[i] for i in g] for g in groups])
