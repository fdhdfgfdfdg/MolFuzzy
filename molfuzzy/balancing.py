"""Convex relaxation to a reliability-weighted consensus (not Q-learning)."""
from dataclasses import dataclass,field
import numpy as np
from .mfv import MFV
from .validation import mfv_stack,weights as check_weights,scalar,integer,mfv_array,_bounded_unit_result

@dataclass
class BalanceResult:
    balanced: dict
    target: list
    iterations: int
    converged: bool
    stop_reason: str
    residual: float
    history: list=field(default_factory=list)
    residual_history: list=field(default_factory=list)
    per_expert_residual: list=field(default_factory=list)


def _matrix(a,operations=1):
    bounded=_bounded_unit_result(a,operations)
    return [[MFV(*v) for v in row] for row in bounded]


def balance_experts(matrices,weights,learning_rate=.1,tolerance=1e-6,max_iterations=1000):
    a=mfv_stack(matrices,square=True)
    w=check_weights(weights,len(a),'expert weights')
    rate=scalar(learning_rate,'learning_rate',0,1,strict_low=True)
    tol=scalar(tolerance,'tolerance',0,1,strict_low=True)
    cap=integer(max_iterations,'max_iterations')
    target_matrix=_matrix(np.sum(a*w[:,None,None,None],axis=0),len(a))
    target=mfv_array(target_matrix)
    balanced={i:list(map(list,m)) for i,m in enumerate(matrices)}
    current=a.copy()
    residuals=[float(np.max(abs(current-target)))]
    history=[]
    for _ in range(cap):
        if residuals[-1]<=tol:
            break
        updated=(1-rate)*current+rate*target
        # Apply the same bounded correction and normalization as MFV, in bulk.
        last_raw=_bounded_unit_result(updated,2)
        totals=last_raw.sum(axis=-1,keepdims=True)
        if np.any(abs(totals-1)>1e-8):
            raise ValueError("internal MFV components must sum to one")
        updated=last_raw/totals
        history.append(float(np.max(abs(updated-current))))
        current=updated
        residuals.append(float(np.max(abs(current-target))))
    if history:
        # Construct from the pre-normalized final update to avoid a second
        # normalization; returned triples exactly match the residual arrays.
        balanced={i:_matrix(x,2) for i,x in enumerate(last_raw)}
    converged=residuals[-1]<=tol
    return BalanceResult(balanced,target_matrix,
        len(history),converged,'residual_tolerance' if converged else 'max_iterations',
        residuals[-1],history,residuals,np.max(abs(current-target),axis=(1,2,3)).tolist())
