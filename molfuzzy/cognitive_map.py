"""Similarity-network criterion weights; no causal interpretation is implied."""
from dataclasses import dataclass
import numpy as np
from .mfv import WeightedTriple,nr_matrix,geometry_scale,ANGLE_FLOOR,MIN_ANGLE_FLOOR
from .data import aggregate_rectangular_mfv
from .validation import mfv_array,mfv_stack,scalar,integer

@dataclass
class CognitiveMapResult:
    aggregated: list
    nr_matrix: np.ndarray
    state_vectors: list
    weights: np.ndarray
    iterations: int
    converged: bool
    stop_reason: str
    residual: float


def aggregate_matrices(matrices,weights=None):
    mfv_stack(matrices,square=True)
    return aggregate_rectangular_mfv(matrices,weights)


def cognitive_map_weights(aggregated,geometry='linear',tolerance=1e-8,max_iterations=500,angle_floor=ANGLE_FLOOR):
    a=mfv_array(aggregated,square=True)
    geometry_scale(geometry)
    scalar(angle_floor,'angle_floor',MIN_ANGLE_FLOOR)
    tol=scalar(tolerance,'tolerance',0,1,strict_low=True)
    cap=integer(max_iterations,'max_iterations')
    n=len(a)
    if n==1:
        return CognitiveMapResult(aggregated,np.zeros((1,1)),[np.ones(1)],np.ones(1),0,True,'single_criterion',0.)
    rows=[[WeightedTriple(0,0,0) if i==j else v for j,v in enumerate(row)] for i,row in enumerate(aggregated)]
    nr=nr_matrix(rows,geometry,angle_floor)
    state=np.ones(n)
    history=[state.copy()]
    converged=False
    for _ in range(cap):
        state=1/(1+np.exp(-(nr@state)))
        history.append(state.copy())
        residual=float(np.max(abs(1/(1+np.exp(-(nr@state)))-state)))
        if residual<=tol:
            converged=True
            break
    return CognitiveMapResult(aggregated,nr,history,state/state.sum(),len(history)-1,
        converged,'fixed_point_residual' if converged else 'max_iterations',residual)
