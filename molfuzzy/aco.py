"""Optional angular traversal analysis. A sequence is NOT a preference ranking.

Minimizes the sum of normalized reciprocal-angle edge costs. Exact enumeration
is used for <=8 alternatives by default; larger problems use a stochastic
Ant-System-inspired search with normalized deposits and patience stopping.
"""
from dataclasses import dataclass,field
from itertools import permutations
import numpy as np
from .mfv import nr_matrix,geometry_scale,ANGLE_FLOOR,MIN_ANGLE_FLOOR
from .validation import mfv_array,weights as check_weights,labels as check_labels,scalar,integer

@dataclass
class AntIterationLog:
    paths: list
    costs: list
    pheromone: np.ndarray

@dataclass
class ACOResult:
    sequence: list
    sequence_labels: list
    path_cost: float
    fitness: float
    final_decision_matrix: np.ndarray
    heuristic_matrix: np.ndarray
    iterations: int
    stop_reason: str
    optimality_proven: bool
    best_cost_history: list=field(default_factory=list)
    log: list=field(default_factory=list)
    interpretation: str='angular traversal only; not best-to-worst preference'


def _path_cost(path,costs):
    return float(sum(costs[path[i],path[i+1]] for i in range(len(path)-1)))


def _path_fitness(path,final_decision):
    cost=_path_cost(path,final_decision)
    return 1/cost if cost>0 else 0.


def aco_rank(*args,**kwargs):
    raise ValueError('aco_rank was removed: use rank_alternatives for preferences or aco_sequence for traversal')


def validate_aco_options(alpha=1.,beta=2.,rho=.5,initial_pheromone=.1,
    max_iterations=200,min_iterations=20,patience=30,improvement_tolerance=1e-12,
    exact_threshold=8,keep_log=False,**unknown):
    """Validate and normalize options even when traversal is disabled."""
    if unknown: raise ValueError(f'unknown ACO options: {sorted(unknown)}')
    if not isinstance(keep_log,bool): raise ValueError('keep_log must be boolean')
    result=dict(alpha=scalar(alpha,'alpha',0,100),beta=scalar(beta,'beta',0,100),
        rho=scalar(rho,'rho',0,1,strict_low=True),
        initial_pheromone=scalar(initial_pheromone,'initial_pheromone',0,1,strict_low=True),
        max_iterations=integer(max_iterations,'max_iterations'),
        min_iterations=integer(min_iterations,'min_iterations'),patience=integer(patience,'patience'),
        improvement_tolerance=scalar(improvement_tolerance,'improvement_tolerance',0),
        exact_threshold=integer(exact_threshold,'exact_threshold',0),keep_log=keep_log)
    if result['exact_threshold']>8: raise ValueError('exact_threshold must be <=8 to bound factorial work')
    if result['min_iterations']>result['max_iterations']:
        raise ValueError('min_iterations must not exceed max_iterations')
    return result


def aco_sequence(decision_matrix,weights,labels,geometry='linear',alpha=1.,beta=2.,rho=.5,
    initial_pheromone=.1,max_iterations=200,seed=42,min_iterations=20,patience=30,
    improvement_tolerance=1e-12,exact_threshold=8,angle_floor=ANGLE_FLOOR,keep_log=False):
    if not isinstance(keep_log, bool):
        raise ValueError('keep_log must be boolean')
    a=mfv_array(decision_matrix,'decision_matrix')
    names=check_labels(labels,'alternative_labels')
    if len(a)!=len(names):
        raise ValueError('one label is required per alternative')
    w=check_weights(weights,a.shape[1],'criterion weights')
    geometry_scale(geometry)
    scalar(angle_floor,'angle_floor',MIN_ANGLE_FLOOR)
    opts=validate_aco_options(alpha,beta,rho,initial_pheromone,max_iterations,
        min_iterations,patience,improvement_tolerance,exact_threshold,keep_log)
    alpha=opts['alpha']; beta=opts['beta']; rho=opts['rho']; initial=opts['initial_pheromone']
    cap=opts['max_iterations']; minimum=opts['min_iterations']; patience=opts['patience']
    threshold=opts['exact_threshold']; tol=opts['improvement_tolerance']
    seed=integer(seed,'seed',0)
    # Canonical label ordering makes seeded search insensitive to input row order.
    canonical=sorted(range(len(names)),key=lambda i:names[i])
    rows=[[v.scale(x) for v,x in zip(decision_matrix[i],w)] for i in canonical]
    costs=nr_matrix(rows,geometry,angle_floor)
    n=len(names)
    heuristic=np.divide(1.,costs,out=np.zeros_like(costs),where=costs>0)
    logs=[]; history=[]; iterations=0
    if n==1:
        best=[0]; best_cost=0.; reason='single_alternative'; proven=True
    elif n<=threshold:
        best=None; best_cost=float('inf')
        for p in permutations(range(n)):
            c=_path_cost(p,costs)
            if c<best_cost:
                best=list(p); best_cost=c
        reason='exact_enumeration'; proven=True
        history=[best_cost]
    else:
        rng=np.random.default_rng(seed)
        pheromone=np.full((n,n),initial)
        best=list(range(n)); best_cost=_path_cost(best,costs)
        stalled=0; reason='max_iterations'; proven=False
        for iteration in range(1,cap+1):
            paths=[]
            for start in range(n):
                path=[start]; unvisited=set(range(n))-{start}
                while unvisited:
                    cand=sorted(unvisited)
                    logscore=alpha*np.log(np.maximum(pheromone[path[-1],cand],1e-300))+beta*np.log(heuristic[path[-1],cand])
                    probabilities=np.exp(logscore-logscore.max())
                    probabilities/=probabilities.sum()
                    chosen=int(rng.choice(cand,p=probabilities))
                    path.append(chosen); unvisited.remove(chosen)
                paths.append(path)
            path_costs=[_path_cost(p,costs) for p in paths]
            idx=min(range(n),key=lambda i:(path_costs[i],tuple(paths[i])))
            old=best_cost
            if path_costs[idx]<best_cost:
                best=paths[idx].copy(); best_cost=path_costs[idx]
            stalled=0 if old-best_cost>tol else stalled+1
            # Normalize inverse-cost deposits by the best ant fitness to bound trails.
            deposit=np.zeros_like(pheromone)
            smallest=min(path_costs)
            for path,c in zip(paths,path_costs):
                for u,v in zip(path,path[1:]):
                    deposit[u,v]+=smallest/c
            pheromone=(1-rho)*pheromone+deposit
            history.append(best_cost); iterations=iteration
            if keep_log:
                logs.append(AntIterationLog([[canonical[i] for i in p] for p in paths],path_costs,pheromone[np.ix_(np.argsort(canonical),np.argsort(canonical))].copy()))
            if iteration>=minimum and stalled>=patience:
                reason='no_improvement_patience'; break
    inverse=np.argsort(canonical)
    seq=[canonical[i] for i in best]
    return ACOResult(seq,[names[i] for i in seq],best_cost,1/best_cost if best_cost else 0.,
        costs[np.ix_(inverse,inverse)],heuristic[np.ix_(inverse,inverse)],iterations,reason,proven,history,logs)
