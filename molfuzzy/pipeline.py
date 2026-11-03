"""Validated decision pipeline with separate optional traversal diagnostics."""
from dataclasses import dataclass,asdict
from copy import deepcopy
import hashlib
import json
import platform
import numpy as np
from .data import linguistic_square_matrix_to_mfv,linguistic_matrix_to_numbers,linguistic_rect_matrix_to_mfv,aggregate_rectangular_mfv
from .lom import consensus_reliability,normalize_reliability,lom_expert_weights
from .balancing import balance_experts
from .cognitive_map import cognitive_map_weights
from .ranking import rank_alternatives
from .aco import aco_sequence,validate_aco_options
from .mfv import geometry_scale,ANGLE_FLOOR,MIN_ANGLE_FLOOR
from .validation import labels,weights,matrix_shape,scalar,integer


def jsonable(value):
    """Convert recursively to strict JSON types; reject nonfinite/overflow values.

    Extended precision floats are rounded to Python binary64 for JSON output.
    """
    if isinstance(value,np.ndarray): return jsonable(value.tolist())
    if isinstance(value,(np.bool_,bool)): return bool(value)
    if isinstance(value,(np.integer,int)): return int(value)
    if isinstance(value,(np.floating,float)):
        result=float(value)
        if not np.isfinite(result):
            raise ValueError('JSON numbers must be finite and representable as binary64')
        return result
    if isinstance(value,dict): return {str(k):jsonable(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)): return [jsonable(v) for v in value]
    if value is None or isinstance(value,str): return value
    raise TypeError(f'unsupported JSON value: {type(value).__name__}')


def scenario_hash(criterion_labels,alternative_labels,criteria_matrices,decision_matrices):
    value=jsonable(dict(criterion_labels=criterion_labels,alternative_labels=alternative_labels,
        criteria_matrices=criteria_matrices,decision_matrices=decision_matrices))
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),
        ensure_ascii=False,allow_nan=False).encode('utf-8')).hexdigest()

@dataclass
class PipelineResult:
    expert_weights: dict
    balancing: object
    criteria: object
    ranking: object
    aggregated_decision: list
    sequence: object
    criterion_labels: list
    alternative_labels: list
    configuration: dict

    def summary(self):
        lines=['Criterion weights:']
        lines += [f'  {n}: {w:.6f}' for n,w in zip(self.criterion_labels,self.configuration['criterion_weights_used'])]
        lines += ['Preference ranking (equal signs denote ties): '+ ' > '.join(' = '.join(g) for g in self.ranking.rank_groups)]
        lines += ['Scores: '+', '.join(f'{self.alternative_labels[i]}={self.ranking.scores[i]:.6f}' for i in self.ranking.ranking)]
        lines += [f'Consensus: {self.balancing.stop_reason}; residual={self.balancing.residual:.3g}',f'Criterion network: {self.criteria.stop_reason}; residual={self.criteria.residual:.3g}']
        if self.sequence is not None:
            lines += ['Angular traversal (not a preference ranking): '+' -> '.join(self.sequence.sequence_labels),f'Path cost: {self.sequence.path_cost:.6f}; {self.sequence.stop_reason}']
        return '\n'.join(lines)

    def to_dict(self):
        return jsonable(asdict(self))


class DecisionPipeline:
    def __init__(self,criterion_labels,alternative_labels,geometry='linear',learning_rate=.1,
        criterion_types=None,expert_weights=None,reliability_dimensions=None,beta=(1/3,1/3,1/3),
        criterion_weights=None,balance_tolerance=1e-6,balance_max_iterations=1000,
        cognitive_tolerance=1e-8,cognitive_max_iterations=500,angle_floor=ANGLE_FLOOR,
        tie_tolerance=1e-10,include_sequence=False,aco_seed=42,aco_options=None):
        self.criterion_labels=labels(criterion_labels,'criterion_labels')
        self.alternative_labels=labels(alternative_labels,'alternative_labels')
        geometry_scale(geometry)
        self.geometry=geometry
        self.learning_rate=scalar(learning_rate,'learning_rate',0,1,strict_low=True)
        self.balance_tolerance=scalar(balance_tolerance,'balance_tolerance',0,1,strict_low=True)
        self.balance_max_iterations=integer(balance_max_iterations,'balance_max_iterations')
        self.cognitive_tolerance=scalar(cognitive_tolerance,'cognitive_tolerance',0,1,strict_low=True)
        self.cognitive_max_iterations=integer(cognitive_max_iterations,'cognitive_max_iterations')
        self.angle_floor=scalar(angle_floor,'angle_floor',MIN_ANGLE_FLOOR)
        self.tie_tolerance=scalar(tie_tolerance,'tie_tolerance',0,1)
        self.criterion_types=deepcopy(criterion_types)
        self.supplied_expert_weights=deepcopy(expert_weights)
        self.reliability_dimensions=deepcopy(reliability_dimensions)
        if expert_weights is not None and reliability_dimensions is not None:
            raise ValueError('supply expert_weights or reliability_dimensions, not both')
        self.beta=weights(beta,3,'beta')
        self.supplied_criterion_weights=None if criterion_weights is None else weights(criterion_weights,len(self.criterion_labels),'criterion weights')
        self.include_sequence=include_sequence
        if not isinstance(include_sequence,bool):
            raise ValueError('include_sequence must be boolean')
        self.aco_seed=integer(aco_seed,'aco_seed',0)
        supplied_options={} if aco_options is None else deepcopy(dict(aco_options))
        self.aco_options=validate_aco_options(**supplied_options)

    def run(self,criteria_matrices,decision_matrices):
        if not len(criteria_matrices) or len(criteria_matrices)!=len(decision_matrices):
            raise ValueError('criteria and decision matrices must contain the same positive number of experts')
        n=len(self.criterion_labels); k=len(self.alternative_labels); e=len(criteria_matrices)
        cm=[linguistic_square_matrix_to_mfv(self.criterion_labels,m) for m in criteria_matrices]
        for i,m in enumerate(decision_matrices):
            matrix_shape(m,k,n,f'decision_matrices[{i}]')
        dm=[linguistic_rect_matrix_to_mfv(m) for m in decision_matrices]
        if self.supplied_expert_weights is not None:
            w=weights(self.supplied_expert_weights,e,'expert weights')
            expert=normalize_reliability(w); expert['source']='user supplied weights'
        elif self.reliability_dimensions is not None:
            expert=lom_expert_weights(self.reliability_dimensions,self.beta)
            weights(expert['weights'],e,'expert weights')
            expert['source']='user supplied reliability dimensions'
        else:
            numeric=np.array([linguistic_matrix_to_numbers(self.criterion_labels,m) for m in criteria_matrices])/5
            expert=normalize_reliability(consensus_reliability(numeric))
            expert['source']='panel agreement heuristic; not validated expertise'
        balancing=balance_experts(cm,expert['weights'],self.learning_rate,self.balance_tolerance,self.balance_max_iterations)
        # Consensus is available analytically. Relaxation is a diagnostic path;
        # it is not allowed to introduce stopping-tolerance bias into the result.
        cog=cognitive_map_weights(balancing.target,self.geometry,self.cognitive_tolerance,self.cognitive_max_iterations,self.angle_floor)
        if not cog.converged and self.supplied_criterion_weights is None:
            raise RuntimeError('criterion network did not converge; inspect parameters or increase cognitive_max_iterations')
        chosen=cog.weights if self.supplied_criterion_weights is None else self.supplied_criterion_weights
        aggregated=aggregate_rectangular_mfv(dm,expert['weights'])
        ranked=rank_alternatives(aggregated,chosen,self.alternative_labels,self.criterion_types,self.tie_tolerance)
        sequence=aco_sequence(aggregated,chosen,self.alternative_labels,self.geometry,seed=self.aco_seed,angle_floor=self.angle_floor,**self.aco_options) if self.include_sequence else None
        input_hash=scenario_hash(self.criterion_labels,self.alternative_labels,criteria_matrices,decision_matrices)
        config={'version':'2.0.10','geometry':self.geometry,'geometry_effect':'reference scale cancels in NR',
            'learning_rate':self.learning_rate,'criterion_weights_used':chosen.tolist(),
            'criterion_weight_source':'network' if self.supplied_criterion_weights is None else 'user',
            'criterion_types':list(self.criterion_types) if self.criterion_types is not None else [1]*n,
            'angle_floor_radians':self.angle_floor,'tie_tolerance':self.tie_tolerance,
            'balance_tolerance':self.balance_tolerance,'balance_max_iterations':self.balance_max_iterations,
            'cognitive_tolerance':self.cognitive_tolerance,'cognitive_max_iterations':self.cognitive_max_iterations,
            'aco_seed':self.aco_seed,'aco_options':deepcopy(self.aco_options),'include_sequence':self.include_sequence,
            'sequence_execution_status':'completed' if sequence is not None else 'disabled',
            'beta':self.beta.tolist(),
            'expert_weight_source':expert['source'],
            'expert_weights_used':expert['weights'].tolist(),
            'supplied_expert_weights':jsonable(self.supplied_expert_weights),
            'reliability_dimensions':None if self.reliability_dimensions is None else jsonable(asdict(self.reliability_dimensions)),
            'input_sha256':input_hash,'input_hash_scheme':'ordered scenario fields; sorted-key compact UTF-8 JSON; original linguistic spellings',
            'environment':{'python':platform.python_version(),'numpy':np.__version__,'platform':platform.platform()}}
        return PipelineResult(expert,balancing,cog,ranked,aggregated,sequence,list(self.criterion_labels),list(self.alternative_labels),deepcopy(config))
