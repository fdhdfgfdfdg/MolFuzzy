"""Explicit criterion, preference, and optional traversal sensitivity records."""
from dataclasses import dataclass,asdict,field
from collections import Counter
from copy import deepcopy
import numpy as np
from .mfv import GEOMETRIES
from .pipeline import DecisionPipeline,jsonable,scenario_hash
from .validation import integer,labels
from .lom import ReliabilityDimensions,lom_expert_weights

@dataclass
class SensitivityRow:
    learning_rate: float
    geometry: str
    seed: int
    weights: list
    criterion_order: list
    alternative_order: list
    alternative_rank_groups: list
    scores: list
    balance_converged: bool
    balance_iterations: int
    balance_residual: float
    sequence: list
    path_cost: object
    sequence_stop_reason: object
    optimality_proven: object
    diagnostics: dict=field(default_factory=dict)
    criterion_labels: list=field(default_factory=list)
    alternative_labels: list=field(default_factory=list)
    configuration: dict=field(default_factory=dict)


def _component_status(records):
    failures=sum(item['status']!='success' for item in records)
    if not failures: return 'success'
    return 'failed' if failures==len(records) and all(item['status']=='failed' for item in records) else 'partial_failure'


def diagnostics(result):
    """Separate valid preference scores from diagnostic convergence."""
    warnings=[]
    if not result.balancing.converged: warnings.append('consensus_relaxation_not_converged_analytic_target_used')
    if not result.criteria.converged: warnings.append('criterion_network_not_converged_user_weights_used')
    return {'scoring_valid':True,'criterion_weight_source':result.configuration['criterion_weight_source'],
        'consensus':{'converged':result.balancing.converged,'residual':result.balancing.residual,'stop_reason':result.balancing.stop_reason},
        'criterion_network':{'converged':result.criteria.converged,'residual':result.criteria.residual,'stop_reason':result.criteria.stop_reason},
        'warnings':warnings}


def run_sensitivity_analysis(criterion_labels,alternative_labels,criteria_matrices,decision_matrices,
    learning_rates=(.1,.5,1.),geometries=GEOMETRIES,seeds=(42,),include_sequence=False,on_error="raise",**options):
    _check_on_error(on_error)
    if not len(learning_rates) or not len(geometries) or not len(seeds):
        raise ValueError('sensitivity grids must be nonempty')
    criterion_labels=labels(criterion_labels, 'criterion_labels')
    alternative_labels=labels(alternative_labels, 'alternative_labels')
    rows=[]
    for lr in learning_rates:
        for g in geometries:
            for seed in seeds:
                try:
                    r=DecisionPipeline(criterion_labels,alternative_labels,geometry=g,learning_rate=lr,aco_seed=seed,include_sequence=include_sequence,**options).run(criteria_matrices,decision_matrices)
                except (ValueError,RuntimeError) as exc:
                    if on_error=='raise': raise
                    rows.append({'learning_rate':_failed_value(lr),'geometry':_failed_value(g),
                        'seed':_failed_value(seed),**_failure(exc)})
                    continue
                seq=r.sequence
                used=r.configuration['criterion_weights_used']
                rows.append(SensitivityRow(lr,g,seed,used,
                    [criterion_labels[i] for i in sorted(range(len(used)), key=lambda i:(-used[i],criterion_labels[i]))],
                    r.ranking.ranking_labels,r.ranking.rank_groups,r.ranking.scores.tolist(),
                    r.balancing.converged,r.balancing.iterations,r.balancing.residual,
                    seq.sequence_labels if seq else [],seq.path_cost if seq else None,
                    seq.stop_reason if seq else None,seq.optimality_proven if seq else None,
                    diagnostics(r),list(r.criterion_labels),list(r.alternative_labels),deepcopy(r.configuration)))
    if on_error=='raise': return rows
    return _scan_report([dict(asdict(r),status='success') if isinstance(r,SensitivityRow) else r for r in rows])


def rankings_are_stable(rows,target='alternatives'):
    if isinstance(rows,dict):
        if rows.get('failed',0) or rows.get('unexecuted',0):
            raise ValueError('stability is undefined for an incomplete scan')
        rows=[SensitivityRow(**{k:v for k,v in r.items() if k!='status'}) for r in rows['records']]
    if not rows:
        raise ValueError('stability requires at least one scenario')
    if target=='alternatives':
        values=[tuple(tuple(g) for g in r.alternative_rank_groups) for r in rows]
    elif target=='criteria':
        values=[tuple(r.criterion_order) for r in rows]
    elif target=='sequence':
        if any(not r.sequence for r in rows):
            raise ValueError('traversal was not evaluated in every scenario')
        values=[tuple(r.sequence) for r in rows]
    else:
        raise ValueError('target must be alternatives, criteria, or sequence')
    return len(set(values))==1


def perturbation_analysis(criterion_labels,alternative_labels,criteria_matrices,decision_matrices,
    trials=100,seed=42,include_perturbations=True,include_leave_one_out=True,**options):
    """Leave-one-expert-out and one-cell one-level decision perturbations.

    Top-group inclusion counts are reported; with ties they need not sum to trials.
    This is a local robustness diagnostic, not a confidence interval.
    """
    if not isinstance(include_perturbations,bool) or not isinstance(include_leave_one_out,bool):
        raise ValueError('analysis switches must be boolean')
    count=integer(trials,'trials'); seed=integer(seed,'seed',0)
    base=DecisionPipeline(criterion_labels,alternative_labels,**options).run(criteria_matrices,decision_matrices)
    e=len(criteria_matrices); omitted=[]
    for i in range(e) if e>1 and include_leave_one_out else []:
        adjusted=dict(options)
        if adjusted.get('reliability_dimensions') is not None:
            dims=adjusted['reliability_dimensions']
            adjusted['reliability_dimensions']=ReliabilityDimensions(*(np.delete(getattr(dims,key),i) for key in ('consistency','stability','suitability')))
            try:
                lom_expert_weights(adjusted['reliability_dimensions'],adjusted.get('beta',(1/3,1/3,1/3)))
            except ValueError as exc:
                omitted.append({'omitted_expert':i,'status':'skipped','reason':str(exc)});continue
        if adjusted.get('expert_weights') is not None:
            w=np.delete(adjusted['expert_weights'],i)
            if w.sum()==0:
                omitted.append({'omitted_expert':i,'status':'skipped','reason':'zero_remaining_weight'}); continue
            adjusted['expert_weights']=(w/w.sum()).tolist()
        try:
            r=DecisionPipeline(criterion_labels,alternative_labels,**adjusted).run(
                [x for j,x in enumerate(criteria_matrices) if j!=i],
                [x for j,x in enumerate(decision_matrices) if j!=i])
        except (ValueError,RuntimeError) as exc:
            omitted.append({'omitted_expert':i,'status':'failed','reason':str(exc)});continue
        omitted.append({'omitted_expert':i,'status':'success','rank_groups':r.ranking.rank_groups,'scores':r.ranking.scores.tolist(),
            'diagnostics':diagnostics(r),'configuration':deepcopy(r.configuration)})
    from .data import term_to_number
    rng=np.random.default_rng(seed); winners=Counter(); unchanged=0
    trial_records=[]
    for trial in range(count if include_perturbations else 0):
        dm=[[list(row) for row in matrix] for matrix in decision_matrices]
        ei=int(rng.integers(e)); ai=int(rng.integers(len(alternative_labels))); ci=int(rng.integers(len(criterion_labels)))
        original=dm[ei][ai][ci]
        old=term_to_number(original)-1
        available=[x for x in (old-1,old+1) if 0<=x<=4]
        dm[ei][ai][ci]='NLMSH'[int(rng.choice(available))]
        metadata={'trial':trial,'cell':[ei,ai,ci],'expert_index':ei,
            'expert_label':f'expert_{ei+1}','expert_label_scheme':'generated one-based positional label',
            'alternative_label':base.alternative_labels[ai],'criterion_label':base.criterion_labels[ci],
            'original_term':original,'original_level':old+1,'new_term':dm[ei][ai][ci],
            'input_sha256':scenario_hash(base.criterion_labels,base.alternative_labels,criteria_matrices,dm)}
        try:
            r=DecisionPipeline(criterion_labels,alternative_labels,**options).run(criteria_matrices,dm)
        except (ValueError,RuntimeError) as exc:
            trial_records.append({**metadata,'status':'failed','reason':str(exc),'exception_type':type(exc).__name__});continue
        winners.update(r.ranking.rank_groups[0]); unchanged+=r.ranking.rank_groups==base.ranking.rank_groups
        trial_records.append({**metadata,'status':'success',
            'new_term':dm[ei][ai][ci],'rank_groups':r.ranking.rank_groups,'scores':r.ranking.scores.tolist(),
            'diagnostics':diagnostics(r),'input_sha256':r.configuration['input_sha256']})
    return {'seed':seed,'trials':count,'perturbation':'one random expert/alternative/criterion linguistic cell moved one adjacent level',
        'base_rank_groups':base.ranking.rank_groups,'unchanged_full_rankings':unchanged,
        'top_group_inclusion_counts':dict(winners),'leave_one_expert_out':omitted,
        'criterion_labels':list(base.criterion_labels),'alternative_labels':list(base.alternative_labels),
        'diagnostics':diagnostics(base),'configuration':deepcopy(base.configuration),'trial_records':trial_records,
        'successful_trials':sum(x['status']=='success' for x in trial_records),
        'failed_trials':sum(x['status']=='failed' for x in trial_records),
        'perturbation_status':'skipped' if not include_perturbations else _component_status(trial_records),
        'leave_one_out_status':'skipped' if not include_leave_one_out or e==1 else _component_status(omitted),
        'perturbation_skip_reason':None if include_perturbations else 'disabled',
        'leave_one_out_skip_reason':'disabled' if not include_leave_one_out else ('requires_at_least_two_experts' if e==1 else None)}


def parameter_sensitivity_analysis(criterion_labels, alternative_labels, criteria_matrices,
    decision_matrices, criterion_weight_grid=(), expert_weight_grid=(),
    angle_floors=(), on_error="raise", **options):
    """One-factor scenarios with full scores and adjacent score gaps.

    Weight scenarios override the corresponding baseline weighting rule. They
    are stress tests, not draws from a calibrated uncertainty distribution.
    """
    _check_on_error(on_error)
    def run(opts):
        return DecisionPipeline(criterion_labels, alternative_labels, **opts).run(criteria_matrices, decision_matrices)
    base = run(options)
    def record(kind, value, result):
        scores = result.ranking.scores
        order = sorted(range(len(scores)),key=lambda i:(-scores[i],result.alternative_labels[i]))
        names=result.alternative_labels
        lookup={name:i for i,name in enumerate(names)}
        groups=result.ranking.rank_groups
        between=[{'higher_group':list(upper),'lower_group':list(lower),
            'gap':float(min(scores[lookup[x]] for x in upper)-max(scores[lookup[x]] for x in lower))}
            for upper,lower in zip(groups,groups[1:])]
        return {'parameter':kind, 'value':jsonable(value),'status':'success',
            'criterion_labels':list(result.criterion_labels),'alternative_labels':list(result.alternative_labels),
            'diagnostics':diagnostics(result),'configuration':deepcopy(result.configuration),
            'criterion_weights_used':result.configuration['criterion_weights_used'],
            'expert_weights_used':result.expert_weights['weights'].tolist(),
            'rank_groups':result.ranking.rank_groups, 'scores':scores.tolist(),
            'adjacent_score_gaps':[float(scores[a]-scores[b]) for a,b in zip(order,order[1:])],
            'score_gap_order':[names[i] for i in order],
            'score_gap_pairs':[[names[a],names[b]] for a,b in zip(order,order[1:])],
            'score_gap_definition':'descending raw scores; lexical order only for exact score equality',
            'between_rank_group_gaps':between,
            'same_rank_groups_as_baseline':result.ranking.rank_groups == base.ranking.rank_groups}
    records=[record('baseline', None, base)]
    for key, grid in [('criterion_weights',criterion_weight_grid),('expert_weights',expert_weight_grid),('angle_floor',angle_floors)]:
        for value in grid:
            opts=dict(options)
            if key == 'expert_weights':
                opts.pop('reliability_dimensions', None)
            opts[key]=value
            try:
                records.append(record(key, value, run(opts)))
            except (ValueError,RuntimeError) as exc:
                if on_error=='raise': raise
                records.append({'parameter':key,'value':_failed_value(value),**_failure(exc)})
    return records if on_error=="raise" else _scan_report(records)


def _check_on_error(value):
    if value not in ('raise','record'): raise ValueError('on_error must be raise or record')


def _failed_value(value):
    try: return jsonable(value)
    except (ValueError,TypeError):
        return {'representation':repr(value),'json_compatible':False}


def _failure(exc):
    return {'status':'failed','failure_kind':'invalid_parameters' if isinstance(exc,ValueError) else 'execution_failure',
        'exception_type':type(exc).__name__,'reason':str(exc)}


def _scan_report(records):
    successes=[r for r in records if r['status']=='success']
    groups=[r.get('rank_groups',r.get('alternative_rank_groups')) for r in successes]
    failed=len(records)-len(successes)
    return jsonable({'records':records,'requested':len(records),'successful':len(successes),
        'failed':failed,'unexecuted':0,'status':_component_status(records),
        'all_scenarios_stable':None if failed or not groups else all(g==groups[0] for g in groups),
        'successful_scenarios_stable':None if not groups else all(g==groups[0] for g in groups)})
