"""Regenerate numerical evidence, environment record and non-portable timings.
Run from the source root after installation: python docs/reproduce.py.
"""
from pathlib import Path
from dataclasses import asdict
import json
import platform
import sys
import time
from collections import Counter
import numpy as np
from molfuzzy import DecisionPipeline,aco_sequence
from molfuzzy.pipeline import jsonable
from molfuzzy.sensitivity import run_sensitivity_analysis,perturbation_analysis,parameter_sensitivity_analysis

ROOT=Path(__file__).resolve().parents[1]

def main():
    s=json.loads((ROOT/'examples/lean_energy_scenario.json').read_text())
    def run(**kw):
        return DecisionPipeline(s['criterion_labels'],s['alternative_labels'],**kw).run(s['criteria_matrices'],s['decision_matrices'])
    r=run(include_sequence=True)
    sensitivity=run_sensitivity_analysis(**s,include_sequence=True)
    robust=perturbation_analysis(**s,trials=100,seed=42)
    cw=np.asarray(r.configuration['criterion_weights_used']); ew=r.expert_weights['weights']
    # Local 20% mixtures toward each simplex vertex, followed by vertices.
    criterion_grid=[(.8*cw+.2*v).tolist() for v in np.eye(len(cw))]+np.eye(len(cw)).tolist()
    expert_grid=[(.8*ew+.2*v).tolist() for v in np.eye(len(ew))]+np.eye(len(ew)).tolist()
    parameter_rows=parameter_sensitivity_analysis(**s,criterion_weight_grid=criterion_grid,
        expert_weight_grid=expert_grid,angle_floors=(1e-12,1e-9,1e-6,1e-3,.1,1.))
    paths=[]
    for seed in range(100):
        seq=aco_sequence(r.aggregated_decision,r.criteria.weights,s['alternative_labels'],seed=seed,exact_threshold=0)
        paths.append({'seed':seed,'sequence':seq.sequence_labels,'path_cost':seq.path_cost,'iterations':seq.iterations,'stop_reason':seq.stop_reason})
    times=[]
    for _ in range(5): run()
    for _ in range(30):
        start=time.perf_counter(); run(); times.append((time.perf_counter()-start)*1000)
    sequence_times=[]
    for _ in range(30):
        start=time.perf_counter(); run(include_sequence=True); sequence_times.append((time.perf_counter()-start)*1000)
    record={'environment':{'python':sys.version,'numpy':np.__version__,'platform':platform.platform(),'processor':platform.processor()},
        'case':r.to_dict(),'sensitivity':[asdict(x) for x in sensitivity],'robustness':robust,
        'parameter_sensitivity':parameter_rows,
        'heuristic_100_seeds':paths,
        'heuristic_summary':{'optimal_cost_hits':int(sum(np.isclose(x['path_cost'],r.sequence.path_cost,rtol=0,atol=1e-12) for x in paths)),
            'distinct_sequences':len(set(tuple(x['sequence']) for x in paths)),
            'first_node_counts':dict(Counter(x['sequence'][0] for x in paths)),
            'cost_min':min(x['path_cost'] for x in paths),'cost_max':max(x['path_cost'] for x in paths)},
        'timing':{'warmups':5,'repetitions':30,'core_ms':times,'with_exact_traversal_ms':sequence_times,
            'core_median_ms':float(np.median(times)),'core_p95_ms':float(np.percentile(times,95)),
            'exact_median_ms':float(np.median(sequence_times)),
            'scope':'Current process on this host; not CPU-isolated; excludes import and CLI startup.'}}
    out=ROOT/'validation';out.mkdir(exist_ok=True)
    (out/'results.json').write_text(json.dumps(jsonable(record),indent=2,allow_nan=False)+'\n')
    (out/'environment.txt').write_text(json.dumps(record['environment'],indent=2)+'\n')
    print(r.summary()); print(json.dumps({'robustness':robust,'heuristic':record['heuristic_summary'],'timing_median':record['timing']['core_median_ms']},indent=2))

if __name__=='__main__': main()
