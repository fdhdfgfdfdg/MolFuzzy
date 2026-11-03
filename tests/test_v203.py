import json
from pathlib import Path
import numpy as np
import pytest
from molfuzzy import DecisionPipeline,MFV,aco_sequence
from molfuzzy.pipeline import jsonable,scenario_hash
from molfuzzy.lom import ReliabilityDimensions
from molfuzzy.sensitivity import parameter_sensitivity_analysis,run_sensitivity_analysis,rankings_are_stable,perturbation_analysis

@pytest.fixture
def s(): return json.loads((Path(__file__).parents[1]/'examples/lean_energy_scenario.json').read_text())

def run(s,**opts): return DecisionPipeline(s['criterion_labels'],s['alternative_labels'],**opts).run(s['criteria_matrices'],s['decision_matrices'])

@pytest.mark.parametrize('mode',['weights','dimensions','nested'])
def test_longdouble_json(s,mode):
    w=np.array([.2,.3,.5],dtype=np.longdouble)
    if mode=='weights': result=run(s,expert_weights=w).to_dict()
    elif mode=='dimensions': result=run(s,reliability_dimensions=ReliabilityDimensions(w,w,w)).to_dict()
    else: result=jsonable({'nested':(np.array([[np.longdouble('.2')]]),np.array(np.longdouble('.3')))})
    assert json.loads(json.dumps(result,allow_nan=False))==result

@pytest.mark.parametrize('value',[float('nan'),float('inf'),np.longdouble('nan'),np.longdouble("1e400")])
def test_unrepresentable_json_rejected(value):
    with pytest.raises(ValueError): jsonable({'nested':np.array([value])})

def test_failed_trial_reproducible(s,monkeypatch):
    original=DecisionPipeline.run
    calls=0
    def fail(self,cm,dm):
        nonlocal calls
        calls+=1
        if calls==2: raise RuntimeError('simulated numerical failure')
        return original(self,cm,dm)
    monkeypatch.setattr(DecisionPipeline,'run',fail)
    report=perturbation_analysis(**s,trials=2,include_leave_one_out=False)
    row=report['trial_records'][0];ei,ai,ci=row['cell']
    assert row['status']=='failed' and row['exception_type']=='RuntimeError'
    assert row['original_term']==s['decision_matrices'][ei][ai][ci]
    assert row['alternative_label']==s['alternative_labels'][ai]
    assert row['criterion_label']==s['criterion_labels'][ci]
    assert row['expert_label']==f'expert_{ei+1}'
    dm=json.loads(json.dumps(s['decision_matrices']));dm[ei][ai][ci]=row['new_term']
    assert row['input_sha256']==scenario_hash(s['criterion_labels'],s['alternative_labels'],s['criteria_matrices'],dm)
    assert report['successful_trials']==1 and report['failed_trials']==1
    json.dumps(report,allow_nan=False)

@pytest.mark.parametrize('kind',['parameter','sweep'])
def test_scan_continues(s,kind):
    fn=parameter_sensitivity_analysis if kind=='parameter' else run_sensitivity_analysis
    opts={'angle_floors':[1e-6,0,.1]} if kind=='parameter' else {'learning_rates':[.1,0,.5],'geometries':['linear']}
    with pytest.raises(ValueError): fn(**s,**opts)
    r=fn(**s,**opts,on_error='record')
    assert r['failed']==1 and r['unexecuted']==0 and r['successful']==(3 if kind=='parameter' else 2)
    assert r['all_scenarios_stable'] is None and r['status']=='partial_failure'
    assert r['records'][-1]['status']=='success'
    assert next(x for x in r['records'] if x['status']=='failed')['failure_kind']=='invalid_parameters'
    json.dumps(jsonable(r),allow_nan=False)
    if kind=='sweep':
        with pytest.raises(ValueError,match='incomplete'): rankings_are_stable(r)

@pytest.mark.parametrize('kind',['parameter','sweep'])
def test_scan_runtime_failure(s,monkeypatch,kind):
    original=DecisionPipeline.run
    def fail(self,cm,dm):
        if self.angle_floor==.1: raise RuntimeError('criterion network did not converge')
        return original(self,cm,dm)
    monkeypatch.setattr(DecisionPipeline,'run',fail)
    if kind=='parameter': r=parameter_sensitivity_analysis(**s,angle_floors=[.1,1e-6],on_error='record')
    else:r=run_sensitivity_analysis(**s,angle_floor=.1,on_error='record')
    failure=next(x for x in r['records'] if x['status']=='failed')
    assert failure['failure_kind']=='execution_failure' and failure['exception_type']=='RuntimeError'
    assert r['all_scenarios_stable'] is None

def test_success_scan_stability(s):
    r=run_sensitivity_analysis(**s,on_error='record')
    assert rankings_are_stable(r) and r['all_scenarios_stable'] and r['failed']==0
    assert parameter_sensitivity_analysis(**s,on_error='record')['all_scenarios_stable']

@pytest.mark.parametrize('options',[{'rho':2},{'alpha':-1},{'beta':101},{'initial_pheromone':0},
    {'max_iterations':0},{'min_iterations':201},{'patience':False},{'improvement_tolerance':-1},
    {'exact_threshold':9},{'keep_log':'false'},{'unknown':2}])
def test_inactive_options_validated(s,options):
    for enabled in (False,True):
        with pytest.raises(ValueError): DecisionPipeline(s['criterion_labels'],s['alternative_labels'],include_sequence=enabled,aco_options=options)
    with pytest.raises((ValueError,TypeError)): aco_sequence([[MFV(.6,.3,.1)]],[1],['A'],**options)

def test_normalized_options_and_status(s):
    for enabled in (False,True):
        c=run(s,include_sequence=enabled,aco_options={'rho':np.longdouble('.2'),'patience':np.int64(3)}).configuration
        assert type(c['aco_options']['rho']) is float and type(c['aco_options']['patience']) is int
        assert c['sequence_execution_status']==('completed' if enabled else 'disabled')
        json.dumps(c,allow_nan=False)

@pytest.mark.parametrize('fn',[parameter_sensitivity_analysis,run_sensitivity_analysis])
def test_unknown_policy_rejected(s,fn):
    with pytest.raises(ValueError,match='on_error'):fn(**s,on_error='ignore')
