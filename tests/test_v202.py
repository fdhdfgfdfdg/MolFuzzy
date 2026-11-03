import json
from dataclasses import asdict
from pathlib import Path
import numpy as np
import pytest
from molfuzzy import MFV,DecisionPipeline,aco_sequence
from molfuzzy.mfv import normalize_angle,register_geometry
from molfuzzy.lom import ReliabilityDimensions
from molfuzzy.sensitivity import run_sensitivity_analysis,parameter_sensitivity_analysis,perturbation_analysis

@pytest.fixture
def s():return json.loads((Path(__file__).parents[1]/'examples/lean_energy_scenario.json').read_text())

def run(s,**kw):return DecisionPipeline(s['criterion_labels'],s['alternative_labels'],**kw).run(s['criteria_matrices'],s['decision_matrices'])

@pytest.mark.parametrize('method',['sweep','parameter','perturbation'])
def test_diagnostic_failure_preserved(s,method):
    opts=dict(criterion_weights=[1,0,0,0],cognitive_max_iterations=1,balance_max_iterations=1)
    if method=='sweep':record=asdict(run_sensitivity_analysis(**s,learning_rates=(.1,),geometries=('linear',),**opts)[0])
    elif method=='parameter':record=parameter_sensitivity_analysis(**s,**opts)[0]
    else:
        record=perturbation_analysis(**s,trials=1,**opts)
        assert not record['trial_records'][0]['diagnostics']['criterion_network']['converged']
        assert not record['leave_one_expert_out'][0]['diagnostics']['criterion_network']['converged']
    d=record['diagnostics']
    assert d['scoring_valid'] and d['criterion_weight_source']=='user'
    assert not d['criterion_network']['converged'] and not d['consensus']['converged']
    assert d['criterion_network']['stop_reason']=='max_iterations' and len(d['warnings'])==2
    assert record['alternative_labels']==s['alternative_labels']
    json.dumps(record,allow_nan=False)

def test_custom_reliability_leaveout_matches_manual(s):
    dims=ReliabilityDimensions([.2,.7,.9],[.4,.5,.8],[.6,.4,.9])
    r=perturbation_analysis(**s,trials=2,reliability_dimensions=dims)
    assert r['successful_trials']==2 and r['leave_one_out_status']=='success'
    for i,row in enumerate(r['leave_one_expert_out']):
        sub=dict(s);sub['criteria_matrices']=[x for j,x in enumerate(s['criteria_matrices']) if j!=i];sub['decision_matrices']=[x for j,x in enumerate(s['decision_matrices']) if j!=i]
        d=ReliabilityDimensions(*(np.delete(getattr(dims,k),i) for k in ('consistency','stability','suitability')))
        expected=run(sub,reliability_dimensions=d)
        np.testing.assert_allclose(row['scores'],expected.ranking.scores)
    assert dims.consistency==[.2,.7,.9]

def test_zero_remaining_reliability_is_skipped(s):
    dims=ReliabilityDimensions([1,0,0],[1,0,0],[1,0,0])
    r=perturbation_analysis(**s,trials=1,reliability_dimensions=dims)
    assert r['leave_one_expert_out'][0]['status']=='skipped'
    assert r['successful_trials']==1

@pytest.mark.parametrize('perturb,omit',[(False,True),(True,False),(False,False)])
def test_independent_analysis_switches(s,perturb,omit):
    r=perturbation_analysis(**s,trials=2,include_perturbations=perturb,include_leave_one_out=omit)
    assert len(r['trial_records'])==(2 if perturb else 0)
    assert len(r['leave_one_expert_out'])==(3 if omit else 0)
    if not perturb: assert r['perturbation_status']=='skipped'
    if not omit: assert r['leave_one_out_status']=='skipped'

def test_leaveout_failure_does_not_block_trials(s,monkeypatch):
    original=DecisionPipeline.run
    def fail_subset(self,cm,dm):
        if len(cm)==2:raise RuntimeError('deliberate subset failure')
        return original(self,cm,dm)
    monkeypatch.setattr(DecisionPipeline,'run',fail_subset)
    r=perturbation_analysis(**s,trials=2)
    assert r['leave_one_out_status']=='failed' and r['successful_trials']==2
    assert all(x['status']=='failed' and 'deliberate' in x['reason'] for x in r['leave_one_expert_out'])

@pytest.mark.parametrize('value',['false','true',0,1,None,[],np.bool_(False)])
def test_log_flag_rejects_non_boolean(s,value):
    with pytest.raises(ValueError):aco_sequence([[MFV(.6,.3,.1)]],[1],['A'],keep_log=value)
    with pytest.raises(ValueError):DecisionPipeline(s['criterion_labels'],s['alternative_labels'],aco_options={'keep_log':value})

@pytest.mark.parametrize('enabled',[False,True])
def test_log_flag_behavior(enabled):
    r=aco_sequence([[MFV(.6,.3,.1)],[MFV(.2,.7,.1)]],[1],['A','B'],exact_threshold=0,keep_log=enabled)
    assert bool(r.log)==enabled

@pytest.mark.parametrize('angle',[-1,1e308,np.inf,np.nan,np.pi+1e-5])
def test_invalid_angle_rejected(angle):
    with pytest.raises(ValueError):normalize_angle(angle)

def test_angle_boundary_tiny_scale():
    register_geometry('v202_tiny',np.finfo(float).tiny)
    assert np.isfinite(normalize_angle(np.pi,'v202_tiny'))
    assert normalize_angle(0,'v202_tiny')==0

def test_provenance_complete_and_stable(s):
    dims=ReliabilityDimensions([.2,.7,.9],[.4,.5,.8],[.6,.4,.9])
    a=run(s,reliability_dimensions=dims,include_sequence=True).configuration
    assert a['aco_options']['alpha']==1 and a['aco_options']['exact_threshold']==8
    assert a['reliability_dimensions']['consistency']==[.2,.7,.9]
    assert set(a['environment'])=={'python','numpy','platform'}
    assert len(a['input_sha256'])==64
    b=run(s,reliability_dimensions=dims).configuration
    assert a['input_sha256']==b['input_sha256']
    altered=json.loads(json.dumps(s));altered['decision_matrices'][0][0][0]='N'
    assert run(altered).configuration['input_sha256']!=a['input_sha256']
    assert run(s,aco_options={'rho':.2}).configuration['aco_options']['rho']==.2

def test_parameter_records_have_label_mapping(s):
    rows=parameter_sensitivity_analysis(**s,angle_floors=[.1])
    for row in rows:
        assert row['alternative_labels']==s['alternative_labels']
        assert len(row['scores'])==len(row['alternative_labels'])
        assert row['diagnostics']['scoring_valid']
