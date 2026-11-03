import json
import os
from pathlib import Path
import numpy as np
import pytest
from molfuzzy import MFV, DecisionPipeline
from molfuzzy.cli import main, _write_output
from molfuzzy.mfv import nr_matrix, register_geometry, reciprocal_matrix, column_normalize, GEOMETRIES, MIN_ANGLE_FLOOR
from molfuzzy.validation import weights, scalar
from molfuzzy.lom import normalize_reliability, consensus_reliability, ReliabilityDimensions
from molfuzzy.sensitivity import run_sensitivity_analysis, perturbation_analysis, parameter_sensitivity_analysis

@pytest.fixture
def scenario():
    return json.loads((Path(__file__).parents[1]/'examples/lean_energy_scenario.json').read_text())

def run(s, **options):
    return DecisionPipeline(s['criterion_labels'],s['alternative_labels'],**options).run(s['criteria_matrices'],s['decision_matrices'])

@pytest.mark.parametrize('alias',['same','relative','symlink','hardlink'])
@pytest.mark.parametrize('force',[False,True])
def test_input_preserved(scenario,tmp_path,capsys,alias,force):
    src=tmp_path/'source.json'; src.write_text(json.dumps(scenario)); original=src.read_bytes()
    dst=src
    if alias=='relative': dst=tmp_path/'.'/'source.json'
    elif alias in ('symlink','hardlink'):
        dst=tmp_path/'alias.json'
        try:
            if alias=='symlink': dst.symlink_to(src)
            else: os.link(src,dst)
        except OSError: pytest.skip('filesystem does not permit this link type')
    args=['run',str(src),'--output',str(dst),'--json']+(['--force'] if force else [])
    assert main(args)==2
    assert src.read_bytes()==original
    assert capsys.readouterr().out==''

def test_output_no_clobber_and_atomic(tmp_path,monkeypatch):
    src=tmp_path/'in.json';src.write_text('input')
    dst=tmp_path/'out.json';dst.write_text('previous')
    with pytest.raises(ValueError): _write_output(src,dst,'next')
    assert dst.read_text()=='previous'
    original=os.replace
    def fail(*args): raise OSError('simulated commit failure')
    monkeypatch.setattr(os,'replace',fail)
    with pytest.raises(OSError): _write_output(src,dst,'next',True)
    assert dst.read_text()=='previous'
    assert not list(tmp_path.glob('.molfuzzy-*'))
    monkeypatch.setattr(os,'replace',original)
    _write_output(src,dst,'next',True)
    assert dst.read_text()=='next'
    fresh=tmp_path/'new.json';_write_output(src,fresh,'complete')
    assert fresh.read_text()=='complete'

@pytest.mark.parametrize('name',GEOMETRIES)
def test_builtin_geometry_immutable(name):
    rows=[[MFV(.6,.3,.1)],[MFV(.60001,.29999,.1)],[MFV(.2,.7,.1)]]
    baseline=nr_matrix(rows)
    with pytest.raises(ValueError):register_geometry(name,1e6)
    np.testing.assert_array_equal(nr_matrix(rows),baseline)

def test_custom_geometry_does_not_change_nr():
    register_geometry('v201_custom',1e100)
    rows=[[MFV(.6,.3,.1)],[MFV(.60001,.29999,.1)],[MFV(.2,.7,.1)]]
    np.testing.assert_array_equal(nr_matrix(rows,'v201_custom'),nr_matrix(rows))
    with pytest.raises(ValueError): register_geometry('v201_custom',3)

def test_explicit_weights_keep_failed_network_diagnostic(scenario):
    r=run(scenario,criterion_weights=[1,0,0,0],cognitive_max_iterations=1)
    assert not r.criteria.converged
    assert r.configuration['criterion_weight_source']=='user'
    np.testing.assert_allclose(r.ranking.scores,r.ranking.utilities[:,0])
    assert 'max_iterations' in r.summary()
    with pytest.raises(RuntimeError): run(scenario,cognitive_max_iterations=1)

@pytest.mark.parametrize('floor',[MIN_ANGLE_FLOOR,1e-100,1e-6,1e308])
def test_nr_extreme_floor(floor):
    rows=[[MFV(.6,.3,.1)],[MFV(.6,.3,.1)],[MFV(.2,.7,.1)]]
    with np.errstate(over='raise',invalid='raise',divide='raise'):
        nr=nr_matrix(rows,angle_floor=floor)
    assert np.isfinite(nr).all()
    np.testing.assert_allclose(nr.sum(axis=0),1)
    np.testing.assert_array_equal(np.diag(nr),0)

def test_reciprocal_rejects_unrepresentable_inverse():
    with pytest.raises(ValueError):reciprocal_matrix(np.zeros((2,2)),1e-320)
    assert np.isfinite(reciprocal_matrix(np.zeros((2,2)),np.finfo(float).tiny)).all()

@pytest.mark.parametrize('bad',[np.array([.5+2j,.5-2j]),[True,False],[True,0.],['.5','.5'],np.array([.5+0j,.5+0j],dtype=object)])
def test_weights_reject_lossy_values(bad):
    with pytest.raises(ValueError):weights(bad,2)
    with pytest.raises(ValueError):normalize_reliability(bad)

@pytest.mark.parametrize('function',[reciprocal_matrix,column_normalize,consensus_reliability])
def test_numeric_matrix_rejects_complex(function):
    with pytest.raises(ValueError):function(np.ones((2,2),dtype=complex))

def test_scalar_rejects_numpy_complex():
    with pytest.raises(ValueError):scalar(np.complex128(1+0j),'value')

def test_container_equivalence(scenario):
    a=run_sensitivity_analysis(**scenario,learning_rates=(.1,),geometries=('linear',))
    s=dict(scenario);s['criterion_labels']=np.array(s['criterion_labels'])
    assert run_sensitivity_analysis(**s,learning_rates=(.1,),geometries=('linear',))==a
    expected=perturbation_analysis(**scenario,trials=3)
    s['decision_matrices']=tuple(tuple(tuple(row) for row in m) for m in s['decision_matrices'])
    assert perturbation_analysis(**s,trials=3)==expected

def test_configuration_ownership(scenario):
    w=[1.,0.,0.];types=[1,1,1,1];options={'max_iterations':60}
    p=DecisionPipeline(scenario['criterion_labels'],scenario['alternative_labels'],expert_weights=w,criterion_types=types,aco_options=options)
    def go():return p.run(scenario['criteria_matrices'],scenario['decision_matrices'])
    a=go();w[:]=[0,0,1];types[0]=-1;options['max_iterations']=1
    b=go();np.testing.assert_array_equal(a.ranking.scores,b.ranking.scores)
    a.criterion_labels[0]='changed';a.configuration['aco_options']['max_iterations']=3
    assert go().criterion_labels[0]=='EE'
    assert go().configuration['aco_options']['max_iterations']==60
    dims=ReliabilityDimensions(np.ones(3),np.ones(3),np.ones(3))
    p=DecisionPipeline(scenario['criterion_labels'],scenario['alternative_labels'],reliability_dimensions=dims)
    a=go();dims.consistency[0]=0
    np.testing.assert_array_equal(go().ranking.scores,a.ranking.scores)

def test_parameter_sensitivity_scores(scenario):
    rows=parameter_sensitivity_analysis(**scenario,criterion_weight_grid=[[1,0,0,0]],expert_weight_grid=[[1,0,0]],angle_floors=[1e-3])
    assert len(rows)==4
    for row in rows: assert len(row['scores'])==5 and len(row['adjacent_score_gaps'])==4
    expected=run(scenario,criterion_weights=[1,0,0,0])
    np.testing.assert_array_equal(rows[1]['scores'],expected.ranking.scores)

@pytest.mark.parametrize('floor',[np.nextafter(0.,1.),1e-320])
def test_angle_floor_rejected_at_entry(scenario,floor):
    from molfuzzy import aco_sequence
    rows=[[MFV(.6,.3,.1)],[MFV(.6,.3,.1)]]
    with pytest.raises(ValueError):nr_matrix(rows,angle_floor=floor)
    with pytest.raises(ValueError):aco_sequence(rows,[1],['A','B'],angle_floor=floor)
    with pytest.raises(ValueError):run(scenario,angle_floor=floor)

@pytest.mark.parametrize('threshold',[0,8])
def test_extreme_supported_traversal(threshold):
    from molfuzzy import aco_sequence
    rows=[[MFV(.6,.3,.1)],[MFV(.6,.3,.1)],[MFV(.2,.7,.1)]]
    with np.errstate(over='raise',invalid='raise',divide='raise'):
        result=aco_sequence(rows,[1],['A','B','C'],angle_floor=MIN_ANGLE_FLOOR,exact_threshold=threshold)
    assert np.isfinite(result.path_cost) and np.isfinite(result.heuristic_matrix).all()
