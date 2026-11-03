import itertools
import json
from pathlib import Path
import numpy as np
import pytest
from molfuzzy import MFV,DecisionPipeline,rank_alternatives,balance_experts,aco_sequence
from molfuzzy.mfv import angle_between,nr_matrix,register_geometry,GEOMETRIES,reciprocal_matrix
from molfuzzy.lom import normalize_reliability,ReliabilityDimensions,lom_expert_weights
from molfuzzy.data import linguistic_rect_matrix_to_mfv,aggregate_rectangular_mfv
from molfuzzy.sensitivity import run_sensitivity_analysis,rankings_are_stable,perturbation_analysis
from molfuzzy.cli import main,_load_scenario

@pytest.fixture
def scenario():
    return json.loads((Path(__file__).parents[1]/'examples/lean_energy_scenario.json').read_text())

def pipeline(s,**kwargs):
    return DecisionPipeline(s['criterion_labels'],s['alternative_labels'],**kwargs).run(s['criteria_matrices'],s['decision_matrices'])

@pytest.mark.parametrize('triple',[(np.nan,0,1),(np.inf,0,0),(-.1,.6,.5),(1.1,0,0),(.6,.3,.3)])
def test_mfv_rejects_invalid(triple):
    with pytest.raises(ValueError): MFV(*triple)

@pytest.mark.parametrize('value',[np.nan,np.inf,-1,0])
def test_geometry_rejects_invalid(value):
    with pytest.raises(ValueError): register_geometry('bad',value)

@pytest.mark.parametrize('triple',[(.2,.7,.1),(.6,.3,.1),(.8,.15,.05),(1,0,0)])
def test_identical_angle_exactly_zero(triple):
    v=MFV(*triple)
    assert angle_between([v],[v])==0

def test_near_duplicate_continuity():
    a=MFV(.6,.3,.1); b=MFV(.6+1e-12,.3-1e-12,.1); c=MFV(.2,.7,.1)
    x=nr_matrix([[a],[a],[c]]); y=nr_matrix([[a],[b],[c]])
    assert np.all(np.isfinite(x)) and np.max(abs(x-y))<1e-9
    assert np.max(abs(x.sum(axis=0)-1))<1e-12

@pytest.mark.parametrize('weights',[[0,0],[-1,2],[1],[np.nan,1],[.2,.2]])
def test_invalid_weights(weights):
    with pytest.raises(ValueError): rank_alternatives([[MFV(.6,.3,.1)]*2],weights,['A'])

def test_benefit_cost_and_tie():
    low=[MFV(.2,.7,.1)]; high=[MFV(.95,.05,0)]
    assert rank_alternatives([low,high],[1],['L','H']).ranking_labels==['H','L']
    assert rank_alternatives([low,high],[1],['L','H'],[-1]).ranking_labels==['L','H']
    assert rank_alternatives([high,high],[1],['Z','A']).rank_groups==[['A','Z']]

@pytest.mark.parametrize('seed',range(10))
def test_random_dominance_monotonicity(seed):
    rng=np.random.default_rng(seed); w=rng.dirichlet(np.ones(4))
    low=rng.uniform(.05,.5,4); high=low+rng.uniform(.01,.3,4)
    matrix=[[MFV(x,1-x,0) for x in row] for row in [low,high]]
    r=rank_alternatives(matrix,w,['low','high'])
    assert r.scores[1]>r.scores[0] and r.ranking_labels[0]=='high'

def test_all_criterion_permutations(scenario):
    base=pipeline(scenario)
    for perm in itertools.permutations(range(4)):
        s=dict(scenario)
        s['criterion_labels']=[scenario['criterion_labels'][i] for i in perm]
        s['criteria_matrices']=[[[m[i][j] for j in perm] for i in perm] for m in scenario['criteria_matrices']]
        s['decision_matrices']=[[[row[j] for j in perm] for row in m] for m in scenario['decision_matrices']]
        r=pipeline(s)
        np.testing.assert_allclose(r.expert_weights['weights'],base.expert_weights['weights'],atol=1e-12)
        np.testing.assert_allclose(r.criteria.weights[np.argsort(perm)],base.criteria.weights,atol=1e-12)
        np.testing.assert_allclose(r.ranking.scores,base.ranking.scores,atol=1e-12)
        assert r.ranking.rank_groups==base.ranking.rank_groups

def test_alternative_and_expert_permutations(scenario):
    base=pipeline(scenario,include_sequence=True)
    for perm in [(4,3,2,1,0),(2,0,4,1,3)]:
        s=dict(scenario); s['alternative_labels']=[scenario['alternative_labels'][i] for i in perm]
        s['decision_matrices']=[[m[i] for i in perm] for m in scenario['decision_matrices']]
        r=pipeline(s,include_sequence=True)
        assert r.ranking.rank_groups==base.ranking.rank_groups
        assert r.sequence.sequence_labels==base.sequence.sequence_labels
    for perm in itertools.permutations(range(3)):
        s=dict(scenario)
        s['criteria_matrices']=[scenario['criteria_matrices'][i] for i in perm]
        s['decision_matrices']=[scenario['decision_matrices'][i] for i in perm]
        r=pipeline(s)
        np.testing.assert_allclose(r.ranking.scores,base.ranking.scores,atol=1e-12)
        assert r.ranking.rank_groups==base.ranking.rank_groups

def test_geometry_invariance(scenario):
    base=pipeline(scenario,include_sequence=True)
    register_geometry('custom',.123)
    for g in (*GEOMETRIES,'custom'):
        r=pipeline(scenario,geometry=g,include_sequence=True)
        np.testing.assert_array_equal(r.criteria.weights,base.criteria.weights)
        np.testing.assert_array_equal(r.sequence.final_decision_matrix,base.sequence.final_decision_matrix)

def test_consensus_residual_and_low_rate(scenario):
    r=pipeline(scenario)
    assert r.balancing.iterations>1 and r.balancing.converged and r.balancing.residual<=1e-6
    assert np.all(np.diff(r.balancing.residual_history)<=1e-14)
    slow=pipeline(scenario,learning_rate=1e-10,balance_max_iterations=2)
    assert not slow.balancing.converged and slow.balancing.stop_reason=='max_iterations'
    np.testing.assert_array_equal(slow.ranking.scores,r.ranking.scores)

@pytest.mark.parametrize('rate',[0,-1,1.1,np.nan,np.inf])
def test_reject_learning_rate(scenario,rate):
    with pytest.raises(ValueError): pipeline(scenario,learning_rate=rate)

def test_analytic_consensus_and_weighted_aggregation():
    a=[[MFV(.2,.7,.1)]]; b=[[MFV(.95,.05,0)]]
    r=balance_experts([a,b],[.25,.75],learning_rate=1)
    assert r.iterations==1 and r.residual==0 and r.converged
    expected=.25*np.array(a[0][0].as_tuple())+.75*np.array(b[0][0].as_tuple())
    np.testing.assert_allclose(r.target[0][0].as_tuple(),expected)
    np.testing.assert_allclose(aggregate_rectangular_mfv([a,b],[.25,.75])[0][0].as_tuple(),expected)

def test_custom_reliability_and_criterion_weights(scenario):
    d=ReliabilityDimensions(np.array([1,0,0]),np.array([1,0,0]),np.array([1,0,0]))
    r=pipeline(scenario,reliability_dimensions=d,criterion_weights=[1,0,0,0])
    np.testing.assert_array_equal(r.expert_weights['weights'],[1,0,0])
    np.testing.assert_allclose(r.ranking.scores,[.95,.825,.95,.825,.825])
    assert r.configuration['criterion_weight_source']=='user'

def test_analytic_lom_and_zero_reliability():
    d=ReliabilityDimensions(np.array([.3,.9]),np.array([.4,.8]),np.array([.5,.7]))
    r=lom_expert_weights(d)
    np.testing.assert_allclose(r['weights'],[1/3,2/3])
    assert r['objective']==0
    with pytest.raises(ValueError): normalize_reliability([0,0])

@pytest.mark.parametrize('mutation',['experts','rows','cols','extra_rows','extra_cols','diagonal','missing','duplicate','empty'])
def test_pipeline_invalid_inputs(scenario,mutation):
    if mutation=='experts': scenario['decision_matrices'].pop()
    elif mutation=='rows': scenario['decision_matrices'][1].pop()
    elif mutation=='cols': scenario['criteria_matrices'][1][0].pop()
    elif mutation=='extra_rows': scenario['criteria_matrices'][0].append([None]*4)
    elif mutation=='extra_cols': scenario['decision_matrices'][0][0].append('H')
    elif mutation=='diagonal': scenario['criteria_matrices'][0][0][0]='H'
    elif mutation=='missing': scenario['criteria_matrices'][0][0][1]=None
    elif mutation=='duplicate': scenario['alternative_labels'][0]=scenario['alternative_labels'][1]
    else: scenario['decision_matrices']=[]
    with pytest.raises(ValueError): pipeline(scenario)

def test_cognitive_nonconvergence_exposed(scenario):
    with pytest.raises(RuntimeError): pipeline(scenario,cognitive_max_iterations=1)

def test_exact_sequence_matches_independent_enumeration(scenario):
    r=pipeline(scenario,include_sequence=True).sequence; f=r.final_decision_matrix
    best=min(sum(f[p[i],p[i+1]] for i in range(4)) for p in itertools.permutations(range(5)))
    assert r.path_cost==pytest.approx(best) and r.optimality_proven
    assert not hasattr(r,'ranking_labels')

def test_heuristic_sequence_stopping_reproducibility_and_permutation(scenario):
    r=pipeline(scenario)
    kw=dict(exact_threshold=0,min_iterations=10,patience=10,max_iterations=40,seed=2)
    a=aco_sequence(r.aggregated_decision,r.criteria.weights,scenario['alternative_labels'],**kw)
    b=aco_sequence(r.aggregated_decision,r.criteria.weights,scenario['alternative_labels'],**kw)
    assert a.sequence==b.sequence and a.iterations>=10 and not a.optimality_proven
    assert all(np.diff(a.best_cost_history)<=1e-14)
    p=[4,3,2,1,0]
    c=aco_sequence([r.aggregated_decision[i] for i in p],r.criteria.weights,[scenario['alternative_labels'][i] for i in p],**kw)
    assert c.sequence_labels==a.sequence_labels

@pytest.mark.parametrize('kwargs',[{'rho':-1},{'rho':1.1},{'max_iterations':0},{'min_iterations':0},{'seed':-1},{'alpha':np.nan},{'exact_threshold':9},{'patience':0}])
def test_aco_invalid_parameters(kwargs):
    with pytest.raises(ValueError): aco_sequence([[MFV(.2,.7,.1)],[MFV(.95,.05,0)]],[1],['a','b'],**kwargs)

def test_single_criterion_alternative_and_duplicate_scenarios():
    r=DecisionPipeline(['C'],['A'],include_sequence=True).run([[[None]]],[[['H']]])
    assert r.ranking.rank_groups==[['A']] and r.sequence.path_cost==0 and r.sequence.fitness==0
    r=DecisionPipeline(['C'],['A','B']).run([[[None]]],[[['H'],['H']]])
    assert r.ranking.rank_groups==[['A','B']]

def test_sensitivity_explicit_outputs(scenario):
    rows=run_sensitivity_analysis(**scenario,include_sequence=True)
    assert len(rows)==15 and rankings_are_stable(rows,'alternatives')
    assert rankings_are_stable(rows,'criteria') and rankings_are_stable(rows,'sequence')
    assert len({r.balance_iterations for r in rows})>1
    with pytest.raises(ValueError): rankings_are_stable([])

def test_perturbations_reproducible(scenario):
    a=perturbation_analysis(**scenario,trials=4); b=perturbation_analysis(**scenario,trials=4)
    assert a==b and len(a['leave_one_expert_out'])==3

def test_json_stdout_and_error_status(scenario,tmp_path,capsys):
    p=tmp_path/'s.json'; p.write_text(json.dumps(scenario))
    assert main(['run',str(p),'--json'])==0
    out=capsys.readouterr(); payload=json.loads(out.out)
    assert payload['configuration']['version']=='2.0.11' and out.err==''
    p.write_text('{"a":NaN}')
    assert main(['run',str(p),'--json'])==2
    out=capsys.readouterr(); assert out.out=='' and 'nonfinite' in out.err
    p.write_text('{"a":1,"a":2}')
    with pytest.raises(ValueError): _load_scenario(p)

def test_invalid_utility_types():
    for types in [[0],[True],['benefit'],[1,1]]:
        with pytest.raises(ValueError): rank_alternatives([[MFV(1,0,0)]],[1],['A'],types)

def test_helpers_reject_empty_or_bad_matrices():
    with pytest.raises(ValueError): MFV.average([])
    with pytest.raises(ValueError): linguistic_rect_matrix_to_mfv([])
    with pytest.raises(ValueError): reciprocal_matrix(np.array([[0,np.nan],[0,0]]))
    with pytest.raises(ValueError): aggregate_rectangular_mfv([[[MFV(1,0,0)]],[[MFV(1,0,0),MFV(1,0,0)]]])
