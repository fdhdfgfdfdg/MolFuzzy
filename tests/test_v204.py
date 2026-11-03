import json
from pathlib import Path
import numpy as np
import pytest
from molfuzzy import MFV
from molfuzzy.data import aggregate_rectangular_mfv
from molfuzzy.ranking import rank_alternatives
from molfuzzy.validation import _bounded_unit_result
from molfuzzy.sensitivity import run_sensitivity_analysis,parameter_sensitivity_analysis

W=[.23198402839841684,.554702073152752,.2133138984488311]
@pytest.fixture
def scenario():return json.loads((Path(__file__).parents[1]/'examples/lean_energy_scenario.json').read_text())

@pytest.mark.parametrize('vertex',[(1,0,0),(0,1,0),(0,0,1)])
@pytest.mark.parametrize('mode',['average','aggregate'])
def test_convex_vertices(vertex,mode):
    values=[MFV(*vertex)]*3
    r=MFV.average(values,W) if mode=='average' else aggregate_rectangular_mfv([[[v]] for v in values],W)[0][0]
    assert r.as_tuple()==vertex

@pytest.mark.parametrize('cost',[False,True])
def test_score_unit_bounds(cost):
    r=rank_alternatives([[MFV(1,0,0)]*3,[MFV(0,1,0)]*3],W,['A','B'],[-1 if cost else 1]*3)
    assert np.all((0<=r.scores)&(r.scores<=1))
    assert r.scores.tolist()==([0,1] if cost else [1,0])

@pytest.mark.parametrize('bad',[(1+1e-12,0,0),(-1e-12,1,0),(0,0,1+1e-12)])
def test_external_bounds_still_strict(bad):
    with pytest.raises(ValueError):MFV(*bad)

@pytest.mark.parametrize('bad',[[1.01],[np.nan],[-.01],[np.inf]])
def test_internal_material_errors_rejected(bad):
    with pytest.raises(ValueError):_bounded_unit_result(bad,3)

def test_convex_random_identity_and_bounds():
    rng=np.random.default_rng(51)
    for count in (3,10,100):
        for _ in range(10):
            w=rng.dirichlet(np.ones(count)); a=rng.dirichlet(np.ones(3),size=count)
            r=MFV.average([MFV(*x) for x in a],w)
            np.testing.assert_allclose(r.as_tuple(),w@a,rtol=0,atol=1e-14)
            assert all(0<=x<=1 for x in r.as_tuple())

@pytest.mark.parametrize('failure',[False,True])
def test_complete_sweep_is_json(scenario,failure):
    rates=[np.longdouble('.1'),np.longdouble('0' if failure else '.5')]
    r=run_sensitivity_analysis(**scenario,learning_rates=rates,geometries=['linear'],seeds=[np.int64(42)],on_error='record')
    assert json.loads(json.dumps(r,allow_nan=False))==r
    assert r['failed']==int(failure)

@pytest.mark.parametrize('tol',[0,1e-10,.03,1])
def test_gaps_sorted_and_labelled(scenario,tol):
    r=parameter_sensitivity_analysis(**scenario,tie_tolerance=tol)[0]
    scores=dict(zip(r['alternative_labels'],r['scores']))
    assert len(r['score_gap_pairs'])==len(scores)-1
    for pair,gap in zip(r['score_gap_pairs'],r['adjacent_score_gaps']):
        assert gap>=0 and gap==scores[pair[0]]-scores[pair[1]]
    assert r['score_gap_order']==sorted(scores,key=lambda x:(-scores[x],x))
    assert len(r['between_rank_group_gaps'])==len(r['rank_groups'])-1
    for item in r['between_rank_group_gaps']:
        assert item['gap']>=0
        assert item['gap']==min(scores[x] for x in item['higher_group'])-max(scores[x] for x in item['lower_group'])
    if tol==1:assert r['between_rank_group_gaps']==[]

def test_parameter_json_mixed_values(scenario):
    r=parameter_sensitivity_analysis(**scenario,angle_floors=[np.longdouble('.1'),np.longdouble('nan')],on_error='record')
    assert r['failed']==1
    json.dumps(r,allow_nan=False)
