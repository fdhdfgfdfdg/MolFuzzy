import json
from pathlib import Path
import numpy as np
import pytest
from molfuzzy import MFV,DecisionPipeline
from molfuzzy.balancing import balance_experts

@pytest.mark.parametrize('vertex',[(1,0,0),(0,1,0),(0,0,1)])
def test_balance_vertices(vertex):
    w=[.23198402839841684,.554702073152752,.2133138984488311]
    r=balance_experts([[[MFV(*vertex)]]]*3,w)
    assert r.target[0][0].as_tuple()==vertex
    assert r.residual==0 and r.converged

@pytest.mark.parametrize('rate',[.1,.5,1])
def test_balance_residual_matches_return(rate):
    r=balance_experts([[[MFV(1,0,0)]],[[MFV(0,1,0)]],[[MFV(0,0,1)]]],[.2,.3,.5],learning_rate=rate,max_iterations=3)
    target=np.array([[v.as_tuple() for v in row] for row in r.target])
    actual=[np.max(abs(np.array([[v.as_tuple() for v in row] for row in m])-target)) for m in r.balanced.values()]
    assert r.residual==max(actual)==r.residual_history[-1]
    assert r.per_expert_residual==actual

@pytest.mark.parametrize('sequence',[False,True])
def test_pipeline_previously_rejected_weights(sequence):
    s=json.loads((Path(__file__).parents[1]/'examples/lean_energy_scenario.json').read_text())
    w=[.6939355203612646,.29495429482837604,.011110184810359306]
    r=DecisionPipeline(s['criterion_labels'],s['alternative_labels'],expert_weights=w,include_sequence=sequence).run(s['criteria_matrices'],s['decision_matrices'])
    assert r.balancing.converged and np.all(np.isfinite(r.ranking.scores))
    json.dumps(r.to_dict(),allow_nan=False)
