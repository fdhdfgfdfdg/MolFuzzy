from decimal import Decimal,localcontext
import json
from pathlib import Path
import numpy as np
import pytest
from molfuzzy import DecisionPipeline
from molfuzzy.lom import ReliabilityDimensions,lom_expert_weights
from molfuzzy.pipeline import jsonable

TINY=np.nextafter(0.,1.)
@pytest.mark.parametrize('x,expected',[([TINY,2*TINY],[1/3,2/3]),([TINY,TINY],[.5,.5]),([0,TINY],[0,1])])
def test_subnormal_dimensions(x,expected):
    r=lom_expert_weights(ReliabilityDimensions(x,x,x))
    np.testing.assert_allclose(r['weights'],expected,rtol=2e-15,atol=0)
    assert np.max(r['R_scaled'])==1
    json.dumps(jsonable(r),allow_nan=False)

@pytest.mark.parametrize('scale',[TINY,1e-310,.25])
def test_common_scale_invariance(scale):
    a=np.array([[1.,2.],[2.,1.],[1.,1.]])
    r=lom_expert_weights(ReliabilityDimensions(*(a*scale)),[.25,.5,.25])
    reference=lom_expert_weights(ReliabilityDimensions(*(a*.25)),[.25,.5,.25])
    np.testing.assert_allclose(r['weights'],reference['weights'],rtol=2e-15,atol=0)

@pytest.mark.parametrize('inactive',[0.,1.])
def test_subnormal_beta_and_zero_weight_dimensions(inactive):
    r=lom_expert_weights(ReliabilityDimensions([TINY,2*TINY],[0,0],[inactive,inactive]),[TINY,1.,0.])
    np.testing.assert_allclose(r['weights'],[1/3,2/3],rtol=2e-15,atol=0)
    assert r['R'].tolist()==[0.,0.]
    assert r['raw_reliability_underflow']==[True,True]
    assert r['reliability_scale']['exponent2'] < -1074

@pytest.mark.parametrize('a,b',[
 ([[TINY,2*TINY],[2*TINY,TINY],[TINY,TINY]],[.2,.3,.5]),
 ([[TINY,2*TINY],[0,0],[1,1]],[TINY,1,0]),
 ([[1.,TINY],[TINY,1.],[0,0]],[TINY,1,0]),
 ([[1e-300,2e-300],[2e-310,1e-310],[0,0]],[1e-100,1,0]),
])
def test_against_decimal_reference(a,b):
    r=lom_expert_weights(ReliabilityDimensions(*a),b)
    with localcontext() as ctx:
        ctx.prec=1500
        totals=[sum(Decimal.from_float(float(b[k]))*Decimal.from_float(float(a[k][i])) for k in range(3)) for i in range(2)]
        expected=[float(v/sum(totals)) for v in totals]
    np.testing.assert_allclose(r['weights'],expected,rtol=3e-15,atol=TINY)

@pytest.mark.parametrize('x',[[0.,0.],[-TINY,TINY],[1+1e-15,1.]])
def test_invalid_dimensions_still_rejected(x):
    with pytest.raises(ValueError):lom_expert_weights(ReliabilityDimensions(x,x,x))

@pytest.mark.parametrize('x,normal',[([TINY,2*TINY],[.5,1.]),([TINY,TINY],[1.,1.])])
def test_pipeline_scaled_reliability_equivalence(x,normal):
    s=json.loads((Path(__file__).parents[1]/'examples/lean_energy_scenario.json').read_text())
    def run(v):return DecisionPipeline(s['criterion_labels'],s['alternative_labels'],reliability_dimensions=ReliabilityDimensions(v,v,v)).run(s['criteria_matrices'][:2],s['decision_matrices'][:2])
    r=run(x);ref=run(normal)
    np.testing.assert_allclose(r.ranking.scores,ref.ranking.scores,rtol=2e-14,atol=0)
    assert r.ranking.rank_groups==ref.ranking.rank_groups
    json.dumps(r.to_dict(),allow_nan=False)
