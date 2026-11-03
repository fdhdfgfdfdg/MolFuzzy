import importlib.util
import io
import json
from pathlib import Path
import tarfile
import zipfile
import numpy as np
import pytest
from molfuzzy import MFV,DecisionPipeline
from molfuzzy.balancing import balance_experts
from molfuzzy.lom import ReliabilityDimensions,lom_expert_weights

BETAS=[[.6939355203612646,.29495429482837604,.011110184810359306],
       [.6209524452286505,.3525944931576355,.026453061613714093]]
@pytest.mark.parametrize('beta',BETAS)
def test_reliability_unit_boundary(beta):
    d=ReliabilityDimensions(np.ones(3),np.ones(3),np.ones(3))
    result=lom_expert_weights(d,beta)
    np.testing.assert_allclose(result['R'],np.ones(3),rtol=0,atol=2*np.finfo(float).eps)
    assert np.all(result['R']<=1)
    np.testing.assert_allclose(result['weights'],[1/3]*3)
    s=json.loads((Path(__file__).parents[1]/'examples/lean_energy_scenario.json').read_text())
    r=DecisionPipeline(s['criterion_labels'],s['alternative_labels'],reliability_dimensions=d,beta=beta).run(s['criteria_matrices'],s['decision_matrices'])
    assert r.balancing.converged
    json.dumps(r.to_dict(),allow_nan=False)

@pytest.mark.parametrize('value',[-1e-15,1+1e-15])
def test_reliability_external_bounds_remain_strict(value):
    with pytest.raises(ValueError):
        lom_expert_weights(ReliabilityDimensions([value],[1],[1]))

@pytest.mark.parametrize('rate',[.1,.5,1.])
@pytest.mark.parametrize('cap',[1,150])
def test_vectorized_consensus_matches_object_recurrence(rate,cap):
    rng=np.random.default_rng(73)
    matrices=[[[MFV(*v) for v in row] for row in rng.dirichlet([1,1,1],size=(4,4))] for _ in range(3)]
    r=balance_experts(matrices,[.2,.3,.5],learning_rate=rate,max_iterations=cap)
    target=np.array([[v.as_tuple() for v in row] for row in r.target])
    current=np.array([[[v.as_tuple() for v in row] for row in matrix] for matrix in matrices])
    for _ in range(r.iterations):
        raw=(1-rate)*current+rate*target
        current=np.array([[[MFV(*v).as_tuple() for v in row] for row in matrix] for matrix in raw])
    actual=np.array([[[v.as_tuple() for v in row] for row in matrix] for matrix in r.balanced.values()])
    np.testing.assert_array_equal(actual,current)
    per=np.max(abs(actual-target),axis=(1,2,3))
    assert r.residual==max(per)==r.residual_history[-1]
    assert r.per_expert_residual==per.tolist()

@pytest.mark.parametrize('mutation',['none','source_missing','source_extra','wheel_extra','sdist_missing','nested_extra'])
def test_release_module_inventory(tmp_path,mutation):
    spec=importlib.util.spec_from_file_location('release_check',Path(__file__).parents[1]/'docs/verify_release.py')
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    source={'molfuzzy/__init__.py':b'__version__="2.0.10"\n','molfuzzy/ranking.py':b'x=1\n'}
    wheel=dict(source);sdist=dict(source)
    if mutation=='source_missing':source.pop('molfuzzy/ranking.py')
    if mutation=='source_extra':source['molfuzzy/extra.py']=b'x=2\n'
    if mutation=='nested_extra':source['molfuzzy/sub/extra.py']=b'x=2\n'
    if mutation=='wheel_extra':wheel['molfuzzy/extra.py']=b'x=2\n'
    if mutation=='sdist_missing':sdist.pop('molfuzzy/ranking.py')
    for name,data in source.items():
        p=tmp_path/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(data)
    citation=b"version: 2.0.10\ndate-released: '2026-10-22'\n"
    (tmp_path/'CITATION.cff').write_bytes(citation)
    evidence={'validation/EVIDENCE_SCOPE.md':b'scope', 'validation/results.json':b'{}'}
    (tmp_path/'validation').mkdir()
    for name,data in evidence.items():(tmp_path/name).write_bytes(data)
    dist=tmp_path/'distributions';dist.mkdir()
    readme=b'# MolFuzzy 2.0.10\n'
    (tmp_path/'README.md').write_bytes(readme)
    meta=b'Name: molfuzzy\nVersion: 2.0.10\n\n'+readme
    with zipfile.ZipFile(dist/'molfuzzy-2.0.10-py3-none-any.whl','w') as z:
        for name,data in wheel.items():z.writestr(name,data)
        z.writestr('molfuzzy-2.0.10.dist-info/METADATA',meta)
    with tarfile.open(dist/'molfuzzy-2.0.10.tar.gz','w:gz') as t:
        for name,data in {**sdist,**evidence,'CITATION.cff':citation,'PKG-INFO':meta,'README.md':readme}.items():
            info=tarfile.TarInfo('molfuzzy-2.0.10/'+name);info.size=len(data);t.addfile(info,io.BytesIO(data))
    if mutation=='none':assert m.verify(tmp_path)=='2.0.10'
    else:
        with pytest.raises(ValueError,match='module set mismatch'):m.verify(tmp_path)
