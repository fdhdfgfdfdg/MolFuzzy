import importlib.util
from pathlib import Path
import pytest

spec = importlib.util.spec_from_file_location('release209', Path(__file__).parents[1] / 'docs/verify_release.py')
release = importlib.util.module_from_spec(spec)
spec.loader.exec_module(release)


@pytest.mark.parametrize('version,date,valid', [
    ('2.0.10', '2026-10-22', True),
    ('2.0.7', '2026-10-22', False),
    ('2.0.10', '2026-02-30', False),
    ('2.0.10', 'not-a-date', False),
])
def test_citation(version, date, valid):
    text = f"version: {version}\ndate-released: '{date}'\n"
    if valid:
        release.verify_citation('2.0.10', text)
    else:
        with pytest.raises(ValueError):
            release.verify_citation('2.0.10', text)


@pytest.mark.parametrize('mutation', ['none', 'missing', 'extra', 'changed', 'empty'])
def test_evidence(mutation):
    source = {'validation/EVIDENCE_SCOPE.md': b'scope', 'validation/results.json': b'{}'}
    archived = dict(source)
    if mutation == 'missing': archived.pop('validation/EVIDENCE_SCOPE.md')
    if mutation == 'extra': archived['validation/extra.txt'] = b'extra'
    if mutation == 'changed': archived['validation/results.json'] = b'changed'
    if mutation == 'empty': source = {}; archived = {}
    if mutation == 'none':
        release.verify_evidence(source, archived)
    else:
        with pytest.raises(ValueError): release.verify_evidence(source, archived)
