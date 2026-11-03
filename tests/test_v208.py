import importlib.util
from pathlib import Path
import pytest


@pytest.mark.parametrize('mutation', ['none', 'title', 'wheel', 'sdist_description', 'sdist_readme', 'history'])
def test_release_documentation(mutation):
    spec = importlib.util.spec_from_file_location('release_docs', Path(__file__).parents[1] / 'docs/verify_release.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    readme = '# MolFuzzy 2.0.10\n\nResearch toolkit.\n'
    if mutation == 'history':
        readme += '\n## Version 2.0.4\nHistorical changes.\n'
    descriptions = [readme, readme]
    archived = readme
    if mutation == 'title':
        readme = readme.replace('2.0.10', '2.0.4')
    elif mutation == 'wheel':
        descriptions[0] = descriptions[0].replace('2.0.10', '2.0.4')
    elif mutation == 'sdist_description':
        descriptions[1] += 'Stale content'
    elif mutation == 'sdist_readme':
        archived += 'Stale content'
    if mutation in ('none', 'history'):
        module.verify_documentation('2.0.10', readme, descriptions, archived)
    else:
        with pytest.raises(ValueError):
            module.verify_documentation('2.0.10', readme, descriptions, archived)
