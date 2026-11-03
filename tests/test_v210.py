import importlib.util
from pathlib import Path
import pytest

spec = importlib.util.spec_from_file_location('release210', Path(__file__).parents[1] / 'docs/verify_release.py')
release = importlib.util.module_from_spec(spec)
spec.loader.exec_module(release)


@pytest.mark.parametrize('text,valid', [
    ("version: 2.0.10\ndate-released: '2026-10-22INVALID'\n", False),
    ("version: '2.0.10\ndate-released: '2026-10-22'\n", False),
    ("version: 2.0.10\ndate-released: '2026-10-22\n", False),
    ("version: 2.0.10\nversion: 2.0.10\ndate-released: '2026-10-22'\n", False),
    ("version: 2.0.10\ndate-released: '2026-10-22'\ndate-released: '2026-10-22'\n", False),
    ("- version: 2.0.10\n- date-released: '2026-10-22'\n", False),
    ("version: 2.0.10\ndate-released: null\n", False),
    ("version: 2.0.10\ndate-released: 2026-10-22T00:00:00\n", False),
    ("version: 2.0.10\ndate-released: 2026-1-2\n", False),
    ("version: 2.0.10\ndate-released: 2026-10-22\n", True),
    ('version: "2.0.10"\ndate-released: "2026-10-22" # date\n', True),
    ("version: '2.0.10'\ndate-released: '2026-10-22'\n", True),
])
def test_complete_citation_yaml(text, valid):
    if valid:
        release.verify_citation('2.0.10', text)
    else:
        with pytest.raises(ValueError): release.verify_citation('2.0.10', text)
