"""Check distribution identity/content and optionally the complete release ZIP."""
import argparse
import ast
import hashlib
import yaml
from datetime import date
from email.parser import Parser
from pathlib import Path
import tarfile
import zipfile

def verify_documentation(version, readme, descriptions, sdist_readme):
    # Metadata writers may use the host newline convention. Compare document
    # text canonically while module and evidence checks below remain byte-exact.
    normalize = lambda text: text.replace('\r\n', '\n').replace('\r', '\n')
    readme = normalize(readme)
    if readme.splitlines()[0] != f"# MolFuzzy {version}":
        raise ValueError("README current version title mismatch")
    if any(normalize(text).strip() != readme.strip() for text in descriptions):
        raise ValueError("distribution description differs from README")
    if normalize(sdist_readme) != readme:
        raise ValueError("sdist README differs from source")


class CitationLoader(yaml.SafeLoader):
    """Safe YAML with duplicate-key rejection and dates kept as text."""

CitationLoader.yaml_implicit_resolvers = {
    key: [(tag, regex) for tag, regex in values
          if tag != 'tag:yaml.org,2002:timestamp']
    for key, values in yaml.SafeLoader.yaml_implicit_resolvers.items()
}


def citation_mapping(loader, node):
    result = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=True)
        if not isinstance(key, str) or key in result:
            raise ValueError('citation keys must be unique strings')
        result[key] = loader.construct_object(value_node, deep=True)
    return result


CitationLoader.add_constructor('tag:yaml.org,2002:map', citation_mapping)


def verify_citation(version, text):
    try:
        metadata = yaml.load(text, Loader=CitationLoader)
    except (yaml.YAMLError, ValueError) as exc:
        raise ValueError('invalid citation YAML: ' + str(exc)) from exc
    if not isinstance(metadata, dict) or metadata.get('version') != version:
        raise ValueError('citation version mismatch')
    released = metadata.get('date-released')
    if not isinstance(released, str):
        raise ValueError('citation date must be an ISO date string')
    try:
        parsed = date.fromisoformat(released)
    except ValueError as exc:
        raise ValueError('invalid citation date') from exc
    if parsed.isoformat() != released:
        raise ValueError('invalid citation date')


def verify_evidence(source, archived):
    required = {'validation/EVIDENCE_SCOPE.md', 'validation/results.json'}
    if not required <= set(source) or set(source) != set(archived):
        raise ValueError('validation evidence inventory mismatch')
    if any(source[name] != archived[name] for name in source):
        raise ValueError('validation evidence content mismatch')


def verify(root,archive=None):
    root=Path(root)
    tree=ast.parse((root/'molfuzzy/__init__.py').read_text())
    version=next(ast.literal_eval(n.value) for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='__version__' for t in n.targets))
    citation=(root/'CITATION.cff').read_text(encoding='utf-8')
    verify_citation(version, citation)
    names={f'molfuzzy-{version}-py3-none-any.whl',f'molfuzzy-{version}.tar.gz'}
    actual={f.name for f in (root/'distributions').iterdir()}
    if actual!=names:raise ValueError(f'distribution files must be exactly {sorted(names)}; got {sorted(actual)}')
    with zipfile.ZipFile(root/'distributions'/f'molfuzzy-{version}-py3-none-any.whl') as w,tarfile.open(root/'distributions'/f'molfuzzy-{version}.tar.gz') as t:
        meta=Parser().parsestr(w.read(f'molfuzzy-{version}.dist-info/METADATA').decode())
        smeta=Parser().parsestr(t.extractfile(f'molfuzzy-{version}/PKG-INFO').read().decode())
        for item in (meta,smeta):
            if item['Name']!='molfuzzy' or item['Version']!=version:raise ValueError('distribution metadata version mismatch')
        verify_documentation(version, (root/'README.md').read_text(encoding='utf-8'),
                             [meta.get_payload(), smeta.get_payload()],
                             t.extractfile(f'molfuzzy-{version}/README.md').read().decode())
        if t.extractfile(f'molfuzzy-{version}/CITATION.cff').read() != (root/'CITATION.cff').read_bytes():
            raise ValueError('sdist citation mismatch')
        prefix=f'molfuzzy-{version}/'
        evidence={f.relative_to(root).as_posix():f.read_bytes() for f in (root/'validation').rglob('*') if f.is_file()}
        members=[m for m in t.getmembers() if m.isfile() and m.name.startswith(prefix+'validation/')]
        if len(members) != len({m.name for m in members}):
            raise ValueError('duplicate validation evidence')
        verify_evidence(evidence, {m.name[len(prefix):]:t.extractfile(m).read() for m in members})
        source={f.relative_to(root).as_posix():f.read_bytes() for f in (root/'molfuzzy').rglob('*.py')}
        wheel_paths=[x for x in w.namelist() if x.startswith('molfuzzy/') and x.endswith('.py')]
        tar_prefix=f'molfuzzy-{version}/'
        tar_paths=[x.name[len(tar_prefix):] for x in t.getmembers() if x.isfile() and x.name.startswith(tar_prefix+'molfuzzy/') and x.name.endswith('.py')]
        if len(wheel_paths)!=len(set(wheel_paths)) or len(tar_paths)!=len(set(tar_paths)):
            raise ValueError('duplicate distribution module')
        if set(source)!=set(wheel_paths) or set(source)!=set(tar_paths):
            raise ValueError('module set mismatch between source, wheel and sdist')
        for rel,content in source.items():
            if w.read(rel)!=content or t.extractfile(tar_prefix+rel).read()!=content:
                raise ValueError('source mismatch: '+rel)
    if archive:
        prefix=f'molfuzzy-{version}/'
        with zipfile.ZipFile(archive) as z:
            paths=z.namelist()
            if len(paths)!=len(set(paths)) or any(not x.startswith(prefix) for x in paths):raise ValueError('invalid archive layout')
            manifest=z.read(prefix+'SHA256SUMS.txt').decode().splitlines()
            entries=dict(line.split('  ',1)[::-1] for line in manifest)
            expected={prefix+x for x in entries}|{prefix+'SHA256SUMS.txt'}
            if set(paths)!=expected or len(entries)!=len(manifest):raise ValueError('manifest coverage mismatch')
            if {Path(x).name for x in paths if x.startswith(prefix+'distributions/')}!=names:raise ValueError('archive contains obsolete distribution')
            for rel,digest in entries.items():
                content=z.read(prefix+rel)
                if hashlib.sha256(content).hexdigest()!=digest or content!=(root/rel).read_bytes():raise ValueError('checksum/source mismatch: '+rel)
    return version

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1]);p.add_argument('--archive',type=Path)
    a=p.parse_args();print('Release verification PASS:',verify(a.root,a.archive))
