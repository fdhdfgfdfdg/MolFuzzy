"""Run copied tests and CLI outside the checkout against the installed wheel.

Invoke with the Python interpreter of the intended isolated environment.
"""
from pathlib import Path
import json
import shutil
import subprocess
import sys
import tempfile

ROOT=Path(__file__).resolve().parents[1]
def main():
    with tempfile.TemporaryDirectory(prefix='molfuzzy-wheel-check-') as tmp:
        root=Path(tmp)
        shutil.copytree(ROOT/'tests',root/'tests',ignore=shutil.ignore_patterns('__pycache__'))
        shutil.copytree(ROOT/'docs',root/'docs',ignore=shutil.ignore_patterns('__pycache__'))
        shutil.copytree(ROOT/'examples',root/'examples',ignore=shutil.ignore_patterns('__pycache__'))
        probe="import molfuzzy,numpy,json; from pathlib import Path; print(json.dumps({'module':molfuzzy.__file__,'version':molfuzzy.__version__,'numpy':numpy.__version__})); assert 'site-packages' in str(Path(molfuzzy.__file__)); assert molfuzzy.__version__=='2.0.11'"
        subprocess.run([sys.executable,'-c',probe],cwd=root,check=True)
        subprocess.run([sys.executable,'-m','pytest','-q','tests'],cwd=root,check=True)
        result=subprocess.run([sys.executable,'-m','molfuzzy','run','examples/lean_energy_scenario.json','--json'],cwd=root,check=True,capture_output=True,text=True)
        payload=json.loads(result.stdout)
        assert payload['configuration']['version']=='2.0.11'
        print('Installed wheel CLI JSON: PASS',flush=True)
if __name__=='__main__':main()
