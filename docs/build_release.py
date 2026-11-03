"""Build in a fresh temporary checkout and publish only verified distributions.
Run after installing development extras. Requires network for build isolation.
"""
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from verify_release import verify

root=Path(__file__).resolve().parents[1]
with tempfile.TemporaryDirectory(prefix='molfuzzy-release-') as tmp:
    checkout=Path(tmp)/'source'
    shutil.copytree(root,checkout,ignore=shutil.ignore_patterns('distributions','dist','build','*.egg-info','__pycache__','.pytest_cache','SHA256SUMS.txt','.git'))
    subprocess.run([sys.executable,'-m','build','--outdir',str(checkout/'distributions'),str(checkout)],check=True)
    verify(checkout)
    destination=root/'distributions'
    if destination.exists():shutil.rmtree(destination)
    shutil.copytree(checkout/'distributions',destination)
    verify(root)
print('Clean release build PASS')
