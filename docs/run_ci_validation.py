"""Record actual runtime and installed-wheel verification; propagate failures."""
import json
from pathlib import Path
import platform
import subprocess
import sys
import numpy
import molfuzzy

root = Path(__file__).resolve().parents[1]
output = root / 'ci-evidence'
output.mkdir(exist_ok=True)
environment = {'python': sys.version, 'platform': platform.platform(),
               'numpy': numpy.__version__, 'molfuzzy': molfuzzy.__version__,
               'module': molfuzzy.__file__}
(output / 'environment.json').write_text(json.dumps(environment, indent=2), encoding='utf-8')
result = subprocess.run([sys.executable, str(root / 'docs/verify_installed.py')],
                        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
(output / 'installed-wheel.txt').write_text(result.stdout, encoding='utf-8')
(output / 'status.json').write_text(json.dumps({'returncode': result.returncode}), encoding='utf-8')
print(result.stdout)
sys.exit(result.returncode)
