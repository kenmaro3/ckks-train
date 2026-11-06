"""Bounded read-only static inventory; never import the observed packages."""
from pathlib import Path
import ast
import hashlib
import json
import os
import sys
import time

root = Path('/home/kmihara/reboot')
site = root / 'venv/lib/python3.11/site-packages'
assert root.is_dir() and site.is_dir()

def info(p, digest=True):
    p = Path(p)
    d = {'path': str(p), 'exists': p.exists(), 'symlink': p.is_symlink()}
    if p.exists():
        s = p.stat()
        d.update(resolved=str(p.resolve()), bytes=s.st_size, mtime_ns=s.st_mtime_ns)
        if digest and p.is_file():
            assert s.st_size < 2_000_000, str(p)
            d['sha256'] = hashlib.sha256(p.read_bytes()).hexdigest()
    return d

def static_python(p):
    d = info(p)
    if not d['exists']:
        return d
    tree = ast.parse(Path(p).read_bytes())
    d['imports'] = sorted(set(
        n.module or '' for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)
    ) | set(a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names))
    for fn in ast.walk(tree):
        if isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)) and fn.name == 'check_support_sve':
            d['sve_function'] = {
                'line': fn.lineno,
                'if_nodes': sum(isinstance(n, ast.If) for n in ast.walk(fn)),
                'call_names': sorted(set(ast.unparse(n.func) for n in ast.walk(fn) if isinstance(n, ast.Call))),
                'lscpu_literal_present': any(isinstance(n, ast.Constant) and n.value == 'lscpu' for n in ast.walk(fn)),
                'exception_types': [ast.unparse(n.type) if n.type else None for n in ast.walk(fn) if isinstance(n, ast.ExceptHandler)]}
    d['direct_top_level_sve_calls'] = [n.lineno for stmt in tree.body
        if not isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.If, ast.Try))
        for n in ast.walk(stmt) if isinstance(n, ast.Call)
        and isinstance(n.func, ast.Name) and n.func.id == 'check_support_sve']
    return d

def proc(pid, expected):
    p = Path('/proc') / str(pid)
    try:
        raw = (p / 'stat').read_text()
        fields = raw[raw.rindex(')') + 2:].split()
        uid = p.stat().st_uid
        return {'pid': pid, 'exists': True, 'state': fields[0],
                'start_ticks': fields[19], 'expected_start_ticks': expected,
                'uid': uid, 'matching_own_live_process': uid == os.getuid()
                and fields[19] == expected and fields[0] not in ('Z', 'X')}
    except FileNotFoundError:
        return {'pid': pid, 'exists': False, 'expected_start_ticks': expected}

sources = [root / 'src/ReBoot/lib/utils/__init__.py', root / 'src/ReBoot/lib/utils/data.py']
sources += [root / 'src/ReBoot/experiments/3_training_encrypted' / n
            for n in ('precision_analysis.py', 'training_encrypted.py')]
sources += [site / n for n in ('numpy/testing/_private/utils.py', 'numpy/testing/__init__.py',
    'seaborn/__init__.py', 'seaborn/utils.py', 'scipy/_lib/_array_api.py',
    'scipy/_lib/array_api_compat/numpy/__init__.py')]
pth = []
for p in sorted(site.glob('*.pth')):
    d = info(p)
    lines = p.read_text(errors='replace').splitlines()
    d['executable_import_line_count'] = sum(x.startswith(('import ', 'import\t')) for x in lines)
    d['noncomment_path_line_count'] = sum(bool(x.strip()) and not x.startswith(('#','import ','import\t')) for x in lines)
    pth.append(d)
custom = [info(base / n) for base in (site, root/'venv/lib/python3.11')
          for n in ('sitecustomize.py', 'usercustomize.py')]
paths = [root/'venv/bin/python', root/'venv/pyvenv.cfg',
    root/'src/ReBoot/experiments/3_training_encrypted/experiments.sh',
    root/'src/openfhe-development', root/'src/openfhe-python',
    Path('/home/kmihara/hekit-runs/out/R-REB-5/reboot_emlp23_full/config_values.json'),
    Path('/home/kmihara/hekit-runs/out/R-REB-2/reboot_emlp1_full/data'),
    Path('/home/kmihara/hekit-exporter-f021616')]
log = Path('/home/kmihara/hekit-runs/logs/R-41-FULL-2.log')
with log.open('rb') as f:
    f.seek(max(0, log.stat().st_size-8192))
    tail = f.read().decode(errors='replace').splitlines()[-6:]
state = Path('/home/kmihara/hekit-runs/post-full-v1')
out = {
    'kind': 'read_only_static_prerequisites_not_capture_or_scientific_result',
    'observed_unix': time.time(), 'observer_python': sys.version, 'owner_uid': os.getuid(),
    'nice': os.nice(0), 'paths': [info(p, digest=p.suffix in ('.cfg','.sh','.json')) for p in paths],
    'python_sources': [static_python(p) for p in sources],
    'pth': pth, 'startup_customization_candidates': custom,
    'processes': [proc(1700979, '465769819'), proc(1739838, '467041669')],
    'full2_log': info(log, False), 'full2_log_tail': tail,
    'full2_result': info('/home/kmihara/hekit-r41full/results/quadruped/mnist196_ni_b48_full_seed0.json', False),
    'queue_state_names': sorted(p.name for p in state.iterdir()) if state.exists() else None,
    'scientific_modules_loaded': [n for n in sys.modules if n.split('.')[0] in
        {'numpy','torch','scipy','seaborn','matplotlib','lib','openfhe'}],
    'limits': 'Static AST import names are potential edges, not proof of dynamic import reachability. No target interpreter startup, full package/native hash, scientific import, job mutation or capture.'}
print(json.dumps(out, indent=2))
