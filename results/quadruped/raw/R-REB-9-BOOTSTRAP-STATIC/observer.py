"""Read fixed startup/import sources as data; no target interpreter or package imports."""
from pathlib import Path
import ast
import hashlib
import json
import os
import time

root = Path('/home/kmihara/reboot')
site = root / 'venv/lib/python3.11/site-packages'
base = (root / 'venv/bin/python').resolve().parent.parent

def info(p):
    p = Path(p)
    d = {'path': str(p), 'exists': p.exists(), 'resolved': str(p.resolve())}
    if p.is_file():
        raw = p.read_bytes()
        assert len(raw) < 2_000_000
        d.update(bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())
    return d

class Facts(ast.NodeVisitor):
    def __init__(self):
        self.scope = []
        self.imports = []
        self.top_calls = []
        self.flag_names = []

    def visit_FunctionDef(self, n):
        self.scope.append('function:' + n.name)
        self.generic_visit(n)
        self.scope.pop()

    visit_AsyncFunctionDef = visit_FunctionDef

    def visit_ClassDef(self, n):
        self.scope.append('class:' + n.name)
        self.generic_visit(n)
        self.scope.pop()

    def visit_If(self, n):
        self.scope.append('if@' + str(n.lineno))
        self.generic_visit(n)
        self.scope.pop()

    def visit_Try(self, n):
        self.scope.append('try@' + str(n.lineno))
        self.generic_visit(n)
        self.scope.pop()

    def visit_Import(self, n):
        self.imports.append({'line': n.lineno, 'scope': list(self.scope),
                             'modules': [a.name for a in n.names]})

    def visit_ImportFrom(self, n):
        self.imports.append({'line': n.lineno, 'scope': list(self.scope),
            'module': n.module, 'level': n.level, 'names': [a.name for a in n.names]})

    def visit_Call(self, n):
        if not any(s.startswith(('function:', 'class:')) for s in self.scope):
            self.top_calls.append({'line': n.lineno, 'scope': list(self.scope), 'callee': ast.unparse(n.func)})
        self.generic_visit(n)

    def visit_Name(self, n):
        if n.id in ('_SUPPORTS_SVE', 'check_support_sve'):
            self.flag_names.append({'line': n.lineno, 'scope': list(self.scope), 'name': n.id})

def source(p):
    d = info(p)
    if Path(p).is_file():
        f = Facts()
        f.visit(ast.parse(Path(p).read_bytes()))
        d.update(imports=f.imports, module_calls=f.top_calls, optional_flag_names=f.flag_names)
    return d

pth = []
for folder in (site, base/'lib/python3.11/site-packages'):
    for p in sorted(folder.glob('*.pth')):
        d = info(p)
        lines = []
        for i, line in enumerate(p.read_text().splitlines(), 1):
            if line.startswith(('import ', 'import\t')):
                f = Facts()
                f.visit(ast.parse(line))
                lines.append({'line': i, 'imports': f.imports, 'calls': f.top_calls})
            elif line.strip() and not line.startswith('#'):
                lines.append({'line': i, 'path_entry': line.strip()})
        d['startup_lines'] = lines
        pth.append(d)

sources = [site/'_virtualenv.py', site/'_distutils_hack/__init__.py', base/'lib/python3.11/site.py']
sources += [site/n for n in ('numpy/__init__.py', 'seaborn/_statistics.py',
    'seaborn/relational.py', 'scipy/_lib/array_api_compat/_internal.py')]
sources += [root/'src/ReBoot'/n for n in ('lib/utils/nn.py', 'lib/models/local_loss_models.py')]
custom = [info(folder/name) for folder in (site, base/'lib/python3.11', base/'lib/python3.11/site-packages')
          for name in ('sitecustomize.py','usercustomize.py')]
print(json.dumps({'kind': 'static_bootstrap_and_import_facts_not_runtime_approval',
    'observed_unix': time.time(), 'nice': os.nice(0), 'base_resolved': str(base),
    'pth': pth, 'sources': [source(p) for p in sources], 'customization_candidates': custom,
    'limits': 'Fixed source-only inspection; AST scope facts are not dynamic execution. No target startup, scientific imports, package-wide consumer scan, process mutation or capture.'}, indent=2))
