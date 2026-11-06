"""Read bounded CPython creator/startup facts without starting that interpreter."""
from pathlib import Path
import ast
import hashlib
import json
import os
import re
import time
import zipfile

original_python = Path('/home/kmihara/reboot/venv/bin/python')
base_python = original_python.resolve()
base = base_python.parent.parent
stdlib = base / 'lib/python3.11'
site = stdlib / 'site-packages'


def identity(path):
    p = Path(path)
    d = {'path': str(p), 'exists': p.exists(), 'symlink': p.is_symlink(),
         'resolved': str(p.resolve())}
    if p.is_file():
        stat = p.stat()
        h = hashlib.sha256()
        with p.open('rb') as f:
            for block in iter(lambda: f.read(1024 * 1024), b''):
                h.update(block)
        d.update(bytes=stat.st_size, sha256=h.hexdigest(), uid=stat.st_uid,
                 mode=oct(stat.st_mode & 0o777))
    return d


def symbol(node):
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return symbol(node.value) + '.' + node.attr
    if isinstance(node, ast.Call):
        return symbol(node.func) + '()'
    return '<' + type(node).__name__ + '>'


def expr(node):
    if isinstance(node, ast.Constant):
        value = node.value
        if value is None or isinstance(value, (bool, int, float)):
            return {'constant': value}
        if isinstance(value, str) and len(value) <= 100 and re.fullmatch(r'[A-Za-z0-9_./=:-]*', value):
            return {'short_string': value}
        return {'constant_type': type(value).__name__}
    if isinstance(node, (ast.Name, ast.Attribute)):
        return {'symbol': symbol(node)}
    if isinstance(node, ast.Call):
        return {'call': symbol(node.func), 'args': [expr(a) for a in node.args],
                'keywords': [{'name': k.arg, 'value': expr(k.value)} for k in node.keywords]}
    if isinstance(node, ast.Compare):
        return {'left': expr(node.left), 'operators': [type(o).__name__ for o in node.ops],
                'right': [expr(c) for c in node.comparators]}
    if isinstance(node, ast.BoolOp):
        return {'boolean': type(node.op).__name__, 'values': [expr(v) for v in node.values]}
    if isinstance(node, ast.UnaryOp):
        return {'unary': type(node.op).__name__, 'operand': expr(node.operand)}
    return {'node': type(node).__name__}


class Facts(ast.NodeVisitor):
    def __init__(self):
        self.scope = []
        self.facts = []

    def record(self, n, kind, **values):
        self.facts.append({'line': n.lineno, 'scope': list(self.scope), 'kind': kind, **values})

    def visit_FunctionDef(self, n):
        self.record(n, 'function', name=n.name)
        self.scope.append('function:' + n.name)
        self.generic_visit(n)
        self.scope.pop()

    visit_AsyncFunctionDef = visit_FunctionDef

    def visit_ClassDef(self, n):
        self.record(n, 'class', name=n.name)
        self.scope.append('class:' + n.name)
        self.generic_visit(n)
        self.scope.pop()

    def visit_If(self, n):
        self.record(n, 'condition', test=expr(n.test))
        for label, body in [('then', n.body), ('else', n.orelse)]:
            self.scope.append(label + '@' + str(n.lineno))
            for child in body:
                self.visit(child)
            self.scope.pop()

    def visit_Call(self, n):
        self.record(n, 'call', callee=symbol(n.func),
                    args=[expr(a) for a in n.args])
        self.generic_visit(n)

    def visit_Import(self, n):
        self.record(n, 'import', modules=[a.name for a in n.names])

    def visit_ImportFrom(self, n):
        self.record(n, 'import_from', module=n.module, level=n.level,
                    names=[a.name for a in n.names])


def source_facts(path):
    d = identity(path)
    raw = Path(path).read_bytes()
    assert len(raw) < 250000
    visitor = Facts()
    visitor.visit(ast.parse(raw))
    d['facts'] = visitor.facts
    return d


pth = []
for p in sorted(site.glob('*.pth')):
    d = identity(p)
    d['lines'] = []
    for index, line in enumerate(p.read_text().splitlines(), 1):
        if line.startswith(('import ', 'import\t')):
            nodes = ast.parse(line).body
            d['lines'].append({'line': index, 'statements': [
                {'kind': type(n).__name__,
                 'value': expr(n.value) if hasattr(n, 'value') else None,
                 'targets': [symbol(t) for t in n.targets] if isinstance(n, ast.Assign) else [],
                 'imports': [a.name for a in n.names] if isinstance(n, ast.Import) else []}
                for n in nodes]})
        elif line.strip() and not line.startswith('#'):
            d['lines'].append({'line': index, 'path_entry': line.strip()})
    pth.append(d)

custom = []
for folder in [stdlib, stdlib / 'lib-dynload', site]:
    for name in ['sitecustomize', 'usercustomize']:
        for suffix in ['.py', '.pyc', '/__init__.py', '/__init__.pyc']:
            custom.append(identity(folder / (name + suffix)))
        for candidate in sorted(folder.glob(name + '*.so')):
            custom.append(identity(candidate))

zpath = base / 'lib/python311.zip'
zinfo = identity(zpath)
if zpath.is_file():
    with zipfile.ZipFile(zpath) as z:
        zinfo['customization_names'] = [n for n in z.namelist()
            if n.split('/')[0] in ('sitecustomize.py', 'sitecustomize.pyc',
                                  'usercustomize.py', 'usercustomize.pyc',
                                  'sitecustomize', 'usercustomize')]

print(json.dumps({
    'kind': 'static_creator_prerequisites_not_runtime_approval',
    'observed_unix': time.time(), 'nice': os.nice(0),
    'original_invocation': identity(original_python), 'base_executable': identity(base_python),
    'base': str(base), 'base_pth': pth,
    'sources': [source_facts(p) for p in [site / '_distutils_hack/__init__.py',
        stdlib / 'site.py', stdlib / 'venv/__init__.py', stdlib / 'venv/__main__.py']],
    'customization_candidates': custom, 'stdlib_zip': zinfo,
    'selected_observer_environment': {k: os.environ.get(k) for k in [
        'SETUPTOOLS_USE_DISTUTILS', 'PYTHONHOME', 'PYTHONPATH', 'PYTHONUSERBASE']},
    'candidate_destination': identity('/home/kmihara/reboot/exporter-venv-task54-v1'),
    'limits': 'Fixed source/hash/path facts only. No target startup, environment creation, '
              'scientific imports, full package/native hashing, job mutation or capture. '
              'Observer environment is not a launched creator environment.'
}, indent=2))
