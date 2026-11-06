"""Fixed-list read-only stdlib identity/import observation, not target execution."""
import ast
import hashlib
import json
import os
from pathlib import Path
import stat
import time

BASE = Path('/home/kmihara/.local/share/uv/python/cpython-3.11.16-linux-x86_64-gnu')
ROOT = BASE / 'lib/python3.11'
FILES = (
    'importlib/metadata/__init__.py', 'importlib/metadata/_adapters.py',
    'importlib/metadata/_collections.py', 'importlib/metadata/_functools.py',
    'importlib/metadata/_itertools.py', 'importlib/metadata/_meta.py',
    'importlib/metadata/_text.py', 'email/__init__.py', 'email/parser.py',
    'email/feedparser.py', 'email/policy.py', 'email/_policybase.py',
    'email/message.py', 'email/utils.py', 'email/header.py', 'email/charset.py',
    'email/encoders.py', 'email/errors.py', 'email/_encoded_words.py',
    'email/iterators.py', 'email/base64mime.py', 'email/quoprimime.py',
    'csv.py', 'platform.py', 'subprocess.py', 'argparse.py', 'gettext.py',
    'locale.py', 'pathlib.py', 'encodings/__init__.py', 'encodings/aliases.py',
    'encodings/ascii.py', 'encodings/utf_8.py', 'encodings/latin_1.py',
)

def read_regular(path, cap):
    if path.resolve() != path:
        raise ValueError('noncanonical source path')
    initial = path.lstat()
    if not stat.S_ISREG(initial.st_mode) or initial.st_size > cap:
        raise ValueError('source pre-open kind or bound')
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    try:
        before = os.fstat(fd)
        if not stat.S_ISREG(before.st_mode) or before.st_size > cap:
            raise ValueError('source kind or bound')
        with os.fdopen(fd, 'rb', closefd=False) as stream:
            data = stream.read(cap + 1)
        after = os.fstat(fd)
        if len(data) > cap or (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns):
            raise ValueError('source changed or exceeded bound')
        return data, before
    finally:
        os.close(fd)

class Imports(ast.NodeVisitor):
    def __init__(self):
        self.scope = []
        self.imports = []
        self.dynamic = []

    def scoped(self, node):
        self.scope.append(node.name)
        self.generic_visit(node)
        self.scope.pop()

    visit_FunctionDef = scoped
    visit_AsyncFunctionDef = scoped
    visit_ClassDef = scoped

    def visit_Import(self, node):
        self.imports.append({'line': node.lineno, 'scope': list(self.scope), 'kind': 'import', 'names': [a.name for a in node.names]})

    def visit_ImportFrom(self, node):
        self.imports.append({'line': node.lineno, 'scope': list(self.scope), 'kind': 'from', 'module': node.module, 'level': node.level, 'names': [a.name for a in node.names]})

    def visit_Call(self, node):
        name = node.func.id if isinstance(node.func, ast.Name) else node.func.attr if isinstance(node.func, ast.Attribute) else None
        if name in ('__import__', 'import_module', 'find_spec', 'exec_module'):
            self.dynamic.append({'line': node.lineno, 'scope': list(self.scope), 'call_name': name})
        self.generic_visit(node)

def main():
    record = {'kind': 'read_only_fixed_stdlib_source_observation_not_scientific_result', 'started_unix': time.time(), 'nice': os.nice(0), 'uid': os.getuid(), 'files': [], 'limits': {'files': 34, 'per_source_bytes': 524288, 'total_source_bytes': 4194304}, 'scope': 'Static identity/import atoms only; no transitive closure claim, target startup, package/native inventory, scientific import, context, capture or process mutation.'}
    status = 0
    try:
        if record['nice'] < 10 or len(FILES) != 34:
            raise ValueError('observer nice or fixed-list mismatch')
        data, _ = read_regular(BASE / 'bin/python3.11', 33554432)
        pin = hashlib.sha256(data).hexdigest()
        if pin != '2874a0b9344d06b7767aebb1e6e25a759ffcbdb544e99400ecc74dc6092d1174':
            raise ValueError('base executable changed')
        record['base_executable_sha256'] = pin
        total = 0
        for relative in FILES:
            path = ROOT / relative
            data, info = read_regular(path, 524288)
            total += len(data)
            if total > 4194304:
                raise ValueError('total source bound')
            visitor = Imports()
            visitor.visit(ast.parse(data, filename=str(path)))
            record['files'].append({'path': str(path), 'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest(), 'uid': info.st_uid, 'mode': stat.S_IMODE(info.st_mode), 'imports': visitor.imports, 'dynamic_import_sites': visitor.dynamic})
        record['total_source_bytes'] = total
        record['status'] = 'fixed_source_observation_complete_no_target_startup'
    except Exception as error:
        record['status'] = 'incomplete_observation'
        record['error'] = {'type': type(error).__name__, 'message': str(error)}
        status = 1
    record['finished_unix'] = time.time()
    print(json.dumps(record, indent=2))
    return status

if __name__ == '__main__':
    raise SystemExit(main())
