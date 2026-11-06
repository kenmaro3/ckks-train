"""One bounded creator child, then static inspection; never start the new venv."""
from pathlib import Path
import hashlib
import json
import os
import stat
import subprocess
import time

BASE = Path('/home/kmihara/.local/share/uv/python/cpython-3.11.16-linux-x86_64-gnu')
PARENT = Path('/home/kmihara/reboot')
DEST = PARENT / 'exporter-venv-task54-v1'
STDLIB = BASE / 'lib/python3.11'
SITE = STDLIB / 'site-packages'
PYTHON = BASE / 'bin/python3.11'
PINS = {
    PYTHON: '2874a0b9344d06b7767aebb1e6e25a759ffcbdb544e99400ecc74dc6092d1174',
    SITE / 'distutils-precedence.pth': '2638ce9e2500e572a5e0de7faed6661eb569d1b696fcba07b0dd223da5f5d224',
    SITE / '_distutils_hack/__init__.py': 'df81e6bcba34ee3e3952f776551fb669143b9490fdd6c4caeb32609f97e985b4',
    STDLIB / 'site.py': '0577e4b379d8ef4aa0ecf88a264ef18c10f72ea72119afae36524d1aae77c9bf',
    STDLIB / 'venv/__init__.py': '8671bc5810ee3e52a5ae7aa3b625764688cf12513b2f8e1b0b82aaec93f118a6',
    STDLIB / 'venv/__main__.py': '722537c68c0622f8293d39bb6ab1288f3637d8dc45d6f9aae96e49af8145ca36',
}
ENV = {'PATH': '/usr/bin:/bin', 'LANG': 'C', 'LC_ALL': 'C',
       'SETUPTOOLS_USE_DISTUTILS': 'local', 'OMP_NUM_THREADS': '1',
       'MKL_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1', 'RAYON_NUM_THREADS': '1'}
ARGV = [str(PYTHON), '-I', '-B', '-m', 'venv', '--without-pip', '--symlinks', str(DEST)]


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def run_creator(argv, cwd, environment, record, timeout=90, cleanup_timeout=5):
    """Own one child until terminal evidence or an explicitly unresolved failure."""
    record['child_started_unix'] = time.time()
    child = subprocess.Popen(argv, cwd=cwd, env=environment, stdin=subprocess.DEVNULL,
                             stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout = stderr = b''
    collected = False
    primary_error = None
    try:
        record['child_pid'] = child.pid
        try:
            procstat = Path('/proc') / str(child.pid) / 'stat'
            record['child_proc_stat_at_launch'] = procstat.read_text()
        except OSError as exc:
            record['child_proc_stat_at_launch'] = None
            record['proc_stat_error'] = {'type': type(exc).__name__, 'message': str(exc)}
        try:
            stdout, stderr = child.communicate(timeout=timeout)
            collected = True
        except subprocess.TimeoutExpired:
            record['timed_out'] = True
    except BaseException as exc:
        primary_error = exc
        record['controller_error'] = {'type': type(exc).__name__, 'message': str(exc)}
    finally:
        cleanup_errors = []
        if not collected:
            try:
                if child.poll() is None:
                    child.kill()  # This exact Popen child only; no PID search or groups.
            except OSError as exc:
                cleanup_errors.append({'operation': 'kill', 'type': type(exc).__name__,
                                       'message': str(exc)})
            try:
                stdout, stderr = child.communicate(timeout=cleanup_timeout)
                collected = True
            except BaseException as exc:
                cleanup_errors.append({'operation': 'collect', 'type': type(exc).__name__,
                                       'message': str(exc)})
                if isinstance(exc, subprocess.TimeoutExpired):
                    stdout = exc.output if isinstance(exc.output, bytes) else b''
                    stderr = exc.stderr if isinstance(exc.stderr, bytes) else b''
        try:
            terminal = child.poll() is not None
        except OSError as exc:
            terminal = False
            cleanup_errors.append({'operation': 'poll', 'type': type(exc).__name__,
                                   'message': str(exc)})
        record.update(child_finished_unix=time.time(), child_returncode=child.returncode,
                      child_terminal_observed=terminal, child_streams_complete=collected,
                      cleanup_errors=cleanup_errors,
                      child_stdout_hex=stdout.hex(), child_stderr_hex=stderr.hex(),
                      child_stdout_sha256=hashlib.sha256(stdout).hexdigest(),
                      child_stderr_sha256=hashlib.sha256(stderr).hexdigest())
        if not terminal or not collected:
            record['child_cleanup_status'] = 'unresolved_terminal_or_streams'
        else:
            record['child_cleanup_status'] = 'terminal_and_streams_collected'
    if primary_error is not None:
        raise primary_error
    require(record['child_terminal_observed'] and record['child_streams_complete'],
            'creator termination or streams unresolved; inspect exact owned child')
    require(not record.get('timed_out'), 'creator timed out; partial destination retained')
    require(child.returncode == 0, 'creator failed; partial destination retained')


def inspect_created_tree(destination, record, expected_uid=1000):
    """Inspect only a fixed shallow layout; reject unknown kinds before content I/O."""
    expected_dirs = {'bin', 'include', 'include/python3.11', 'lib', 'lib/python3.11',
                     'lib/python3.11/site-packages'}
    expected_files = {'pyvenv.cfg', 'bin/Activate.ps1', 'bin/activate', 'bin/activate.csh',
                      'bin/activate.fish'}
    expected_links = {'lib64', 'bin/python', 'bin/python3', 'bin/python3.11'}
    expected = {p: kind for kind, names in [('directory', expected_dirs),
        ('file', expected_files), ('symlink', expected_links)] for p in names}
    tree = record['created_tree'] = []
    # Parents precede children. A directory must first have passed lstat kind checks.
    directories = [''] + sorted(expected_dirs, key=lambda p: (p.count('/'), p))
    seen = set()
    for relative in directories:
        folder = destination / relative
        if relative:
            require(relative in seen, 'expected directory missing: ' + relative)
        require(folder.is_dir() and not folder.is_symlink(), 'unexpected directory kind')
        with os.scandir(folder) as entries:
            for item in entries:
                require(len(tree) < 128, 'unexpected large created tree')
                path = Path(item.path)
                name = str(path.relative_to(destination))
                metadata = item.stat(follow_symlinks=False)
                kind = ('symlink' if stat.S_ISLNK(metadata.st_mode) else
                        'directory' if stat.S_ISDIR(metadata.st_mode) else
                        'file' if stat.S_ISREG(metadata.st_mode) else 'other')
                entry = {'path': name, 'uid': metadata.st_uid,
                         'mode': oct(metadata.st_mode & 0o777), 'kind': kind}
                tree.append(entry)  # Preserve partial evidence on the next failed check.
                require(name in expected and expected[name] == kind,
                        'unexpected created name/kind: ' + name)
                require(metadata.st_uid == expected_uid, 'unexpected created owner')
                require(name not in seen, 'duplicate created entry')
                seen.add(name)
                if kind == 'file':
                    require(metadata.st_size <= 65536, 'oversized config/script: ' + name)
                    entry['bytes'] = metadata.st_size
                elif kind == 'symlink':
                    entry['target'] = os.readlink(path)
    require(seen == set(expected), 'created layout is incomplete')
    # Only after all names/kinds/sizes pass may any expected file content be read.
    for entry in tree:
        path = destination / entry['path']
        if entry['kind'] == 'file':
            entry['sha256'] = digest(path)
        elif entry['kind'] == 'symlink':
            entry['resolved'] = str(path.resolve())
    tree.sort(key=lambda entry: entry['path'])
    return tree


record = {'kind': 'operational_empty_venv_creation_not_scientific_result',
          'started_unix': time.time(), 'argv': ARGV, 'environment': ENV,
          'cwd': str(PARENT), 'destination': str(DEST), 'nice': os.nice(0),
          'assumption': 'No concurrent writer to this new owned destination.',
          'status': 'preflight', 'preflight': {}}
try:
    require(os.getuid() == 1000 and os.geteuid() == 1000, 'unexpected owner')
    require(os.nice(0) == 10, 'expected nice10')
    require(PARENT.resolve() == PARENT and PARENT.is_dir(), 'unexpected canonical parent')
    require(PARENT.stat().st_uid == 1000 and os.access(PARENT, os.W_OK | os.X_OK),
            'parent is not owned writable directory')
    require(BASE.resolve() == BASE, 'unexpected base alias')
    require(not os.path.lexists(DEST), 'destination already exists, including symlink')
    record['preflight']['parent'] = {'uid': PARENT.stat().st_uid,
        'mode': oct(PARENT.stat().st_mode & 0o777), 'resolved': str(PARENT.resolve())}
    record['preflight']['destination_lexists'] = False
    checked = []
    for path, expected in PINS.items():
        require(path.resolve() == path and stat.S_ISREG(path.lstat().st_mode),
                'unexpected pinned file kind or canonical path: ' + str(path))
        actual = digest(path)
        require(actual == expected, 'changed pinned file: ' + str(path))
        checked.append({'path': str(path), 'bytes': path.stat().st_size, 'sha256': actual})
    record['preflight']['pinned_files'] = checked
    require(sorted(p.name for p in SITE.glob('*.pth')) == ['distutils-precedence.pth'],
            'changed base pth inventory')
    absent = [BASE / 'bin/pyvenv.cfg', BASE / 'pyvenv.cfg', BASE / 'lib/python311.zip']
    for folder in [STDLIB, STDLIB / 'lib-dynload', SITE]:
        for name in ['sitecustomize', 'usercustomize']:
            absent += [folder / (name + suffix) for suffix in
                       ['.py', '.pyc', '/__init__.py', '/__init__.pyc']]
            require(not list(folder.glob(name + '*.so')), 'unexpected customization extension')
            require(not os.path.lexists(folder / name), 'unexpected customization package')
    for path in absent:
        require(not os.path.lexists(path), 'unexpected prefix/customization file: ' + str(path))
    record['preflight']['absent_paths'] = [str(p) for p in absent]
    # Dispatch exactly once after every preflight succeeds. Timeout only stops this child.
    record['status'] = 'creator_running'
    run_creator(ARGV, PARENT, ENV, record)
    record['status'] = 'static_postcheck'
    require(DEST.is_dir() and not DEST.is_symlink(), 'unexpected destination kind')
    inspect_created_tree(DEST, record)
    cfgpath = DEST / 'pyvenv.cfg'
    raw = cfgpath.read_bytes()
    record['config_hex'] = raw.hex()
    record['config_sha256'] = hashlib.sha256(raw).hexdigest()
    cfg = {}
    for line in raw.decode('utf-8').splitlines():
        key, separator, value = line.partition('=')
        require(bool(separator) and key.strip() not in cfg, 'malformed config')
        cfg[key.strip()] = value.strip()
    require(cfg.get('include-system-site-packages') == 'false', 'system site enabled')
    require(cfg.get('home') == str(BASE / 'bin'), 'wrong base home')
    require(cfg.get('executable') == str(PYTHON), 'wrong base executable')
    require(cfg.get('version') == '3.11.16', 'unexpected config version')
    require('--without-pip' in cfg.get('command', '') and str(DEST) in cfg.get('command', ''),
            'unexpected config command')
    own_site = DEST / 'lib/python3.11/site-packages'
    require(own_site.is_dir() and not own_site.is_symlink() and not list(own_site.iterdir()),
            'own site is not empty directory')
    require(os.readlink(DEST / 'lib64') == 'lib', 'unexpected lib64 alias')
    for name in ['python', 'python3', 'python3.11']:
        p = DEST / 'bin' / name
        require(p.is_symlink() and p.resolve() == PYTHON, 'unexpected interpreter link')
        require(digest(p) == PINS[PYTHON], 'interpreter hash changed')
    record['status'] = 'created_static_checks_passed_new_interpreter_not_started'
except BaseException as exc:
    record.update(status='failed_no_retry_or_cleanup', error_type=type(exc).__name__, error=str(exc))
finally:
    record['finished_unix'] = time.time()
    print(json.dumps(record, indent=2))
if record['status'] != 'created_static_checks_passed_new_interpreter_not_started':
    raise SystemExit(1)
