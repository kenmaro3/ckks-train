#!/usr/bin/env python3
"""Offline value-only ReBoot block inputs; runtime requires separately reviewed pins.

This module imports only the standard library. ActualBackend is loaded only AFTER
approval, source, native, archive, raw-data and no-clobber checks. Use an isolated
standalone interpreter: python -I -B ... --help. A compatibility receipt is NOT a
resume token: complete mode always freshly captures and compares both models0–2.
No training, encrypted execution, accuracy, latency or scientific reference claim.
"""
from __future__ import annotations

import argparse
import ast
from datetime import datetime, timezone
import errno
import hashlib
import importlib
import importlib.abc
import importlib.machinery
import importlib.metadata
import io
import json
import math
import os
from pathlib import Path
import platform
import re
import shlex
import struct
import subprocess
import sys


class ExportError(RuntimeError):
    """A prerequisite, capture invariant or exact compatibility check failed."""


PYTHON_SOURCE_POLICY = 'validated-source-no-bytecode-v1'
NUMPY_UTILS_SHA256 = '55d1a25fe7dbe7146deb16f0827833b207b88aadad6536654b4ec7020a559e40'
OPTIONAL_HOST_QUERY_POLICY = 'numpy-sve-source-bound-denial-eperm-v2'
MODELS = ('emlp2', 'emlp3')
SHAPES = {
    'emlp2': {'linear_fwd_00': (196, 64), 'linear_lrn_00': (64, 16),
              'linear_01': (64, 32), 'linear_02': (32, 16)},
    'emlp3': {'linear_fwd_00': (196, 128), 'linear_lrn_00': (128, 16),
              'linear_fwd_01': (128, 64), 'linear_lrn_01': (64, 16),
              'linear_02': (64, 32), 'linear_03': (32, 16)},
}
DATA_SHAPES = {'train_images_14x14.npy': ('<f4', (4800, 196)),
               'train_labels.npy': ('|u1', (4800,)),
               'test_images_14x14.npy': ('<f4', (10000, 196)),
               'test_labels.npy': ('|u1', (10000,))}
RAW_PINS = {
    't10k-images-idx3-ubyte': '0fa7898d509279e482958e8ce81c8e77db3f2f8254e26661ceb7762c4d494ce7',
    't10k-images-idx3-ubyte.gz': '8d422c7b0a1c1c79245a5bcf07fe86e33eeafee792b84584aec276f5a2dbc4e6',
    't10k-labels-idx1-ubyte': 'ff7bcfd416de33731a308c3f266cc351222c34898ecbeaf847f06e48f7ec33f2',
    't10k-labels-idx1-ubyte.gz': 'f7ae60f92e00ec6debd23a6088c31dbd2371eca3ffa0defaefb259924204aec6',
    'train-images-idx3-ubyte': 'ba891046e6505d7aadcbbe25680a0738ad16aec93bde7f9b65e87a2fc25776db',
    'train-images-idx3-ubyte.gz': '440fcabf73cc546fa21475e81ea370265605f56be210a4024d2ca8f203523609',
    'train-labels-idx1-ubyte': '65a50cbbf4e906d70832878ad85ccda5333a97f0f4c3dd2ef09a8a9eef7101c5',
    'train-labels-idx1-ubyte.gz': '3552534a0a558bbed6aed32b30c495cca23d567ec52cac8be1a0730e8010255c',
}
ARCHIVE_SHA256 = '76c493e28692531d07d54337dfcf89a21d8d78fa2979e02b20dccad47fb303c5'
DATA_PINS = dict(zip(DATA_SHAPES, (
    '7b1590d8884172b853d8037063b290d20b41112f6c0aa5309769639552b9276c',
    '1d63491b1c54835583c5ca6886305d7b83d7c9ca0c3ab24433c431b1f7e84856',
    '9464f6e1544cd65d7387036bab2e71e3826b121b108d2bb4f1360e9bfb0d6d87',
    'ff7e84b144c037e7215dfa787d6773550c5db83029d9a4e7bae6e90f605f081d')))
EXP = 'experiments/3_training_encrypted'
CONFIG_NAMES = ('args', 'dataset_config', 'network_config', 'training_config', 'configs')
SEQUENCE = ('config-parser;seed;load-data;shuffle-seed42;seed;plain-constructor;'
            'seed;captured-constructor;CosineLR;10*(cumulative-shuffle;100*(step;get_lr));inspect;save')
SCHEDULE = ('constant for 5 epochs (500 steps), then cosine annealing with warm restarts '
            "every 50 steps, stepped once per batch (ReBoot's scheduler)")


def require(ok, reason):
    if not ok:
        raise ExportError(reason)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def file_hash(path):
    p = Path(path)
    require(p.is_file() and not p.is_symlink(), f'missing/nonregular/symlink prerequisite: {p}')
    h = hashlib.sha256()
    with p.open('rb') as f:
        for part in iter(lambda: f.read(1024 * 1024), b''):
            h.update(part)
    return h.hexdigest()


def json_bytes(value):
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n').encode()


def strict_json(data):
    def pairs(items):
        out = {}
        for key, value in items:
            require(key not in out, 'duplicate JSON key')
            out[key] = value
        return out
    return json.loads(data, object_pairs_hook=pairs,
                      parse_constant=lambda x: (_ for _ in ()).throw(ExportError('nonfinite JSON')))


def now():
    return datetime.now(timezone.utc).isoformat()


def write_new(path, content):
    with Path(path).open('xb') as f:
        f.write(content)


def validate_seeds(seeds, mode):
    require(mode in ('compatibility', 'complete'), 'unsupported mode')
    expected = list(range(3 if mode == 'compatibility' else 10))
    require(all(type(s) is int for s in seeds) and seeds == expected, 'missing/duplicate/out-of-range seeds')


def released_argv(model, seed):
    require(model in MODELS and type(seed) is int and 0 <= seed <= 9, 'unsupported model/seed')
    hidden = ['64', '32'] if model == 'emlp2' else ['128', '64', '32']
    return ['MNIST', '--subsample', '4800', '--num-layers', str(len(hidden) + 1),
            '--num-hidden', *hidden, '--batch-size', '48', '--num-epochs', '10', '--seed', str(seed)]


def seed_argv(original, seed):
    require(type(seed) is int and 0 <= seed <= 9, 'unsupported seed override')
    require(original.count('--seed') == 1, 'missing/duplicate seed argument')
    result = list(original)
    i = result.index('--seed')
    require(i + 1 < len(result), 'missing seed value')
    result[i + 1] = str(seed)
    return result


def read_commands(text):
    sections = {}
    model = None
    for line in text.replace('\r', '').splitlines():
        if line.strip() in ('# MLP-2', '# MLP-3'):
            model = 'emlp' + line.strip()[-1]
            require(model not in sections, 'duplicate model section')
            sections[model] = []
        elif model and line.strip().startswith('python'):
            tokens = shlex.split(line)
            require(tokens[:2] == ['python3', 'precision_analysis.py'], 'unsupported official command')
            sections[model].append(tokens[2:])
    require(set(sections) == set(MODELS), 'missing model section')
    for m, commands in sections.items():
        require(commands == [released_argv(m, s) for s in range(3)], 'unsupported official argv/seeds')
    return sections


def configuration_assignments(text):
    """Select actual upstream expressions, never persist or reimplement their values."""
    tree = ast.parse(text)
    assignments = [n for n in tree.body if isinstance(n, (ast.Assign, ast.AnnAssign))]
    names = []
    for n in assignments:
        target = n.target if isinstance(n, ast.AnnAssign) else n.targets[0] if len(n.targets) == 1 else None
        require(isinstance(target, ast.Name), 'unsupported config assignment target')
        names.append(target.id)
    require(tuple(names) == CONFIG_NAMES, 'unsupported/missing/duplicate config assignments')
    allowed = (ast.Constant, ast.Name, ast.Load, ast.Dict, ast.List, ast.Tuple,
               ast.Call, ast.keyword, ast.Attribute, ast.BinOp, ast.BitOr, ast.UnaryOp, ast.USub)
    symbols = set(CONFIG_NAMES) | {'dict', 'get_parser_args', 'NonLinearity', 'OptimizerName'}
    for index, n in enumerate(assignments):
        for x in ast.walk(n.value):
            require(isinstance(x, allowed), 'unsupported config expression')
            if isinstance(x, ast.Name):
                require(x.id in symbols and (x.id not in CONFIG_NAMES or x.id in CONFIG_NAMES[:index]),
                        'unsupported config symbol/dependency')
            if isinstance(x, ast.Attribute):
                require(not x.attr.startswith('_'), 'private config attribute')
                root = x.value
                chain = [x.attr]
                while isinstance(root, ast.Attribute):
                    chain.append(root.attr)
                    root = root.value
                require(isinstance(root, ast.Name) and
                        ((root.id == 'args' and len(chain) == 1) or
                         (root.id in ('NonLinearity', 'OptimizerName') and len(chain) <= 2 and
                          (len(chain) == 1 or chain[0] == 'name'))), 'unsupported config attribute')
            if isinstance(x, ast.Call):
                require(isinstance(x.func, ast.Name) and
                        ((index == 0 and x.func.id == 'get_parser_args' and not x.args and not x.keywords) or
                         (index > 0 and x.func.id == 'dict' and not x.args)), 'unsupported config call')
    require(isinstance(assignments[0].value, ast.Call), 'parser call missing')
    return assignments


def npy_identity(data, require_f64=False):
    require(data[:6] == b'\x93NUMPY' and data[6:8] in (b'\x01\x00', b'\x02\x00', b'\x03\x00'), 'invalid NPY magic/version')
    width = 2 if data[6] == 1 else 4
    require(len(data) >= 8 + width, 'truncated NPY')
    size = int.from_bytes(data[8:8 + width], 'little')
    require(0 < size <= 65536 and len(data) >= 8 + width + size, 'invalid NPY header length')
    try:
        header = ast.literal_eval(data[8 + width:8 + width + size].decode('latin1'))
    except (SyntaxError, ValueError, UnicodeError) as e:
        raise ExportError('invalid NPY header') from e
    require(isinstance(header, dict) and set(header) == {'descr', 'fortran_order', 'shape'}, 'unsupported NPY header')
    dtype, shape, fortran = header['descr'], header['shape'], header['fortran_order']
    require(dtype in ('<f8', '<f4', '|u1') and isinstance(shape, tuple) and
            1 <= len(shape) <= 2 and all(type(s) is int and s > 0 for s in shape) and
            type(fortran) is bool, 'unsupported NPY dtype/shape/orientation')
    require(not require_f64 or dtype == '<f8', 'weight must be little-endian F64')
    payload = data[8 + width + size:]
    require(len(payload) == math.prod(shape) * {'<f8': 8, '<f4': 4, '|u1': 1}[dtype], 'NPY payload length')
    if dtype != '|u1':
        require(all(math.isfinite(v[0]) for v in struct.iter_unpack('<d' if dtype == '<f8' else '<f', payload)),
                'nonfinite NPY value')
    return {'dtype': dtype, 'shape': list(shape), 'fortran_order': fortran, 'payload_sha256': digest(payload)}


def rate_bits(rates):
    require(isinstance(rates, list) and len(rates) == 1000 and
            all(type(x) in (float, int) and math.isfinite(x) and x >= 0 for x in rates), 'unsupported rates')
    return b''.join(struct.pack('<d', x) for x in rates)


def order_bytes(model, argv, epochs):
    require(len(epochs) == 10, 'order must have ten passes')
    lines = [f"# ReBoot {model} full setting (precision_analysis.py {' '.join(argv)}): MNIST training-set indices per step\n",
             '# step<TAB>epoch<TAB>batch<TAB>48 indices (0-based IDX order; step/epoch/batch 1-based)\n']
    for e, indices in enumerate(epochs):
        require(len(indices) == 4800 and sorted(indices) == list(range(4800)), 'order is not a permutation')
        for b in range(100):
            lines.append(f'{e * 100 + b + 1}\t{e + 1}\t{b + 1}\t' +
                         ' '.join(str(i) for i in indices[b * 48:(b + 1) * 48]) + '\n')
    return ''.join(lines).encode()


def order_indices(data):
    try:
        lines = data.decode('ascii').splitlines()
        require(len(lines) == 1002 and all(x.startswith('#') for x in lines[:2]), 'order rows/header')
        indices = []
        for i, line in enumerate(lines[2:]):
            cols = line.split('\t')
            require(len(cols) == 4 and cols[:3] == [str(i + 1), str(i // 100 + 1), str(i % 100 + 1)], 'order step/epoch/batch')
            values = cols[3].split(' ')
            require(len(values) == 48 and all(re.fullmatch(r'0|[1-9][0-9]*', v) for v in values), 'order indices encoding')
            indices.extend(map(int, values))
        for e in range(10):
            require(sorted(indices[e * 4800:(e + 1) * 4800]) == list(range(4800)), 'order pass is not permutation')
        return indices
    except (UnicodeError, ValueError) as e:
        raise ExportError('invalid order encoding') from e


def exact_json(a, b):
    # JSON's numeric equality loses -0.0 and bool/int distinctions. Compare encoded forms.
    return json_bytes(a) == json_bytes(b)


def compare_arrays(actual, expected, prefix):
    categories = []
    if actual != expected:
        categories.append(prefix + '_bytes')
    if npy_identity(actual) != npy_identity(expected):
        categories.append(prefix + '_decoded')
    return categories


def compare_capture(model, seed, captured, historical, archive, canonical):
    """Complete required historical projection; no tolerance or serialization repair."""
    old = historical['models'][model]
    entry, expected = captured['entry'], old['per_seed'][str(seed)]
    differences = []
    for key in ('config', 'structure'):
        if not exact_json(captured[key], old[key]):
            differences.append(key)
    for key in ('argv', 'indices_file', 'init_weights'):
        if not exact_json(entry[key], expected[key]):
            differences.append('weight_metadata' if key == 'init_weights' else key)
    if rate_bits(entry['lr_per_step']) != rate_bits(expected['lr_per_step']):
        differences.append('rate_bits')
    for w in expected['init_weights']:
        name = w['file']
        differences.extend(f'{name}:{c}' for c in compare_arrays(captured['artifacts'][name], (archive / name).read_bytes(), 'npy'))
    name = expected['indices_file']
    actual, prior = captured['artifacts'][name], (archive / name).read_bytes()
    if actual != prior:
        differences.append('order_bytes')
    if order_indices(actual) != order_indices(prior):
        differences.append('order_indices')
    for name in DATA_SHAPES:
        differences.extend(f'{name}:{c}' for c in compare_arrays(captured['data'][name], (canonical / name).read_bytes(), 'data'))
    return differences


GIT_CONFIG_ARGS = ('-c', 'core.fsmonitor=false', '-c', 'core.untrackedCache=false',
                   '-c', 'core.hooksPath=/dev/null', '-c', 'submodule.recurse=false')
GIT_TAILS = (('config', '--null', '--list', '--includes'), ('rev-parse', 'HEAD'),
             ('status', '--porcelain=v1', '--untracked-files=all', '--ignore-submodules=all'),
             ('diff', '--binary', '--no-ext-diff', '--no-textconv', '--no-renames', '--ignore-submodules=all', 'HEAD', '--'))
EXPORTER_REL = 'tools/export_reboot_block_seed_inputs.py'
GIT_ENV = {'PATH': '/usr/bin:/bin', 'LC_ALL': 'C', 'GIT_CONFIG_NOSYSTEM': '1',
           'GIT_CONFIG_GLOBAL': '/dev/null', 'GIT_CONFIG_SYSTEM': '/dev/null',
           'GIT_ATTR_NOSYSTEM': '1', 'GIT_OPTIONAL_LOCKS': '0', 'GIT_TERMINAL_PROMPT': '0'}


def git_commands(root, tool, exporter_only=False):
    prefix = [tool['executable'], *GIT_CONFIG_ARGS, '-C', str(Path(root).resolve())]
    tails = [list(t) for t in GIT_TAILS]
    if exporter_only:
        tails[2].extend(['--', EXPORTER_REL])
        tails[3].append(EXPORTER_REL)
    return [prefix + t for t in tails]


def git_identity(root, tool, exporter_only=False):
    require(set(tool) == {'executable', 'sha256'} and Path(tool['executable']).is_absolute() and
            str(Path(tool['executable']).resolve()) == tool['executable'] and
            file_hash(tool['executable']) == tool['sha256'], 'unapproved Git executable')
    commands = git_commands(root, tool, exporter_only)
    def invoke(command):
        return subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                              env=dict(GIT_ENV)).stdout
    config = invoke(commands[0])
    for item in config.split(b'\0'):
        if not item:
            continue
        key = item.split(b'\n', 1)[0].decode('ascii').lower()
        require(not key.startswith(('filter.', 'include.', 'includeif.')) and key != 'core.worktree',
                'unsafe Git configuration: filters/includes/worktree redirection prohibited')
    outputs = [invoke(command) for command in commands[1:]]
    return {'head': outputs[0].decode().strip(), 'status': outputs[1].decode(),
            'status_scope': EXPORTER_REL if exporter_only else 'repository',
            'config_sha256': digest(config), 'diff_sha256': digest(outputs[2])}


def inventory(root, paths):
    return {str(p.relative_to(root)): file_hash(p) for p in sorted(set(paths))}


def optional_host_query_policy():
    return {'policy': OPTIONAL_HOST_QUERY_POLICY,
            'source_sha256': NUMPY_UTILS_SHA256,
            'caller': 'numpy.testing._private.utils.check_support_sve',
            'event': 'subprocess.Popen', 'executable': 'lscpu', 'argv': ['lscpu'],
            'cwd': None, 'env': None, 'PATH': '/usr/bin:/bin',
            'platform_rules': [
                {'sys_platform': 'darwin', 'absent_paths': ['/usr/bin/lscpu', '/bin/lscpu']},
                {'sys_platform': 'linux', 'machine': 'x86_64',
                 'candidate_presence': 'irrelevant_to_denial'}],
            'exception': 'PermissionError(EPERM)'}


def runtime_identity():
    """Hash actual installed distribution files, not merely RECORD's claimed hashes."""
    packages = {}
    for d in importlib.metadata.distributions():
        name = d.metadata['Name'].lower().replace('_', '-')
        require(name not in packages and d.files is not None, 'duplicate/unrecorded installed distribution')
        files = {}
        for relative in sorted(d.files, key=str):
            if str(relative).endswith(('.pyc', '.pyo')):
                continue  # governed source is compiled directly; cache deserialization is denied
            path = Path(d.locate_file(relative))
            require(path.is_file(), f'missing installed package file: {name}/{relative}')
            files[str(relative)] = file_hash(path.resolve())
        packages[name] = {'version': d.version, 'root': str(Path(d.locate_file('')).resolve()), 'files': files}
    required = {'numpy', 'torch', 'torchvision', 'pybind11', 'pyyaml', 'scikit-learn',
                'imbalanced-learn', 'ucimlrepo', 'wandb', 'pandas', 'matplotlib', 'seaborn'}
    require(required <= packages.keys(), 'missing runtime distribution prerequisite')
    executable = str(Path(sys.executable).resolve())
    return {'executable': executable, 'executable_sha256': file_hash(executable),
            'version': sys.version, 'prefix': sys.prefix, 'base_prefix': sys.base_prefix,
            'platform': platform.platform(), 'sys_platform': sys.platform,
            'machine': platform.machine(), 'byteorder': sys.byteorder,
            'isolated': sys.flags.isolated, 'dont_write_bytecode': sys.dont_write_bytecode,
            'python_source_loading_policy': PYTHON_SOURCE_POLICY,
            'optional_host_query_denial_policy': optional_host_query_policy(),
            'environment': {k: os.environ.get(k) for k in
                            ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
                             'PYTHONHASHSEED', 'REBOOT_DATA', 'LD_LIBRARY_PATH', 'DYLD_LIBRARY_PATH', 'PATH')},
            'packages': packages}


def observe_context(paths):
    baseline = Path(paths['baseline'])
    archive, canonical = Path(paths['archive']), Path(paths['canonical'])
    source_paths = list(baseline.rglob('*.py')) + [baseline / EXP / n for n in
                    ('experiments.sh', 'precision_analysis.py', 'training_encrypted.py')]
    source_paths += [baseline / 'cpp/CMakeLists.txt']
    require(len(source_paths) >= 10 and (baseline / 'lib/parser.py') in source_paths, 'incomplete baseline source')
    raw = {n: file_hash(Path(paths['data']) / 'MNIST/raw' / n) for n in RAW_PINS}
    require(raw == RAW_PINS, 'raw MNIST pins mismatch')
    require(file_hash(archive / 'config_values.json') == ARCHIVE_SHA256, 'historical JSON pin mismatch')
    historical = strict_json((archive / 'config_values.json').read_bytes())
    old_files = {'config_values.json': ARCHIVE_SHA256}
    for model in MODELS:
        for seed in range(3):
            entry = historical['models'][model]['per_seed'][str(seed)]
            for weight in entry['init_weights']:
                rel = weight['file']
                require(rel == f'init_weights_{model}_seed{seed}/{weight["layer"]}.npy' and
                        weight['layer'] in SHAPES[model], 'unsafe historical array path')
                old_files[rel] = file_hash(archive / rel)
                require(old_files[rel] == weight['sha256'], 'historical array hash mismatch')
            rel = entry['indices_file']
            require(rel == f'train_indices_{model}_seed{seed}.txt', 'unsafe historical order path')
            old_files[rel] = file_hash(archive / rel)
    canonical_files = {n: file_hash(canonical / n) for n in DATA_SHAPES}
    require(canonical_files == DATA_PINS, 'canonical dataset pins mismatch')
    native = paths['native']
    require(set(native) == {'reboot', 'openfhe_python', 'openfhe_libraries'} and
            isinstance(native['openfhe_libraries'], list) and len(native['openfhe_libraries']) >= 3,
            'missing native prerequisite roles')
    native_paths = [native['reboot'], native['openfhe_python'], *native['openfhe_libraries']]
    require(len(native_paths) == len(set(native_paths)), 'duplicate native prerequisite')
    require(Path(native['reboot']).parent == baseline and 'reboot_cpp' in Path(native['reboot']).name,
            'foreign reboot native prerequisite')
    require(Path(native['openfhe_python']).name.startswith('openfhe'), 'foreign OpenFHE extension')
    exporter = Path(__file__).resolve()
    project = exporter.parent.parent
    return {'baseline': {'git': git_identity(baseline, paths['git']), 'files': inventory(baseline, source_paths)},
            'openfhe': git_identity(paths['openfhe'], paths['git']), 'openfhe_python': git_identity(paths['openfhe_python'], paths['git']),
            'exporter': {'path': str(exporter), 'sha256': file_hash(exporter), 'git': git_identity(project, paths['git'], exporter_only=True)},
            'runtime': runtime_identity(), 'git_tool': paths['git'], 'native': {n: file_hash(n) for n in native_paths},
            'raw_mnist': raw, 'historical': old_files, 'canonical': canonical_files}


def load_approval(path, expected_hash):
    require(re.fullmatch('[0-9a-f]{64}', expected_hash) is not None, 'invalid approval digest')
    raw = Path(path).read_bytes()
    require(digest(raw) == expected_hash, 'approval digest mismatch')
    approval = strict_json(raw)
    require(set(approval) == {'schema', 'approval_id', 'paths', 'expected_context', 'authorized_mode'}, 'unsupported approval fields')
    require(approval['schema'] == 'hekit-block-seed-export-context-v1' and
            isinstance(approval['approval_id'], str) and approval['approval_id'].strip(), 'invalid approval identity')
    require(set(approval['paths']) == {'baseline', 'data', 'archive', 'canonical', 'openfhe', 'openfhe_python', 'native', 'git'},
            'missing/extra approval paths')
    for key, value in approval['paths'].items():
        if key not in ('native', 'git'):
            require(isinstance(value, str) and Path(value).is_absolute() and str(Path(value).resolve()) == value,
                    'approval roots must be canonical absolute paths')
    require(approval['authorized_mode'] in ('compatibility', 'complete'), 'unsupported approved execution mode')
    context = approval['expected_context']
    require(isinstance(context, dict) and set(context) ==
            {'baseline', 'openfhe', 'openfhe_python', 'exporter', 'runtime', 'git_tool', 'native', 'raw_mnist', 'historical', 'canonical'},
            'incomplete approved context')
    approval['approval_sha256'] = expected_hash
    return approval


def validate_capture(model, seed, capture, historical):
    require(set(capture) == {'config', 'structure', 'entry', 'artifacts', 'data', 'observed_config'}, 'unsupported capture fields')
    entry = capture['entry']
    require(set(entry) == {'argv', 'indices_file', 'init_weights', 'lr_per_step'}, 'unsupported/training seed fields')
    require(entry['argv'] == released_argv(model, seed), 'unsupported captured argv')
    # Config/structure are seed-independent and every seed must satisfy the complete
    # released graph, optimizer and scheduler metadata, not merely seeds0–2.
    require(exact_json(capture['config'], historical['models'][model]['config']), 'unsupported model config')
    require(exact_json(capture['structure'], historical['models'][model]['structure']), 'unsupported model structure')
    require(capture['structure']['trainable_layer_order'] == list(SHAPES[model]), 'unsupported layer order')
    expected_files = {f'train_indices_{model}_seed{seed}.txt'}
    require(entry['indices_file'] in expected_files, 'unsupported order basename')
    order_indices(capture['artifacts'][entry['indices_file']])
    require([w['layer'] for w in entry['init_weights']] == list(SHAPES[model]), 'unsupported weight order/names')
    for w in entry['init_weights']:
        require(set(w) == {'file', 'layer', 'shape', 'orientation', 'dtype_in_reboot', 'sha256'}, 'unsupported weight metadata')
        rel = f'init_weights_{model}_seed{seed}/{w["layer"]}.npy'
        require(w['file'] == rel and w['shape'] == list(SHAPES[model][w['layer']]) and
                w['orientation'] == 'in x out' and w['dtype_in_reboot'] == 'float32', 'unsupported weight shape/orientation/dtype')
        data = capture['artifacts'][rel]
        identity = npy_identity(data, require_f64=True)
        require(identity['shape'] == w['shape'] and not identity['fortran_order'] and digest(data) == w['sha256'],
                'unsupported weight array/hash')
        expected_files.add(rel)
    require(set(capture['artifacts']) == expected_files, 'extra/missing capture files')
    rate_bits(entry['lr_per_step'])
    # All ten source schedules must be byte-identical to the validated released
    # schedule; seeds change initialization/order only.
    require(rate_bits(entry['lr_per_step']) == rate_bits(historical['models'][model]['per_seed']['0']['lr_per_step']),
            'unsupported scheduler values')
    require(set(capture['data']) == set(DATA_SHAPES), 'missing/extra canonical arrays')
    for name, (dtype, shape) in DATA_SHAPES.items():
        identity = npy_identity(capture['data'][name])
        require(identity['dtype'] == dtype and identity['shape'] == list(shape) and not identity['fortran_order'],
                'unsupported canonical data shape/dtype/orientation')


def artifact_manifest(out):
    return {str(p.relative_to(out)): file_hash(p) for p in sorted(out.rglob('*')) if p.is_file() and
            str(p.relative_to(out)) not in ('status.json', 'completion_manifest.json')}


def run_export(approval, out, mode, *, backend_factory=None, observer=observe_context):
    """Single-process gate; injection exists only for stdlib unit tests, never CLI."""
    require(mode in ('compatibility', 'complete'), 'unsupported mode')
    require(mode == approval['authorized_mode'], 'execution mode is not approved')
    out = Path(out).absolute()
    require(out == out.resolve(), 'output path contains symlink/traversal')
    require(not out.exists() and not out.is_symlink(), 'output collision')
    # Output is additive and outside all bound input trees. Exporter Git state is
    # scoped to its source path, allowing a new host-neutral capture directory.
    for key in ('baseline', 'data', 'archive', 'canonical', 'openfhe', 'openfhe_python'):
        require(not out.is_relative_to(Path(approval['paths'][key])), 'output overlaps bound input tree')
    before = observer(approval['paths'])
    require(exact_json(before, approval['expected_context']), 'unapproved/missing/changed context')
    baseline = Path(approval['paths']['baseline'])
    commands = read_commands((baseline / EXP / 'experiments.sh').read_text())
    # Structural config validation is also before any baseline import.
    configuration_assignments((baseline / EXP / 'precision_analysis.py').read_text())
    archive, canonical = Path(approval['paths']['archive']), Path(approval['paths']['canonical'])
    historical = strict_json((archive / 'config_values.json').read_bytes())
    out.mkdir()  # exclusive: a concurrent creator is a failure, never reuse
    started = now()
    status = {'state': 'partial', 'capture_kind': 'value_only_inputs', 'training_execution': 'not_performed',
              'mode': mode, 'started_utc': started, 'exit_status': None}
    write_new(out / 'status.json', json_bytes(status))
    result = {'source': 'actual baseline APIs; value-only export', 'capture_kind': 'value_only_inputs',
              'training_execution': 'not_performed', 'training_results': 'unavailable_not_run',
              'models': {m: {'seeds_listed': [0, 1, 2], 'per_seed': {}} for m in MODELS}}
    comparisons, observed, data_identities, emitted = {}, {}, {}, {}
    try:
        backend = (backend_factory(approval['paths']) if backend_factory is not None
                   else ActualBackend(approval['paths'], before))
        phases = [range(3)] + ([range(3, 10)] if mode == 'complete' else [])
        for phase in phases:
            for model in MODELS:
                for seed in phase:
                    capture = backend.capture(model, seed, seed_argv(commands[model][0], seed))
                    validate_capture(model, seed, capture, historical)
                    differences = compare_capture(model, seed, capture, historical, archive, canonical) if seed < 3 else []
                    # All captures, including3–9, must retain canonical data exactly.
                    for name in DATA_SHAPES:
                        differences.extend(f'{name}:{c}' for c in compare_arrays(
                            capture['data'][name], (canonical / name).read_bytes(), 'data'))
                    key = f'{model}/seed{seed}'
                    comparisons[key] = sorted(set(differences))
                    for rel, content in capture['artifacts'].items():
                        dest = out / rel
                        dest.parent.mkdir(exist_ok=True)
                        write_new(dest, content)
                        emitted[rel] = digest(content)
                    require(not differences, f'compatibility mismatch {key}: {", ".join(sorted(set(differences)))}')
                    model_out = result['models'][model]
                    model_out['config'] = capture['config']
                    model_out['structure'] = capture['structure']
                    model_out['per_seed'][str(seed)] = capture['entry']
                    observed[key] = capture['observed_config']
                    data_identities[key] = {name: {'sha256': digest(data), **npy_identity(data)}
                                            for name, data in capture['data'].items()}
            current = observer(approval['paths'])
            require(exact_json(current, before), 'source/runtime/data context drift')
            require(artifact_manifest(out) == emitted, 'output drift before compatibility/phase completion')
            if len(comparisons) == 6:
                receipt = {'schema': 'hekit-block-seed-compatibility-v1', 'status': 'passed',
                           'reusable_as_authorization': False, 'capture_kind': 'value_only_inputs',
                           'training_execution': 'not_performed', 'exit_status': 0, 'approval_id': approval['approval_id'],
                           'approval_sha256': approval['approval_sha256'], 'context': before,
                           'historical_json_sha256': digest((archive / 'config_values.json').read_bytes()),
                           'comparisons': comparisons.copy(), 'data': data_identities.copy(),
                           'observed_configs': observed.copy(), 'sequence': SEQUENCE,
                           'started_utc': started, 'verified_utc': now(),
                           'artifacts': artifact_manifest(out)}
                receipt_bytes = json_bytes(receipt)
                write_new(out / 'compatibility_receipt.json', receipt_bytes)
                emitted['compatibility_receipt.json'] = digest(receipt_bytes)
        seeds = list(range(3 if mode == 'compatibility' else 10))
        validate_seeds(seeds, mode)
        for m in MODELS:
            require(list(result['models'][m]['per_seed']) == list(map(str, seeds)), 'partial model/seed family')
        result_bytes = json_bytes(result)
        write_new(out / 'config_values.json', result_bytes)
        emitted['config_values.json'] = digest(result_bytes)
        provenance = {'approval_id': approval['approval_id'], 'approval_sha256': approval['approval_sha256'],
                      'context_before': before, 'context_after': current, 'constructor_rng_sequence': SEQUENCE,
                      'upstream_argv': commands, 'official_seeds': [0, 1, 2],
                      'statistical_override_seeds': list(range(3, 10)) if mode == 'complete' else [],
                      'captured_seeds': seeds, 'observed_configs': observed, 'data': data_identities,
                      'capture_kind': 'value_only_inputs', 'training_execution': 'not_performed',
                      'started_utc': started, 'finished_utc': now(), 'exit_status': 0}
        provenance_bytes = json_bytes(provenance)
        write_new(out / 'provenance.json', provenance_bytes)
        emitted['provenance.json'] = digest(provenance_bytes)
        files = artifact_manifest(out)
        expected_count = (120 if mode == 'complete' else 36) + 3
        require(len(files) == expected_count and files == emitted, 'unbounded/missing/changed output artifact set')
        require(files == artifact_manifest(out), 'output changed during final hashing')
        status.update(state='complete', finished_utc=now(), exit_status=0)
        (out / 'status.json').write_bytes(json_bytes(status))
        # Last file: consumers require this AND complete status. No atomic/power-loss claim.
        write_new(out / 'completion_manifest.json', json_bytes({'schema': 'hekit-value-only-export-v1',
                  'mode': mode, 'captured_seeds': seeds, 'files': files,
                  'status_sha256': file_hash(out / 'status.json'), 'exit_status': 0}))
        return out
    except BaseException as e:
        status.update(state='failed_partial', finished_utc=now(), exit_status=1,
                      error=f'{type(e).__name__}: {e}', comparisons=comparisons)
        (out / 'status.json').write_bytes(json_bytes(status))
        raise


def install_source_loading_guard(paths, approved_context, optional_host_query=None):
    """Compile exact approved source bytes, ignoring every existing bytecode cache.

Install only in the exporter child, after matching approved context and before
any governed imports. -B prevents writes, not reads; it is not this guarantee.
Already-executed interpreter/stdlib bootstrap remains the interpreter trust
boundary. Previously imported governed code cannot be validated retroactively.
"""
    baseline = Path(paths['baseline']).resolve()
    roots = {baseline}
    source_hashes = {}

    def add_sources(root, files):
        for relative, expected_hash in files.items():
            if not relative.endswith('.py'):
                continue
            source = (root / relative).resolve()
            require(re.fullmatch('[0-9a-f]{64}', expected_hash) is not None,
                    'invalid approved Python source hash')
            require(source not in source_hashes or source_hashes[source] == expected_hash,
                    'conflicting approved Python source identities')
            source_hashes[source] = expected_hash

    add_sources(baseline, approved_context['baseline']['files'])
    for package in approved_context['runtime']['packages'].values():
        root = Path(package['root']).resolve()
        roots.add(root)
        add_sources(root, package['files'])
    require(source_hashes, 'missing approved Python source inventory')
    numpy = approved_context['runtime']['packages'].get('numpy')
    optional_origin = (Path(numpy['root']) / 'numpy/testing/_private/utils.py').resolve() if numpy else None
    optional_platform = approved_context['runtime'].get('sys_platform')
    optional_machine = approved_context['runtime'].get('machine')
    optional_approved = (optional_host_query is not None and optional_origin is not None and
                         source_hashes.get(optional_origin) == NUMPY_UTILS_SHA256 and
                         optional_platform == sys.platform and optional_machine == platform.machine() and
                         exact_json(approved_context['runtime'].get('optional_host_query_denial_policy'),
                                    optional_host_query_policy()))

    def governed(path):
        absolute = Path(path).absolute()
        canonical = absolute.resolve()
        return canonical in source_hashes or any(
            absolute.is_relative_to(root) or canonical.is_relative_to(root) for root in roots)

    for name, module in tuple(sys.modules.items()):
        filename = getattr(module, '__file__', None)
        require(filename is None or not governed(filename),
                'governed module already loaded before source guard: ' + name)

    def source_get_code(loader, fullname):
        filename = loader.get_filename(fullname)
        canonical = Path(filename).resolve()
        require(canonical.is_file(), 'Python source unavailable: ' + str(canonical))
        # Hash and compile the SAME byte buffer; no hash/open/execute race or
        # SourceFileLoader cache lookup, including valid timestamp/hash caches.
        source = canonical.read_bytes()
        if governed(filename):
            require(canonical in source_hashes and digest(source) == source_hashes[canonical],
                    'unapproved/changed governed Python source: ' + str(canonical))
        code = compile(source, filename, 'exec', dont_inherit=True, optimize=-1)
        if (optional_approved and canonical == optional_origin and
                fullname == 'numpy.testing._private.utils' and
                optional_platform == sys.platform and optional_machine == platform.machine()):
            # Retain the actual compiled object, never a filename/name assertion
            # or code-object equality (freshly compiled code can compare equal).
            candidates = [c for c in code.co_consts if isinstance(c, type(code)) and
                          c.co_name == 'check_support_sve']
            require(len(candidates) == 1, 'optional host-query source function identity missing')
            optional_host_query.bind_validated_code(candidates[0], canonical,
                                                    optional_platform, optional_machine)
        return code

    def deny_bytecode(event, args):
        if event in ('marshal.load', 'marshal.loads'):
            raise ExportError('bytecode deserialization prohibited by source-only policy')

    # Covers direct SourceFileLoader calls as well as normal PathFinder imports.
    # Sourceless/direct/zip bytecode cannot bypass the source loader via marshal.
    sys.addaudithook(deny_bytecode)
    importlib.machinery.SourceFileLoader.get_code = source_get_code


class ForbiddenTrainingImports(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'lib.utils.train' or fullname.startswith('lib.utils.train.'):
            raise ExportError('training import prohibited')
        return None


class OptionalHostQueryDenial:
    """Only source-loader-bound code may receive the optional EPERM denial."""
    def __init__(self):
        self._bindings = []
        # Trust the already-loaded stdlib, as with the source guard bootstrap.
        # A foreign frame anywhere in the call chain keeps strict ExportError.
        self._subprocess_codes = (subprocess.run.__code__, subprocess.Popen.__init__.__code__,
                                  subprocess.Popen._execute_child.__code__)

    def bind_validated_code(self, code, origin, sys_platform, machine):
        self._bindings.append((code, origin, sys_platform, machine))

    def deny_if_exact(self, args):
        executable, command, cwd, env = args
        if not (self._bindings and type(executable) is str and executable == 'lscpu' and
                type(command) in (list, tuple) and len(command) == 1 and
                type(command[0]) is str and command[0] == 'lscpu' and
                cwd is None and env is None and os.environ.get('PATH') == '/usr/bin:/bin'):
            return
        sys_platform, machine = sys.platform, platform.machine()
        if sys_platform == 'darwin':
            if any(os.path.lexists(p) for p in ('/usr/bin/lscpu', '/bin/lscpu')):
                return
            arm = 'darwin_absent'
        elif sys_platform == 'linux' and machine == 'x86_64':
            # Presence is irrelevant: raise at the audit boundary before launch.
            arm = 'linux_x86_64_denied'
        else:
            return
        frame = sys._getframe(2)  # skip this method and the exporter audit hook
        try:
            while frame is not None and any(frame.f_code is c for c in self._subprocess_codes):
                frame = frame.f_back
            module = sys.modules.get('numpy.testing._private.utils')
            for code, origin, reviewed_platform, reviewed_machine in self._bindings:
                if (sys_platform == reviewed_platform and machine == reviewed_machine and
                        frame is not None and frame.f_code is code and module is not None and
                        frame.f_globals is module.__dict__ and
                        frame.f_globals.get('__name__') == 'numpy.testing._private.utils' and
                        frame.f_globals.get('__file__') is not None and
                        Path(frame.f_globals['__file__']).resolve() == origin):
                    print(json.dumps({'decision': 'denied_optional_host_query',
                                      'policy': OPTIONAL_HOST_QUERY_POLICY,
                                      'source': str(origin), 'source_sha256': NUMPY_UTILS_SHA256,
                                      'caller': 'numpy.testing._private.utils.check_support_sve',
                                      'event': 'subprocess.Popen', 'argv': ['lscpu'],
                                      'sys_platform': sys_platform, 'machine': machine,
                                      'arm': arm, 'no_launch': True,
                                      'errno': errno.EPERM}, sort_keys=True), file=sys.stderr, flush=True)
                    raise PermissionError(errno.EPERM, 'policy denied optional NumPy host query: lscpu')
        finally:
            del frame


def install_process_guards(paths):
    """Permanent only in this standalone exporter process; no parent state changes.

Python networking/process launch is denied. torchvision's loader is separately
forced into download=False. This is not a hostile-native-code sandbox; approved
native source/binaries and host isolation remain part of root's runtime review.
"""
    require(sys.flags.isolated and sys.dont_write_bytecode, 'actual capture requires python -I -B')
    require('lib.utils.train' not in sys.modules, 'training module already imported')
    sys.meta_path.insert(0, ForbiddenTrainingImports())
    optional_host_query = OptionalHostQueryDenial()
    sys.addaudithook(process_audit_guard(paths, optional_host_query))
    return optional_host_query


def process_audit_guard(paths, optional_host_query=None):
    allowed = []
    for key in ('baseline', 'openfhe', 'openfhe_python'):
        allowed.extend(git_commands(paths[key], paths['git']))
    allowed.extend(git_commands(Path(__file__).resolve().parent.parent, paths['git'], exporter_only=True))
    def audit(event, args):
        if event.startswith('socket.') or event in ('os.system', 'os.fork', 'os.forkpty', 'os.posix_spawn') or event.startswith('os.exec'):
            raise ExportError('network/process operation prohibited: ' + event)
        if event == 'subprocess.Popen':
            if optional_host_query is not None:
                optional_host_query.deny_if_exact(args)
            executable, command, cwd, env = args
            require(executable == paths['git']['executable'] and isinstance(command, (list, tuple)) and
                    list(command) in allowed and cwd is None and env == GIT_ENV,
                    'subprocess operation prohibited')
    return audit


def config_constraints(c, model, seed):
    hidden = [64, 32] if model == 'emlp2' else [128, 64, 32]
    required = {'subsample': 4800, 'resize': 14, 'normalize': False, 'pad_data': False, 'pad_labels': True,
                'num_fc_layers': len(hidden) + 1, 'num_fc_hidden': hidden, 'non_linearity': 'POLY_RELU',
                'dropout_rate': 0.0, 'seed': seed, 'bias': False, 'debug': True, 'encrypted': True,
                'instantiate_last_block': False, 'subnet_trainable': True, 'output_non_linearity': False,
                'batch_size': 48, 'num_epochs': 10, 'lr': 0.0005, 'fwd_decay': 0.0, 'lrn_decay': 0.0,
                'momentum': 0.9, 'optimizer': 'ENCRYPTED_NESTEROV_SGD', 'wandb': True}
    require(set(c) == set(required) | {'dataset'}, 'unsupported baseline configuration keys')
    require(getattr(c['dataset'], 'name', c['dataset']) == 'MNIST', 'unsupported baseline dataset')
    for key, expected in required.items():
        require(exact_json(c[key], expected), 'unsupported baseline configuration: ' + key)


def loaded_library_paths():
    """Observe loaded image locations, without loading an additional FHE library."""
    if sys.platform == 'darwin':
        import ctypes
        process = ctypes.CDLL(None)
        count = process._dyld_image_count
        count.restype = ctypes.c_uint32
        name = process._dyld_get_image_name
        name.argtypes = [ctypes.c_uint32]
        name.restype = ctypes.c_char_p
        return {str(Path(name(i).decode()).resolve()) for i in range(count()) if name(i)}
    if sys.platform.startswith('linux'):
        paths = set()
        for line in Path('/proc/self/maps').read_text().splitlines():
            parts = line.split(None, 5)
            if len(parts) == 6 and parts[5].startswith('/'):
                path = re.sub(r'\\([0-7]{3})', lambda m: chr(int(m.group(1), 8)), parts[5])
                paths.add(str(Path(path).resolve()))
        return paths
    raise ExportError('unsupported host loaded-library observation')


def verify_loaded_native(native, loaded):
    expected = set(native['openfhe_libraries'])
    require(all(Path(p).is_absolute() and str(Path(p).resolve()) == p for p in expected),
            'native library pins require canonical absolute paths')
    observed = {p for p in loaded if Path(p).name.lower().startswith('libopenfhe')}
    require(observed == expected, 'loaded OpenFHE libraries differ from approved native paths')
    return {p: file_hash(p) for p in sorted(observed)}


def verify_baseline_module_origin(name, module, baseline):
    """Verify live origins after guarded source loading, including PEP420 parents.

Namespace directories execute no source buffer. Their exact singleton search
locations must correspond to the module's dotted name under the approved root.
Regular Python source hashes remain enforced by install_source_loading_guard.
"""
    prefix = f'baseline module {name}: '
    require(isinstance(name, str) and name.split('.')[0] == 'lib' and
            all(part.isidentifier() for part in name.split('.')), prefix + 'invalid name')
    require(sys.modules.get(name) is module, prefix + 'live registration mismatch')
    spec = getattr(module, '__spec__', None)
    require(isinstance(spec, importlib.machinery.ModuleSpec), prefix + 'missing/invalid spec')
    require(spec.name == name and getattr(module, '__name__', None) == name,
            prefix + 'module/spec name mismatch')
    baseline = Path(baseline).resolve()
    source = getattr(module, '__file__', None)
    if source is not None:
        require(not isinstance(spec.loader, importlib.machinery.NamespaceLoader),
                prefix + 'regular source has namespace loader')
        require(isinstance(source, str) and Path(source).is_file() and
                Path(source).resolve().is_relative_to(baseline), prefix + 'foreign/unavailable source')
        require(isinstance(spec.origin, str) and Path(spec.origin).resolve() == Path(source).resolve(),
                prefix + 'source/spec origin mismatch')
        return

    require(isinstance(spec.loader, importlib.machinery.NamespaceLoader) and
            getattr(module, '__loader__', None) is spec.loader, prefix + 'namespace loader mismatch')
    require(spec.origin is None, prefix + 'namespace spec origin must be absent')
    require(not spec.has_location, prefix + 'namespace spec has file location')
    require(getattr(module, '__package__', None) == name, prefix + 'namespace package name mismatch')
    expected = baseline.joinpath(*name.split('.'))
    require(expected.is_dir() and expected.resolve() == expected and
            expected.is_relative_to(baseline), prefix + 'namespace directory unavailable/noncanonical')
    for label, locations in (('namespace path', getattr(module, '__path__', None)),
                             ('namespace spec search locations', spec.submodule_search_locations)):
        require(locations is not None and not isinstance(locations, (str, bytes)),
                prefix + label + ' missing/invalid')
        try:
            entries = list(locations)
        except (TypeError, ValueError) as error:
            raise ExportError(prefix + label + ' not iterable') from error
        require(len(entries) == 1 and isinstance(entries[0], str), prefix + label + ' must be singleton')
        actual = Path(entries[0])
        require(actual.is_absolute() and actual == expected and actual.resolve() == expected,
                prefix + label + ' differs from exact canonical directory')


class ActualBackend:
    """Thin API adapter. No baseline algorithm implementation or training methods."""
    def __init__(self, paths, approved_context):
        optional_host_query = install_process_guards(paths)
        install_source_loading_guard(paths, approved_context, optional_host_query)
        self.paths = paths
        self.baseline = Path(paths['baseline'])
        require(not any(n == 'lib' or n.startswith('lib.') for n in sys.modules), 'foreign baseline already imported')
        sys.path.insert(0, str(self.baseline))
        self.np = importlib.import_module('numpy')
        tv = importlib.import_module('torchvision')
        original = tv.datasets.MNIST.__init__
        def offline_init(instance, root, train=True, transform=None, target_transform=None, download=False):
            require(Path(root).resolve() == Path(paths['data']), 'unexpected MNIST data root')
            # This controls the real upstream constructor; it never enters download().
            original(instance, root=root, train=train, transform=transform,
                     target_transform=target_transform, download=False)
        tv.datasets.MNIST.__init__ = offline_init
        def forbidden(*args, **kwargs):
            raise ExportError('context/key/encrypted/training/download path prohibited')
        tv.datasets.MNIST.download = forbidden
        crypto = importlib.import_module('lib.cryptocontext')
        crypto.CryptoContext.__init__ = forbidden
        self.model = importlib.import_module('lib.models.local_loss_models').LocalLossMLP
        self.scheduler = importlib.import_module('lib.optim.schedulers').CosineLR
        self.data = importlib.import_module('lib.utils.data')
        enums = importlib.import_module('lib.utils.enums')
        self.nonlinearity, self.optimizer = enums.NonLinearity, enums.OptimizerName
        self.parser = importlib.import_module('lib.parser').get_parser_args
        for name, module in tuple(sys.modules.items()):
            if name == 'lib' or name.startswith('lib.'):
                verify_baseline_module_origin(name, module, self.baseline)
        require(Path(importlib.import_module('reboot_cpp').__file__).resolve() == Path(paths['native']['reboot']),
                'foreign reboot native loaded')
        openfhe = importlib.import_module('openfhe')
        require(Path(openfhe.__file__).resolve() == Path(paths['native']['openfhe_python']), 'foreign OpenFHE native loaded')
        self.check_native()

    def check_native(self):
        return verify_loaded_native(self.paths['native'], loaded_library_paths())

    def read_config(self, argv):
        path = self.baseline / EXP / 'precision_analysis.py'
        nodes = configuration_assignments(path.read_text())
        namespace = {'__builtins__': {}, 'dict': dict, 'get_parser_args': self.parser,
                     'NonLinearity': self.nonlinearity, 'OptimizerName': self.optimizer}
        old_argv, old_cwd = sys.argv, os.getcwd()
        try:
            sys.argv = [str(path), *argv]
            os.chdir(path.parent)
            for name, node in zip(CONFIG_NAMES, nodes):
                expression = ast.Expression(node.value)
                namespace[name] = eval(compile(expression, str(path), 'eval'), namespace)
            return namespace['configs']
        finally:
            sys.argv = old_argv
            os.chdir(old_cwd)

    def serialize(self, array):
        stream = io.BytesIO()
        self.np.save(stream, array, allow_pickle=False)
        return stream.getvalue()

    def describe_layer(self, layer):
        d = {'name': layer.name, 'class': type(layer).__name__}
        for attr in ('in_features', 'out_features', 'propagate_backward', 'decay_rate', 'momentum', 'trainable'):
            if hasattr(layer, attr):
                value = getattr(layer, attr)
                require(type(value) in (bool, int, float, str, type(None)), 'unsupported layer property')
                d[attr] = value
        if getattr(layer, 'optimizer', None) is not None:
            d['optimizer'] = type(layer.optimizer).__name__
            if hasattr(layer.optimizer, 'weight_decay'):
                d['optimizer_weight_decay'] = float(layer.optimizer.weight_decay)
        if getattr(layer, 'weights', None) is not None:
            d['weight_shape'] = list(self.np.shape(layer.weights))
        return d

    def capture(self, model, seed, argv):
        require(argv == released_argv(model, seed), 'unsupported adapter argv')
        np = self.np
        c = self.read_config(argv)
        config_constraints(c, model, seed)
        np.random.seed(c['seed'])
        X, _, Xt, Y, _, Yt = self.data.load_float_dataset(c, show_images=False, show_log=False,
                                                       data_path=self.paths['data'])
        X = X.reshape(4800, -1)
        Xt = Xt.reshape(Xt.shape[0], -1)
        require(tuple(X.shape) == (4800, 196) and tuple(Xt.shape) == (10000, 196) and
                tuple(Y.shape) == (4800, 16) and tuple(Yt.shape) == (10000, 16), 'unsupported data dimensions')
        for labels in (Y, Yt):
            require(bool(np.all(labels[:, 10:] == 0)) and bool(np.all((labels[:, :10] == 0) | (labels[:, :10] == 1))) and
                    bool(np.all(np.sum(labels[:, :10], axis=1) == 1)), 'unsupported real/padded class labels')
        idx = np.arange(4800).reshape(-1, 1)
        perm1, _ = self.data.shuffle_dataset(idx, idx, seed=42)
        perm1 = perm1.reshape(-1)
        require(sorted(int(v) for v in perm1) == list(range(4800)), 'unsupported preprocessing permutation')

        def make(optimizer):
            return self.model(cc=None, num_fc_layers=c['num_fc_layers'], num_fc_hidden=c['num_fc_hidden'],
                              num_classes=Y.shape[1], input_dim=X.shape[1], dropout_rate=c['dropout_rate'],
                              non_linearity=self.nonlinearity[c['non_linearity']], fwd_decay_inv=c['fwd_decay'],
                              subnet_decay_inv=c['lrn_decay'], bias=c['bias'], debug=False,
                              optimizer=self.optimizer[optimizer], momentum=c['momentum'],
                              instantiate_last_block=c['instantiate_last_block'], subnet_trainable=c['subnet_trainable'],
                              output_non_linearity=c['output_non_linearity'], encrypted=False)
        np.random.seed(c['seed'])
        first_plain_model = make('PLAIN_NESTEROV_SGD')
        np.random.seed(c['seed'])
        captured_model = make(c['optimizer'])
        # Keep both objects alive, as in the original wrapper. Neither is executed.
        require(first_plain_model is not captured_model, 'constructor reused model')
        pos = np.arange(4800)
        rates, epochs = [], []
        sched = self.scheduler(base_lr=c['lr'], steps_before_restart=50, from_step=500)
        for _ in range(10):
            p, _ = self.data.shuffle_dataset(np.arange(4800).reshape(-1, 1), np.arange(4800).reshape(-1, 1))
            p = p.reshape(-1)
            require(sorted(int(v) for v in p) == list(range(4800)), 'unsupported epoch permutation')
            pos = pos[p]
            epoch = []
            for b in range(100):
                sched.step()
                rates.append(float(sched.get_lr()))
                epoch.extend(int(perm1[i]) for i in pos[b * 48:(b + 1) * 48])
            epochs.append(epoch)
        layers = captured_model.get_layers_with_parameters()
        structure = {'blocks': [{'block_name': block.name, 'class': type(block).__name__,
                                 'forward_layers': [self.describe_layer(l) for l in block.layers.layers],
                                 'local_head': [self.describe_layer(l) for l in block.pred_loss_net.layers]}
                                for block in captured_model.blocks],
                     'output_stack': [self.describe_layer(l) for l in captured_model.output_layers.layers],
                     'trainable_layer_order': [l.name for l in layers]}
        require(structure['trainable_layer_order'] == list(SHAPES[model]), 'unsupported captured layer names/order')
        config = {'argv': seed_argv(argv, 0), 'lr': c['lr'], 'momentum': c['momentum'],
                  'fwd_decay': c['fwd_decay'], 'lrn_decay': c['lrn_decay'], 'batch_size': c['batch_size'],
                  'epochs': c['num_epochs'], 'steps_per_epoch': 100, 'steps': len(rates), 'subsample_K': c['subsample'],
                  'input_resize': c['resize'], 'input_features': int(X.shape[1]), 'output_width': int(Y.shape[1]),
                  'hidden': c['num_fc_hidden'], 'num_fc_layers': c['num_fc_layers'],
                  'instantiate_last_block': c['instantiate_last_block'], 'bias': c['bias'], 'optimizer': c['optimizer'],
                  'activation': 'x^2 + x', 'lr_first': rates[0], 'lr_last': rates[-1], 'lr_schedule': SCHEDULE}
        order_name = f'train_indices_{model}_seed{seed}.txt'
        artifacts = {order_name: order_bytes(model, argv, epochs)}
        weights = []
        for layer in layers:
            w = np.asarray(layer.weights)
            require(tuple(w.shape) == SHAPES[model][layer.name] and str(w.dtype) == 'float32', 'unsupported source weights')
            rel = f'init_weights_{model}_seed{seed}/{layer.name}.npy'
            raw = self.serialize(w.astype(np.float64))
            artifacts[rel] = raw
            weights.append({'file': rel, 'layer': layer.name, 'shape': list(w.shape), 'orientation': 'in x out',
                            'dtype_in_reboot': str(w.dtype), 'sha256': digest(raw)})
        canonical_x = np.empty_like(X)
        canonical_x[perm1] = X
        canonical_y = np.empty(4800, dtype=np.uint8)
        canonical_y[perm1] = np.argmax(Y[:, :10], axis=1).astype(np.uint8)
        arrays = {'train_images_14x14.npy': canonical_x, 'train_labels.npy': canonical_y,
                  'test_images_14x14.npy': Xt, 'test_labels.npy': np.argmax(Yt[:, :10], axis=1).astype(np.uint8)}
        self.check_native()
        observed_config = {k: getattr(v, 'name', v) for k, v in c.items()}
        return {'config': config, 'structure': structure, 'observed_config': observed_config,
                'entry': {'argv': argv, 'indices_file': order_name, 'init_weights': weights, 'lr_per_step': rates},
                'artifacts': artifacts, 'data': {name: self.serialize(a) for name, a in arrays.items()}}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--approval', required=True, type=Path)
    parser.add_argument('--approval-sha256', required=True)
    parser.add_argument('--mode', choices=('compatibility', 'complete'), required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        approval = load_approval(args.approval, args.approval_sha256)
        run_export(approval, args.out, args.mode)
        return 0
    except (ExportError, OSError, ValueError, KeyError, TypeError, subprocess.SubprocessError) as e:
        print(f'export failed closed: {type(e).__name__}: {e}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
