"""Original synthetic fixtures; never import any external scientific runtime."""
import ast
import copy
import hashlib
import json
from pathlib import Path
import struct
import tempfile
import unittest
from unittest import mock

import export_reboot_block_seed_inputs as ex


def npy(shape=(2, 2), value=0.25, descr='<f8', extra_space=0):
    """Tiny independent test serializer, not a captured scientific input."""
    h = repr(dict(descr=descr, fortran_order=False, shape=shape))
    padding = (64 - ((10 + len(h) + 1) % 64)) % 64
    header = (h + ' ' * (padding + extra_space) + '\n').encode('ascii')
    count = 1
    for n in shape:
        count *= n
    code = {'<f8': 'd', '<f4': 'f', '|u1': 'B'}[descr]
    return b'\x93NUMPY\x01\x00' + struct.pack('<H', len(header)) + header + struct.pack('<' + code, value) * count


def argv(model='emlp2', seed=0):
    hidden = ['64', '32'] if model == 'emlp2' else ['128', '64', '32']
    return ['MNIST', '--subsample', '4800', '--num-layers', str(len(hidden) + 1),
            '--num-hidden', *hidden, '--batch-size', '48', '--num-epochs', '10', '--seed', str(seed)]


def commands():
    return '\n'.join('# MLP-' + m[-1] + '\n' + '\n'.join(
        'python3 precision_analysis.py ' + ' '.join(argv(m, s)) for s in range(3))
        for m in ('emlp2', 'emlp3'))


class GuardTests(unittest.TestCase):
    def test_seeds_reject_missing_duplicate_range(self):
        for bad in ([], [0, 1], [0, 1, 1], [-1, 0, 1], list(range(11)), [False, 1, 2]):
            with self.subTest(bad=bad), self.assertRaises(ex.ExportError):
                ex.validate_seeds(bad, 'compatibility')
        ex.validate_seeds([0, 1, 2], 'compatibility')
        ex.validate_seeds(list(range(10)), 'complete')

    def test_official_commands_and_only_seed_override(self):
        c = ex.read_commands(commands())
        self.assertEqual(ex.seed_argv(c['emlp3'][0], 9), argv('emlp3', 9))
        for text in (commands().replace('--seed 2', '--seed 1'),
                     commands().replace('--batch-size 48', '--batch-size 64', 1),
                     commands().replace('--seed 2', '--seed 2 --foo 1', 1),
                     commands().replace('precision_analysis.py', 'training.py', 1)):
            with self.assertRaises(ex.ExportError):
                ex.read_commands(text)

    def test_synthetic_ast_allowed_and_no_import_main_execution(self):
        # These invented expressions are unrelated to upstream configuration values.
        text = ('import forbidden\nargs = get_parser_args()\n'
                'dataset_config = dict(marker=args.marker)\n'
                'network_config = dict(flag=False)\n'
                'training_config = dict(opt=OptimizerName.EXAMPLE.name)\n'
                'configs: dict = dataset_config | network_config | training_config\n'
                'if __name__ == "__main__":\n    raise RuntimeError("must not execute")\n')
        nodes = ex.configuration_assignments(text)
        self.assertEqual(len(nodes), 5)
        for bad in (text.replace('args.marker', 'args.__dict__'),
                    text.replace('False', 'open("bad")'),
                    text + 'other = 1\n', text.replace('dataset_config | network_config', 'dataset_config["x"]')):
            with self.assertRaises(ex.ExportError):
                ex.configuration_assignments(bad)

    def test_npy_bit_identity_and_serialization_distinction(self):
        a, b = npy(), npy(extra_space=64)
        self.assertEqual(ex.npy_identity(a), ex.npy_identity(b))
        self.assertNotEqual(a, b)
        self.assertNotEqual(ex.npy_identity(npy(value=-0.0)), ex.npy_identity(npy(value=0.0)))
        for bad in (a[:-1], a + b'x', b'not numpy', npy(descr='<f4')):
            with self.assertRaises(ex.ExportError):
                ex.npy_identity(bad, require_f64=True)

    def test_rate_bits_are_exact_and_finite(self):
        self.assertNotEqual(ex.rate_bits([0.0] * 1000), ex.rate_bits([-0.0] * 1000))
        for bad in ([0.1] * 999, [float('nan')] * 1000, [-1.] * 1000, [True] * 1000):
            with self.assertRaises(ex.ExportError):
                ex.rate_bits(bad)

    def test_order_requires_complete_ten_permutations(self):
        a = ex.order_bytes('emlp2', argv(), [list(range(4800)) for _ in range(10)])
        self.assertEqual(len(ex.order_indices(a)), 48000)
        for bad in (a.replace(b'1\t1\t1\t', b'2\t1\t1\t', 1), a[:-20], a.replace(b'0 1 2 ', b'0 0 2 ', 1)):
            with self.assertRaises(ex.ExportError):
                ex.order_indices(bad)




SYNTHETIC_CONFIG = ('args = get_parser_args()\n'
                    'dataset_config = dict(marker=args.marker)\n'
                    'network_config = dict(flag=False)\n'
                    'training_config = dict(rate=0.125)\n'
                    'configs: dict = dataset_config | network_config | training_config\n')


def synthetic_capture(model, seed):
    # No scientific interpretation. Arrays are constants and orders identity.
    config = {'argv': argv(model, 0), 'synthetic_fixture': True}
    structure = {'trainable_layer_order': list(ex.SHAPES[model]), 'synthetic_fixture': True}
    order_name = f'train_indices_{model}_seed{seed}.txt'
    artifacts = {order_name: ex.order_bytes(model, argv(model, seed), [list(range(4800)) for _ in range(10)])}
    weights = []
    for name, shape in ex.SHAPES[model].items():
        rel = f'init_weights_{model}_seed{seed}/{name}.npy'
        raw = npy(shape, value=seed + 0.25)
        artifacts[rel] = raw
        weights.append({'file': rel, 'layer': name, 'shape': list(shape), 'orientation': 'in x out',
                        'dtype_in_reboot': 'float32', 'sha256': ex.digest(raw)})
    return {'config': config, 'structure': structure,
            'entry': {'argv': argv(model, seed), 'indices_file': order_name, 'init_weights': weights,
                      'lr_per_step': [0.125] * 1000}, 'artifacts': artifacts,
            'data': {name: npy(shape, value=0, descr=dtype) for name, (dtype, shape) in ex.DATA_SHAPES.items()},
            'observed_config': {'fixture_only': True, 'seed': seed}}


class ExportTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.paths = {}
        for key in ('baseline', 'data', 'archive', 'canonical', 'openfhe', 'openfhe_python'):
            p = self.root / key
            p.mkdir()
            self.paths[key] = str(p)
        exp = Path(self.paths['baseline']) / ex.EXP
        exp.mkdir(parents=True)
        (exp / 'experiments.sh').write_text(commands())
        (exp / 'precision_analysis.py').write_text(SYNTHETIC_CONFIG)
        self.historical = {'models': {m: {'per_seed': {}} for m in ex.MODELS}}
        for model in ex.MODELS:
            for seed in range(3):
                c = synthetic_capture(model, seed)
                self.historical['models'][model].update(config=c['config'], structure=c['structure'])
                self.historical['models'][model]['per_seed'][str(seed)] = c['entry']
                for name, raw in c['artifacts'].items():
                    p = Path(self.paths['archive']) / name
                    p.parent.mkdir(exist_ok=True)
                    p.write_bytes(raw)
        (Path(self.paths['archive']) / 'config_values.json').write_bytes(ex.json_bytes(self.historical))
        for name, raw in c['data'].items():
            (Path(self.paths['canonical']) / name).write_bytes(raw)
        self.approval = {'approval_id': 'SYNTHETIC-TEST-ONLY', 'approval_sha256': 'a' * 64,
                         'paths': self.paths, 'expected_context': {'synthetic': True}, 'authorized_mode': 'complete'}
        self.observer = mock.Mock(return_value={'synthetic': True})
        self.calls = []
        def capture(model, seed, actual_argv):
            self.calls.append((model, seed))
            self.assertEqual(actual_argv, argv(model, seed))
            return synthetic_capture(model, seed)
        self.backend = mock.Mock(capture=mock.Mock(side_effect=capture))
        self.factory = mock.Mock(return_value=self.backend)

    def run_export(self, mode='complete', out=None):
        approved = {**self.approval, 'authorized_mode': mode}
        return ex.run_export(approved, out or self.root / 'out', mode,
                             backend_factory=self.factory, observer=self.observer)

    def test_complete_all_twenty_actual_backend_calls_after_gate(self):
        out = self.run_export()
        self.assertEqual(self.calls[:6], [(m, s) for m in ex.MODELS for s in range(3)])
        self.assertEqual(len(self.calls), 20)
        self.assertEqual(self.observer.call_count, 3)
        result = ex.strict_json((out / 'config_values.json').read_bytes())
        self.assertEqual(result['capture_kind'], 'value_only_inputs')
        self.assertEqual(result['training_execution'], 'not_performed')
        for model in ex.MODELS:
            self.assertEqual(set(result['models'][model]['per_seed']), set(map(str, range(10))))
            self.assertEqual(result['models'][model]['seeds_listed'], [0, 1, 2])
        raw = (out / 'config_values.json').read_text()
        for forbidden in ('twin', 'accuracy', 'latency', 'test_acc'):
            self.assertNotIn(forbidden, raw)
        manifest = ex.strict_json((out / 'completion_manifest.json').read_bytes())
        self.assertEqual(len(manifest['files']), 123)
        for name, hash_ in manifest['files'].items():
            self.assertEqual(ex.file_hash(out / name), hash_)
        receipt = ex.strict_json((out / 'compatibility_receipt.json').read_bytes())
        self.assertFalse(receipt['reusable_as_authorization'])
        self.assertEqual(len(receipt['comparisons']), 6)

    def test_compatibility_only_and_no_receipt_resume_interface(self):
        out = self.run_export('compatibility')
        self.assertEqual(len(self.calls), 6)
        manifest = ex.strict_json((out / 'completion_manifest.json').read_bytes())
        self.assertEqual(len(manifest['files']), 39)
        for extra in ('--compatibility-receipt', '--seed', '--threshold'):
            with mock.patch('sys.stderr', new=__import__('io').StringIO()), self.assertRaises(SystemExit):
                ex.main(['--approval', 'x', '--approval-sha256', 'a' * 64, '--mode', 'complete', '--out', 'x', extra, 'bad'])

    def test_compatibility_approval_cannot_authorize_complete(self):
        approval = {**self.approval, 'authorized_mode': 'compatibility'}
        with self.assertRaisesRegex(ex.ExportError, 'mode is not approved'):
            ex.run_export(approval, self.root / 'out', 'complete', backend_factory=self.factory, observer=self.observer)
        self.observer.assert_not_called(); self.factory.assert_not_called()

    def test_collision_blocks_before_context_and_backend(self):
        target = self.root / 'out'
        target.mkdir()
        with self.assertRaisesRegex(ex.ExportError, 'collision'):
            self.run_export()
        self.observer.assert_not_called()
        self.factory.assert_not_called()

    def test_missing_changed_data_native_context_prevents_any_backend_call(self):
        for reason in ('raw MNIST pins mismatch', 'missing native prerequisite', 'source hash changed'):
            self.observer.side_effect = ex.ExportError(reason)
            with self.assertRaises(ex.ExportError):
                self.run_export()
            self.factory.assert_not_called()
            self.assertFalse((self.root / 'out').exists())
        self.observer.side_effect = None
        self.observer.return_value = {'synthetic': False}
        with self.assertRaisesRegex(ex.ExportError, 'unapproved'):
            self.run_export()
        self.factory.assert_not_called()

    def test_source_drift_after_six_blocks_overrides_and_leaves_partial(self):
        self.observer.side_effect = [{'synthetic': True}, {'synthetic': False}]
        with self.assertRaisesRegex(ex.ExportError, 'drift'):
            self.run_export()
        self.assertEqual(len(self.calls), 6)
        self.assertFalse((self.root / 'out/completion_manifest.json').exists())
        self.assertFalse((self.root / 'out/compatibility_receipt.json').exists())
        self.assertEqual(ex.strict_json((self.root / 'out/status.json').read_bytes())['state'], 'failed_partial')

    def test_partial_backend_exception_never_completes(self):
        self.backend.capture.side_effect = RuntimeError('synthetic constructor failure')
        with self.assertRaisesRegex(RuntimeError, 'constructor failure'):
            self.run_export()
        self.assertFalse((self.root / 'out/completion_manifest.json').exists())
        self.assertEqual(ex.strict_json((self.root / 'out/status.json').read_bytes())['exit_status'], 1)

    def test_byte_only_mismatch_stops_before_override_and_retains_failed_output(self):
        def bad(model, seed, actual_argv):
            self.calls.append((model, seed))
            c = synthetic_capture(model, seed)
            rel = c['entry']['indices_file']
            c['artifacts'][rel] = c['artifacts'][rel].replace(b'# ReBoot ', b'# Synthetic ', 1)
            return c
        self.backend.capture.side_effect = bad
        with self.assertRaisesRegex(ex.ExportError, 'order_bytes'):
            self.run_export()
        self.assertEqual(len(self.calls), 1)
        self.assertTrue((self.root / 'out/train_indices_emlp2_seed0.txt').exists())
        self.assertFalse((self.root / 'out/completion_manifest.json').exists())

    def test_unsupported_capture_shape_structure_order_rate_model(self):
        c = synthetic_capture('emlp2', 0)
        cases = []
        q = copy.deepcopy(c); q['entry']['argv'][0] = 'OTHER'; cases.append(q)
        q = copy.deepcopy(c); q['structure']['trainable_layer_order'].reverse(); cases.append(q)
        q = copy.deepcopy(c); q['entry']['init_weights'][0]['shape'] = [1, 1]; cases.append(q)
        q = copy.deepcopy(c); q['entry']['lr_per_step'][0] = 0.12500000000000003; cases.append(q)
        q = copy.deepcopy(c); q['entry']['twin'] = {}; cases.append(q)
        for q in cases:
            with self.assertRaises(ex.ExportError):
                ex.validate_capture('emlp2', 0, q, self.historical)

    def test_all_exact_comparison_categories(self):
        original = synthetic_capture('emlp2', 0)
        def compare(c):
            return ex.compare_capture('emlp2', 0, c, self.historical,
                                      Path(self.paths['archive']), Path(self.paths['canonical']))
        self.assertEqual(compare(original), [])
        for key in ('config', 'structure'):
            c = copy.deepcopy(original); c[key]['synthetic_fixture'] = False
            self.assertIn(key, compare(c))
        c = copy.deepcopy(original); c['entry']['argv'][-1] = '1'
        self.assertIn('argv', compare(c))
        c = copy.deepcopy(original); c['entry']['init_weights'][0]['orientation'] = 'out x in'
        self.assertIn('weight_metadata', compare(c))
        c = copy.deepcopy(original); c['entry']['lr_per_step'][0] += 1e-16
        self.assertIn('rate_bits', compare(c))
        weight = original['entry']['init_weights'][0]
        for raw, category in ((npy(tuple(weight['shape']), extra_space=64), 'npy_bytes'),
                              (npy(tuple(weight['shape']), value=0.5), 'npy_decoded')):
            c = copy.deepcopy(original); c['artifacts'][weight['file']] = raw
            self.assertTrue(any(category in item for item in compare(c)))
        c = copy.deepcopy(original)
        c['artifacts'][c['entry']['indices_file']] = ex.order_bytes('emlp2', argv(), [list(reversed(range(4800))) for _ in range(10)])
        self.assertIn('order_indices', compare(c)); self.assertIn('order_bytes', compare(c))
        for name, (dtype, shape) in ex.DATA_SHAPES.items():
            c = copy.deepcopy(original); c['data'][name] = npy(shape, value=1, descr=dtype)
            self.assertIn(name + ':data_bytes', compare(c)); self.assertIn(name + ':data_decoded', compare(c))

    def test_approval_rejects_missing_foreign_digest_and_fields(self):
        p = self.root / 'approval.json'
        for content, expected in ((b'{}', 'a' * 64), (b'{}', ex.digest(b'{}')),
                                  (b'{"schema":1,"schema":1}', ex.digest(b'{"schema":1,"schema":1}'))):
            p.write_bytes(content)
            with self.assertRaises(ex.ExportError):
                ex.load_approval(p, expected)

    def test_written_artifact_drift_is_not_blessed_by_final_manifest(self):
        checks = 0
        def observe(paths):
            nonlocal checks
            checks += 1
            if checks == 2:
                p = self.root / 'out/train_indices_emlp2_seed0.txt'
                p.write_bytes(p.read_bytes().replace(b'# ReBoot', b'# Corrupt', 1))
            return {'synthetic': True}
        self.observer.side_effect = observe
        with self.assertRaisesRegex(ex.ExportError, 'output'):
            self.run_export('compatibility')
        self.assertFalse((self.root / 'out/completion_manifest.json').exists())

    def test_output_symlink_parent_rejected_before_backend(self):
        real = self.root / 'real'
        real.mkdir()
        link = self.root / 'link'
        link.symlink_to(real, target_is_directory=True)
        with self.assertRaisesRegex(ex.ExportError, 'output'):
            self.run_export('compatibility', link / 'out')
        self.factory.assert_not_called()

    def test_training_import_finder_blocks_transitive_path(self):
        finder = ex.ForbiddenTrainingImports()
        for name in ('lib.utils.train', 'lib.utils.train.something'):
            with self.assertRaises(ex.ExportError):
                finder.find_spec(name)
        self.assertIsNone(finder.find_spec('lib.parser'))


class ProcessAndContextTests(unittest.TestCase):
    def test_hardened_git_observation_and_audit_share_exact_allowlist(self):
        paths = {k: '/synthetic/' + k for k in ('baseline', 'openfhe', 'openfhe_python')}
        paths['git'] = {'executable': '/synthetic/git', 'sha256': 'f' * 64}
        audit = ex.process_audit_guard(paths)
        calls = []
        def run(command, **kwargs):
            audit('subprocess.Popen', (command[0], command, None, kwargs['env']))
            calls.append((command, kwargs))
            return mock.Mock(stdout=b'core.fsmonitor\nfalse\0' if 'config' in command else b'fake\n')
        with mock.patch.object(ex, 'file_hash', return_value='f' * 64), mock.patch.object(ex.subprocess, 'run', side_effect=run):
            info = ex.git_identity(paths['baseline'], paths['git'])
            ex.git_identity(Path(ex.__file__).resolve().parent.parent, paths['git'], exporter_only=True)
        self.assertEqual(len(calls), 8)
        self.assertEqual(info['head'], 'fake')
        self.assertIn('--no-ext-diff', calls[3][0]); self.assertIn('--no-textconv', calls[3][0])
        self.assertEqual(calls[0][1]['env'], ex.GIT_ENV)
        good = ex.git_commands(paths['baseline'], paths['git'])[0]
        for executable, command, cwd, env in (
                ('/other/git', good, None, ex.GIT_ENV),
                (good[0], good + ['--evil'], None, ex.GIT_ENV),
                (good[0], good, '/foreign', ex.GIT_ENV),
                (good[0], good, None, {**ex.GIT_ENV, 'GIT_EXTERNAL_DIFF': 'evil'}),
                (good[0], good, None, None)):
            with self.assertRaises(ex.ExportError):
                audit('subprocess.Popen', (executable, command, cwd, env))
        for event in ('socket.connect', 'socket.__new__', 'os.system', 'os.posix_spawn', 'os.exec', 'os.fork'):
            with self.assertRaises(ex.ExportError):
                audit(event, ())
        audit('open', ('synthetic', 'r', 0))

    def test_git_filters_and_includes_rejected_before_worktree_commands(self):
        for setting in (b'filter.evil.clean\ncommand\0', b'filter.evil.process\ncommand\0',
                        b'include.path\n/foreign\0', b'includeif.onbranch:main.path\n/foreign\0'):
            with mock.patch.object(ex, 'file_hash', return_value='f' * 64), mock.patch.object(ex.subprocess, 'run', return_value=mock.Mock(stdout=setting)) as run:
                with self.assertRaisesRegex(ex.ExportError, 'Git configuration'):
                    ex.git_identity('/synthetic', {'executable': '/synthetic/git', 'sha256': 'f' * 64})
                self.assertEqual(run.call_count, 1)

    def test_loaded_native_paths_must_equal_approved_libraries(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp).resolve()
            expected = []
            for name in ('core', 'pke', 'binfhe'):
                p = root / f'libOPENFHE{name}.so'
                p.write_bytes(b'synthetic binary identity only')
                expected.append(str(p))
            native = {'openfhe_libraries': expected}
            hashes = ex.verify_loaded_native(native, set(expected) | {'/other/unrelated.so'})
            self.assertEqual(set(hashes), set(expected))
            for bad in (set(expected[:-1]), set(expected) | {'/foreign/libOPENFHEcore.so'}):
                with self.assertRaisesRegex(ex.ExportError, 'loaded OpenFHE'):
                    ex.verify_loaded_native(native, bad)

    def test_unapproved_git_bytes_fail_before_subprocess(self):
        with mock.patch.object(ex, 'file_hash', return_value='a' * 64), mock.patch.object(ex.subprocess, 'run') as run:
            with self.assertRaises(ex.ExportError):
                ex.git_identity('/synthetic', {'executable': '/synthetic/git', 'sha256': 'b' * 64})
            run.assert_not_called()

    def test_raw_data_missing_or_changed_fails_before_git_runtime_import(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            paths = {key: str(root / key) for key in ('baseline', 'data', 'archive', 'canonical')}
            lib = Path(paths['baseline']) / 'lib'
            lib.mkdir(parents=True)
            for i in range(8):
                (lib / f'fake{i}.py').write_text('# synthetic\n')
            (lib / 'parser.py').write_text('# synthetic\n')
            with mock.patch.object(ex, 'git_identity') as git, mock.patch.object(ex, 'runtime_identity') as runtime:
                with self.assertRaisesRegex(ex.ExportError, 'prerequisite'):
                    ex.observe_context(paths)
                raw = Path(paths['data']) / 'MNIST/raw'
                raw.mkdir(parents=True)
                for name in ex.RAW_PINS:
                    (raw / name).write_bytes(b'synthetic wrong data')
                with self.assertRaisesRegex(ex.ExportError, 'raw MNIST pins'):
                    ex.observe_context(paths)
                git.assert_not_called(); runtime.assert_not_called()

    def test_native_missing_fails_before_git_runtime_import(self):
        # File reader fakes expected pin facts, but absence of native roles cannot pass.
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            paths = {key: str(root / key) for key in ('baseline', 'data', 'archive', 'canonical')}
            lib = Path(paths['baseline']) / 'lib'
            lib.mkdir(parents=True)
            for i in range(8):
                (lib / f'fake{i}.py').write_text('# synthetic\n')
            (lib / 'parser.py').write_text('# synthetic\n')
            archive = Path(paths['archive']); archive.mkdir()
            old = {'models': {m: {'per_seed': {str(s): synthetic_capture(m, s)['entry'] for s in range(3)}} for m in ex.MODELS}}
            (archive / 'config_values.json').write_bytes(ex.json_bytes(old))
            paths['native'] = {}
            def hash_(path):
                p = Path(path)
                if p.name in ex.RAW_PINS: return ex.RAW_PINS[p.name]
                if p.name in ex.DATA_PINS: return ex.DATA_PINS[p.name]
                if p.name == 'config_values.json': return ex.ARCHIVE_SHA256
                if p.suffix == '.npy':
                    for model in ex.MODELS:
                        for seed in range(3):
                            for w in old['models'][model]['per_seed'][str(seed)]['init_weights']:
                                if p.as_posix().endswith(w['file']): return w['sha256']
                return 'a' * 64
            with mock.patch.object(ex, 'file_hash', side_effect=hash_), mock.patch.object(ex, 'git_identity') as git:
                with self.assertRaisesRegex(ex.ExportError, 'native prerequisite'):
                    ex.observe_context(paths)
                git.assert_not_called()


class FakeArray:
    """Shape-only fake for data; explicit integers only for index API calls."""
    def __init__(self, shape, dtype='float32', values=None):
        self.shape, self.dtype, self.values = tuple(shape), dtype, values
    def reshape(self, *shape):
        shape = list(shape)
        if -1 in shape:
            i = shape.index(-1); shape[i] = __import__('math').prod(self.shape) // __import__('math').prod(x for x in shape if x != -1)
        return FakeArray(shape, self.dtype, self.values)
    def __iter__(self):
        if self.values is None: raise AssertionError('no real data computation in fake')
        return iter(self.values)
    def __getitem__(self, index):
        if self.values is not None:
            if isinstance(index, FakeArray): return FakeArray((len(index.values),), self.dtype, [self.values[i] for i in index.values])
            if isinstance(index, slice): return self.values[index]
            return self.values[index]
        return FakeArray(self.shape, self.dtype)
    def __setitem__(self, index, value):
        pass
    def __eq__(self, other): return True
    def __or__(self, other): return True
    def astype(self, dtype): return FakeArray(self.shape, dtype, self.values)


class FakeNumpy:
    float64 = 'float64'
    uint8 = 'uint8'
    def __init__(self, events):
        self.random = mock.Mock(seed=lambda seed: events.append(('seed', seed)))
    def arange(self, n): return FakeArray((n,), 'int64', list(range(n)))
    def asarray(self, value): return value
    def shape(self, value): return value.shape
    def all(self, value): return value
    def sum(self, value, axis): return 1
    def empty_like(self, value): return FakeArray(value.shape, value.dtype)
    def empty(self, n, dtype): return FakeArray((n,), dtype)
    def argmax(self, value, axis): return FakeArray((value.shape[0],), 'int64')


def actual_config(model='emlp2', seed=0):
    return {'dataset': 'MNIST', 'subsample': 4800, 'resize': 14, 'normalize': False, 'pad_data': False, 'pad_labels': True,
            'num_fc_layers': 3, 'num_fc_hidden': [64, 32], 'non_linearity': 'POLY_RELU', 'dropout_rate': 0.0,
            'seed': seed, 'bias': False, 'debug': True, 'encrypted': True, 'instantiate_last_block': False,
            'subnet_trainable': True, 'output_non_linearity': False, 'batch_size': 48, 'num_epochs': 10,
            'lr': 0.0005, 'fwd_decay': 0.0, 'lrn_decay': 0.0, 'momentum': 0.9,
            'optimizer': 'ENCRYPTED_NESTEROV_SGD', 'wandb': True}


class AdapterTests(unittest.TestCase):
    def test_actual_adapter_calls_fake_apis_in_original_order_without_training(self):
        from types import SimpleNamespace as NS
        events = []
        backend = object.__new__(ex.ActualBackend)
        backend.np = FakeNumpy(events)
        backend.paths = {'data': '/synthetic/data'}
        backend.check_native = lambda: None
        backend.read_config = lambda args: actual_config()
        backend.nonlinearity = {'POLY_RELU': 'fake_activation'}
        backend.optimizer = {'PLAIN_NESTEROV_SGD': 'plain', 'ENCRYPTED_NESTEROV_SGD': 'captured'}
        def load(c, **kwargs):
            events.append(('load', kwargs))
            return (FakeArray((4800, 196)), None, FakeArray((10000, 196)),
                    FakeArray((4800, 16)), None, FakeArray((10000, 16)))
        def shuffle(x, y, seed=None):
            events.append(('shuffle', seed))
            return x, y
        backend.data = NS(load_float_dataset=load, shuffle_dataset=shuffle)
        def model(**kwargs):
            events.append(('model', kwargs))
            layers = [NS(name=n, weights=FakeArray(shape)) for n, shape in ex.SHAPES['emlp2'].items()]
            # No forward/update/train methods exist. Any accidental use fails the test.
            return NS(blocks=[], output_layers=NS(layers=layers), get_layers_with_parameters=lambda: layers)
        backend.model = model
        class Scheduler:
            def __init__(self, **kwargs): events.append(('scheduler', kwargs))
            def step(self): events.append(('step',))
            def get_lr(self): events.append(('get_lr',)); return 0.125
        backend.scheduler = Scheduler
        backend.serialize = lambda a: npy(a.shape, value=0, descr={'float32': '<f4', 'float64': '<f8', 'uint8': '|u1'}[a.dtype])
        result = backend.capture('emlp2', 0, argv())
        self.assertEqual([e[0] for e in events[:8]], ['seed', 'load', 'shuffle', 'seed', 'model', 'seed', 'model', 'scheduler'])
        self.assertEqual(events[2], ('shuffle', 42))
        for i, optimizer in ((4, 'plain'), (6, 'captured')):
            self.assertEqual(events[i][1]['optimizer'], optimizer)
            self.assertIsNone(events[i][1]['cc']); self.assertFalse(events[i][1]['encrypted'])
        self.assertEqual([e for e in events if e[0] == 'shuffle'], [('shuffle', 42)] + [('shuffle', None)] * 10)
        self.assertEqual([e[0] for e in events if e[0] in ('step', 'get_lr')], ['step', 'get_lr'] * 1000)
        self.assertEqual(events[7][1], {'base_lr': 0.0005, 'steps_before_restart': 50, 'from_step': 500})
        self.assertEqual(len(result['entry']['lr_per_step']), 1000)
        self.assertEqual(set(result['data']), set(ex.DATA_SHAPES))

    def test_configuration_executes_only_selected_actual_expressions_and_restores_state(self):
        from types import SimpleNamespace as NS
        with tempfile.TemporaryDirectory() as temp:
            backend = object.__new__(ex.ActualBackend)
            backend.baseline = Path(temp)
            exp = backend.baseline / ex.EXP; exp.mkdir(parents=True)
            (exp / 'precision_analysis.py').write_text('import definitely_not_a_real_module\n' + SYNTHETIC_CONFIG +
                                                       'if __name__ == "__main__":\n    raise RuntimeError("forbidden")\n')
            backend.parser = lambda: NS(marker='actual fake parser observation')
            backend.nonlinearity = backend.optimizer = None
            old_argv, old_cwd = ex.sys.argv, ex.os.getcwd()
            result = backend.read_config(argv())
            self.assertEqual(result['marker'], 'actual fake parser observation')
            self.assertIs(ex.sys.argv, old_argv); self.assertEqual(ex.os.getcwd(), old_cwd)
            backend.parser = lambda: (_ for _ in ()).throw(RuntimeError('fake parser failure'))
            with self.assertRaises(RuntimeError): backend.read_config(argv())
            self.assertIs(ex.sys.argv, old_argv); self.assertEqual(ex.os.getcwd(), old_cwd)

    def test_config_rejects_every_changed_released_field_and_missing_seed(self):
        c = actual_config()
        ex.config_constraints(c, 'emlp2', 0)
        for key in c:
            bad = dict(c); bad[key] = 'changed'
            with self.subTest(key=key), self.assertRaises(ex.ExportError):
                ex.config_constraints(bad, 'emlp2', 0)
        del c['seed']
        with self.assertRaises(ex.ExportError): ex.config_constraints(c, 'emlp2', 0)


class IsolatedAuditTests(unittest.TestCase):
    def test_actual_child_audit_hook_allows_hardened_readonly_git_and_denies_socket(self):
        import shutil
        import subprocess
        import sys
        tool = Path(shutil.which('git')).resolve()
        root = Path(ex.__file__).resolve().parent.parent
        paths = {k: str(root) for k in ('baseline', 'openfhe', 'openfhe_python')}
        paths['git'] = {'executable': str(tool), 'sha256': ex.file_hash(tool)}
        program = '''import json,sys
sys.path.insert(0, sys.argv[1])
import export_reboot_block_seed_inputs as ex
paths=json.loads(sys.argv[2])
ex.install_process_guards(paths)
identity=ex.git_identity(paths['baseline'],paths['git'],exporter_only=True)
assert len(identity['head']) == 40
import socket
try:
    socket.socket()
except ex.ExportError:
    pass
else:
    raise AssertionError('network guard missing')
assert not any(n in sys.modules for n in ('numpy','torch','openfhe','lib.utils.train'))
print('read-only provenance and socket denial passed')
'''
        result = subprocess.run([sys.executable, '-I', '-B', '-c', program, str(root / 'tools'), json.dumps(paths)],
                                capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('socket denial passed', result.stdout)


class SourceLoadingCacheTests(unittest.TestCase):
    def test_isolated_valid_caches_cannot_override_approved_source(self):
        import importlib.util
        import marshal
        import os
        import py_compile
        import zipfile
        import subprocess
        import sys
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp).resolve()
            baseline, packages = root / 'baseline', root / 'packages'
            baseline.mkdir(); packages.mkdir()
            old, approved = b"VALUE = 'cached'\n", b"VALUE = 'source'\n"
            stale, foreign = baseline / 'synthetic_stale.py', packages / 'synthetic_foreign.py'
            stale.write_bytes(old)
            timestamp = stale.stat()
            py_compile.compile(str(stale), doraise=True)
            stale.write_bytes(approved)
            os.utime(stale, ns=(timestamp.st_atime_ns, timestamp.st_mtime_ns))
            foreign.write_bytes(approved)
            foreign_cache = Path(importlib.util.cache_from_source(str(foreign)))
            foreign_cache.parent.mkdir(exist_ok=True)
            # A checked-hash cache carries the approved source hash but foreign code.
            foreign_cache.write_bytes(importlib.util.MAGIC_NUMBER + struct.pack('<I', 3) +
                                      importlib.util.source_hash(approved) +
                                      marshal.dumps(compile(old, str(foreign), 'exec')))
            zip_path = root / 'synthetic_cache.zip'
            with zipfile.ZipFile(zip_path, 'w') as z:
                z.writestr('synthetic_zip.pyc', foreign_cache.read_bytes())
            cache_before = {str(p): ex.file_hash(p) for p in root.rglob('*.pyc')}
            context = {'baseline': {'files': {'synthetic_stale.py': ex.digest(approved)}},
                       'runtime': {'packages': {'synthetic': {'root': str(packages),
                                   'files': {'synthetic_foreign.py': ex.digest(approved)}}}}}
            # Control child demonstrates the exact -I -B cache vulnerability.
            control = subprocess.run([sys.executable, '-I', '-B', '-c',
                "import sys;sys.path[:0]=sys.argv[1:];import synthetic_stale,synthetic_foreign;"
                "assert synthetic_stale.VALUE==synthetic_foreign.VALUE=='cached';print('cached control')",
                str(baseline), str(packages)], capture_output=True, text=True, timeout=30)
            self.assertEqual(control.returncode, 0, control.stderr)
            program = '''import json,sys,importlib,marshal,io
sys.path.insert(0,sys.argv[1])
import export_reboot_block_seed_inputs as ex
paths=json.loads(sys.argv[2]);context=json.loads(sys.argv[3])
sys.path[:0]=[paths['baseline'],context['runtime']['packages']['synthetic']['root']]
ex.install_source_loading_guard(paths,context)
import synthetic_stale,synthetic_foreign
assert synthetic_stale.VALUE==synthetic_foreign.VALUE=='source'
# A direct sourceless loader must not evade the normal source loader.
loader=importlib.machinery.SourcelessFileLoader('synthetic_direct',sys.argv[4])
try:
    loader.get_code('synthetic_direct')
except ex.ExportError:
    pass
else:
    raise AssertionError('sourceless bytecode executed')
payload=marshal.dumps(compile('VALUE=1','synthetic','exec'))
for read in (lambda:marshal.loads(payload),lambda:marshal.load(io.BytesIO(payload))):
    try:
        read()
    except ex.ExportError:
        pass
    else:
        raise AssertionError('direct marshal bypass')
sys.path.insert(0,sys.argv[5])
try:
    import synthetic_zip
except ex.ExportError:
    pass
else:
    raise AssertionError('zip bytecode bypass')
assert not any(n in sys.modules for n in ('numpy','torch','openfhe','lib'))
print('approved source; sourceless/direct bytecode denied')
'''
            guarded = subprocess.run([sys.executable, '-I', '-B', '-c', program,
                str(Path(ex.__file__).resolve().parent), json.dumps({'baseline': str(baseline)}),
                json.dumps(context), str(foreign_cache), str(zip_path)], capture_output=True, text=True, timeout=30)
            self.assertEqual(guarded.returncode, 0, guarded.stderr)
            self.assertIn('approved source', guarded.stdout)
            self.assertEqual(cache_before, {str(p): ex.file_hash(p) for p in root.rglob('*.pyc')})


    def test_isolated_guard_rejects_changed_unlisted_and_preloaded_sources(self):
        import subprocess
        import sys
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp).resolve()
            approved = b"VALUE = 'source'\n"
            context = {'baseline': {'files': {'synthetic_bound.py': ex.digest(approved)}},
                       'runtime': {'packages': {}}}
            bound, unlisted = root / 'synthetic_bound.py', root / 'synthetic_unlisted.py'
            bound.write_bytes(approved); unlisted.write_bytes(approved)
            program = '''import json,sys
from pathlib import Path
sys.path.insert(0,sys.argv[1])
import export_reboot_block_seed_inputs as ex
paths=json.loads(sys.argv[2]);context=json.loads(sys.argv[3]);scenario=sys.argv[4]
sys.path.insert(0,paths['baseline'])
if scenario=='preloaded':
    import synthetic_bound
    try:
        ex.install_source_loading_guard(paths,context)
    except ex.ExportError as error:
        assert 'already loaded' in str(error)
    else:
        raise AssertionError('preloaded governed code accepted')
else:
    ex.install_source_loading_guard(paths,context)
    if scenario=='changed':
        Path(paths['baseline'],'synthetic_bound.py').write_bytes(b"VALUE = 'change'" + bytes([10]))
        name='synthetic_bound'
    else:
        name='synthetic_unlisted'
    try:
        __import__(name)
    except ex.ExportError as error:
        assert 'unapproved/changed' in str(error)
    else:
        raise AssertionError('unapproved source executed')
    assert name not in sys.modules
assert not any(n in sys.modules for n in ('numpy','torch','openfhe','lib'))
print(scenario+' rejected')
'''
            for scenario in ('changed', 'unlisted', 'preloaded'):
                bound.write_bytes(approved)
                result = subprocess.run([sys.executable, '-I', '-B', '-c', program,
                    str(Path(ex.__file__).resolve().parent), json.dumps({'baseline': str(root)}),
                    json.dumps(context), scenario], capture_output=True, text=True, timeout=30)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn(scenario + ' rejected', result.stdout)
            self.assertFalse(list(root.rglob('*.pyc')))

    def test_actual_backend_installs_bound_source_guard_before_first_dependency(self):
        class StopBeforeDependency(Exception):
            pass
        events = []
        paths, context = {'baseline': '/synthetic/baseline'}, {'approved': 'context'}
        denial = object()
        def process(p):
            events.append(('process', p))
            return denial
        def source(p, c, state): events.append(('source', p, c, state))
        def imported(name):
            events.append(('import', name))
            raise StopBeforeDependency()
        with mock.patch.object(ex, 'install_process_guards', side_effect=process), \
             mock.patch.object(ex, 'install_source_loading_guard', side_effect=source), \
             mock.patch.object(ex.importlib, 'import_module', side_effect=imported), \
             mock.patch.object(ex.sys, 'path', list(ex.sys.path)):
            with self.assertRaises(StopBeforeDependency):
                ex.ActualBackend(paths, context)
        self.assertEqual(events, [('process', paths), ('source', paths, context, denial), ('import', 'numpy')])


# Fictional fixture: an audit emitter with arguments and optional exception handling.
# It contains no NumPy implementation or copied third-party source.
OPTIONAL_QUERY_FIXTURE = b"""import sys
EXECUTIONS = globals().get('EXECUTIONS', 0) + 1
def check_support_sve(event, arguments, catch=False):
    if catch:
        try:
            sys.audit(event, *arguments)
        except OSError:
            return False
        raise AssertionError('audit unexpectedly returned')
    sys.audit(event, *arguments)
def different_caller(event, arguments):
    sys.audit(event, *arguments)
"""


class OptionalHostQueryTests(unittest.TestCase):
    def test_runtime_policy_identity_rejects_old_context_before_backend(self):
        # Break caught: omitting new denial semantics from approval identity.
        from types import SimpleNamespace
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp).resolve()
            (root / 'fictional.py').write_text('# fictional metadata-only fixture\n')
            names = ('numpy', 'torch', 'torchvision', 'pybind11', 'pyyaml', 'scikit-learn',
                     'imbalanced-learn', 'ucimlrepo', 'wandb', 'pandas', 'matplotlib', 'seaborn')
            distributions = [SimpleNamespace(metadata={'Name': name}, version='fictional',
                files=['fictional.py'], locate_file=lambda rel: root / rel) for name in names]
            with mock.patch.object(ex.importlib.metadata, 'distributions', return_value=distributions), \
                 mock.patch.object(ex, 'file_hash', return_value='a' * 64), \
                 mock.patch.object(ex.platform, 'platform', return_value='fictional'), \
                 mock.patch.object(ex.platform, 'machine', return_value='fictional'), \
                 mock.patch.dict(ex.os.environ, {'PATH': '/usr/bin:/bin'}):
                runtime = ex.runtime_identity()
            self.assertIn('optional_host_query_denial_policy', runtime)
            policy = runtime['optional_host_query_denial_policy']
            self.assertEqual(policy['source_sha256'], '55d1a25fe7dbe7146deb16f0827833b207b88aadad6536654b4ec7020a559e40')
            self.assertEqual(policy['exception'], 'PermissionError(EPERM)')
            self.assertEqual(runtime['environment']['PATH'], '/usr/bin:/bin')
            old = copy.deepcopy(runtime)
            del old['optional_host_query_denial_policy']
            paths = {key: '/fictional/input' for key in
                     ('baseline', 'data', 'archive', 'canonical', 'openfhe', 'openfhe_python')}
            approval = {'paths': paths, 'authorized_mode': 'compatibility', 'expected_context': old}
            backend = mock.Mock()
            with self.assertRaisesRegex(ex.ExportError, 'unapproved/missing/changed context'):
                ex.run_export(approval, root / 'output', 'compatibility',
                              backend_factory=backend, observer=lambda paths: runtime)
            backend.assert_not_called()
            self.assertFalse((root / 'output').exists())

    def run_child(self, scenario):
        import subprocess
        import sys
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp).resolve()
            baseline, packages = root / 'baseline', root / 'packages'
            baseline.mkdir(); packages.mkdir()
            relative = 'numpy/testing/_private/utils.py'
            source = packages / relative
            source.parent.mkdir(parents=True)
            source.write_bytes(OPTIONAL_QUERY_FIXTURE)
            paths = {key: str(baseline) for key in ('baseline', 'openfhe', 'openfhe_python')}
            paths['git'] = {'executable': '/fictional/git', 'sha256': 'a' * 64}
            context = {'baseline': {'files': {}}, 'runtime': {'packages': {
                'numpy': {'root': str(packages), 'files': {relative: ex.digest(OPTIONAL_QUERY_FIXTURE)}}}}}
            program = r'''import errno,importlib.util,json,os,sys,types
from pathlib import Path
from unittest import mock
sys.path.insert(0,sys.argv[1])
import export_reboot_block_seed_inputs as ex
paths=json.loads(sys.argv[2]); context=json.loads(sys.argv[3]); scenario=sys.argv[4]
relative='numpy/testing/_private/utils.py'
source=Path(context['runtime']['packages']['numpy']['root'])/relative
fixture=source.read_bytes()
# Test-only substitution of the fixed hash for completely fictional source.
# Production has no CLI/context override of its reviewed pin.
ex.NUMPY_UTILS_SHA256=ex.digest(fixture)
if hasattr(ex,'optional_host_query_policy'):
    context['runtime']['optional_host_query_denial_policy']=ex.optional_host_query_policy()
if scenario=='old_context':
    context['runtime'].pop('optional_host_query_denial_policy',None)
if scenario=='wrong_pin':
    ex.NUMPY_UTILS_SHA256='f'*64
name='numpy.testing._private.utils'
spec=importlib.util.spec_from_file_location(name,source)
module=importlib.util.module_from_spec(spec)
if scenario=='preloaded':
    sys.modules[name]=module
state=ex.install_process_guards(paths)
try:
    if state is None:  # Also reproduces the original strict-denial bug for RED.
        ex.install_source_loading_guard(paths,context)
    else:
        ex.install_source_loading_guard(paths,context,state)
except ex.ExportError:
    assert scenario=='preloaded'
    print('preloaded rejected');sys.exit(0)
assert scenario!='preloaded', 'preloaded source accepted'
if scenario in ('changed_source','unlisted_source'):
    if scenario=='changed_source':
        source.write_bytes(fixture+b'CHANGED=True\n'.replace(b'\\n',b'\n'))
    else:
        source=source.with_name('unlisted.py');source.write_bytes(fixture)
        spec=importlib.util.spec_from_file_location(name,source)
    try:
        spec.loader.get_code(name)
    except ex.ExportError:
        print(scenario+' rejected');sys.exit(0)
    raise AssertionError('unapproved source accepted')
sys.modules[name]=module
code=spec.loader.get_code(name)
exec(code,module.__dict__)
assert module.EXECUTIONS==1
# Exact audit event only; no Popen or command invocation occurs in this child.
event='subprocess.Popen'; arguments=['lscpu',['lscpu'],None,None]
original=module.check_support_sve
call=original
if scenario=='extra_argument': arguments[1].append('--parse=core')
if scenario=='absolute_executable': arguments[0]='/usr/bin/lscpu'
if scenario=='absolute_argv': arguments[1]=['/usr/bin/lscpu']
if scenario=='string_argv': arguments[1]='lscpu'
if scenario=='bytes_executable': arguments[0]=b'lscpu'
if scenario=='cwd': arguments[2]='/tmp'
if scenario=='env': arguments[3]={'PATH':'/usr/bin:/bin'}
if scenario=='other_command': arguments[:2]=['uname',['uname']]
if scenario.startswith('operation:'): event=scenario.split(':',1)[1];arguments=[]
if scenario=='other_caller': call=module.different_caller
if scenario in ('dynamic_code','equal_code'):
    other={**module.__dict__}
    exec(compile(fixture,str(source),'exec'),other)
    call=other['check_support_sve']
    if scenario=='equal_code':
        assert call.__code__==original.__code__ and call.__code__ is not original.__code__
        # Keep every namespace/origin predicate true so this case tests only
        # object identity, not a coincidentally rejected globals dictionary.
        call=types.FunctionType(call.__code__,module.__dict__,argdefs=original.__defaults__)
if scenario=='spoofed_filename':
    other={'__name__':name,'__file__':str(source)}
    exec(compile(fixture,str(source),'exec'),other)
    call=other['check_support_sve']
if scenario=='spoofed_module':
    call=types.FunctionType(original.__code__,{**module.__dict__},argdefs=original.__defaults__)
if scenario=='wrong_module_name': module.__dict__['__name__']='fictional.other'
if scenario=='wrong_module_origin': module.__dict__['__file__']=str(source.with_name('other.py'))
if scenario=='intervening_caller':
    def extra(event,*arguments):
        return real_audit(event,*arguments)
    real_audit=sys.audit
    sys.audit=extra
# Only host predicates are mocked; loader, compiled identity, module namespace,
# audit registration/dispatch, exception handling, and diagnostics are real.
platform_value='linux' if scenario=='platform' else 'darwin'
path_value='/usr/local/bin:/usr/bin:/bin' if scenario=='path' else '/usr/bin:/bin'
real_lexists=os.path.lexists
present={'/usr/bin/lscpu'} if scenario=='present_usr_bin' else {'/bin/lscpu'} if scenario=='present_bin' else set()
def candidate_exists(path):
    if str(path) in ('/usr/bin/lscpu','/bin/lscpu'): return str(path) in present
    return real_lexists(path)
with mock.patch.object(sys,'platform',platform_value), mock.patch.dict(os.environ,{'PATH':path_value}), mock.patch.object(os.path,'lexists',side_effect=candidate_exists):
    if scenario=='fallback':
        assert call(event,arguments,True) is False
    elif scenario in ('exact','tuple_argv'):
        if scenario=='tuple_argv': arguments[1]=('lscpu',)
        try:
            call(event,arguments)
        except PermissionError as error:
            assert error.errno==errno.EPERM
        else:
            raise AssertionError('optional query not denied')
    else:
        try:
            call(event,arguments)
        except ex.ExportError:
            pass
        else:
            raise AssertionError('strict denial missing for '+scenario)
assert 'numpy' not in sys.modules and 'torch' not in sys.modules and 'openfhe' not in sys.modules
print(scenario+' passed; fictional source only; no command invoked')
'''
            result = subprocess.run([sys.executable, '-I', '-B', '-c', program,
                str(Path(ex.__file__).resolve().parent), json.dumps(paths), json.dumps(context), scenario],
                capture_output=True, text=True, timeout=30)
            self.assertEqual(result.returncode, 0, result.stderr)
            return result

    def test_source_bound_optional_query_raises_eperm_and_ordinary_oserror_fallback(self):
        # Break caught: retaining ExportError (the original import-time defect),
        # returning instead of raising, or losing the inspectable denial record.
        for scenario in ('fallback', 'exact', 'tuple_argv'):
            with self.subTest(scenario=scenario):
                result = self.run_child(scenario)
                records = [json.loads(line) for line in result.stderr.splitlines()]
                self.assertEqual(len(records), 1)
                self.assertEqual(records[0]['event'], 'subprocess.Popen')
                self.assertEqual(records[0]['errno'], 1)
                self.assertEqual(records[0]['decision'], 'denied_optional_host_query')
                self.assertEqual(records[0]['caller'], 'numpy.testing._private.utils.check_support_sve')

    def test_each_query_context_and_caller_constraint_remains_strict(self):
        # Break caught: dropping any conjunct, accepting equal/recompiled code,
        # a filename/name spoof, a namespace clone, or an intervening caller.
        for scenario in ('extra_argument', 'absolute_executable', 'absolute_argv',
                         'string_argv', 'bytes_executable', 'cwd', 'env', 'other_command',
                         'platform', 'path', 'present_usr_bin', 'present_bin',
                         'other_caller', 'dynamic_code', 'equal_code', 'spoofed_filename',
                         'spoofed_module', 'wrong_module_name', 'wrong_module_origin',
                         'intervening_caller', 'old_context', 'wrong_pin'):
            with self.subTest(scenario=scenario):
                result = self.run_child(scenario)
                self.assertEqual(result.stderr, '')

    def test_optional_caller_never_softens_other_operations_or_source_checks(self):
        # Break caught: softening other process/network operations or registering
        # changed, unlisted, or preloaded code as an approved optional caller.
        for scenario in ('operation:socket.__new__', 'operation:socket.connect',
                         'operation:os.system', 'operation:os.fork', 'operation:os.forkpty',
                         'operation:os.posix_spawn', 'operation:os.exec',
                         'changed_source', 'unlisted_source', 'preloaded'):
            with self.subTest(scenario=scenario):
                result = self.run_child(scenario)
                self.assertEqual(result.stderr, '')


if __name__ == '__main__':
    unittest.main()
