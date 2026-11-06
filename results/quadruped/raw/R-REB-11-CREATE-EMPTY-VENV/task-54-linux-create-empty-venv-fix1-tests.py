"""Isolated tests of exact wrapper helpers; never execute its remote top level."""
from pathlib import Path
import ast
import hashlib
import os
import stat
import subprocess
import sys
import tempfile
import time
import unittest
from unittest import mock

SOURCE = Path(__file__).with_name('task-54-linux-create-empty-venv-20261107.py')
tree = ast.parse(SOURCE.read_text())
names = {'digest', 'require', 'run_creator', 'inspect_created_tree'}
helpers = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
assert {n.name for n in helpers} == names
namespace = dict(Path=Path, hashlib=hashlib, os=os, stat=stat, subprocess=subprocess, time=time)
exec(compile(ast.Module(body=helpers, type_ignores=[]), str(SOURCE), 'exec'), namespace)


class LifecycleTests(unittest.TestCase):
    def command(self, body):
        return [sys.executable, '-I', '-B', '-c', body]

    def test_proc_sample_permission_error_still_waits_and_collects(self):
        record = {}
        with mock.patch.object(Path, 'read_text', side_effect=PermissionError('fixture stat denied')):
            namespace['run_creator'](self.command('print("done")'), '/',
                                     {'PATH': '/usr/bin:/bin'}, record, timeout=5)
        self.assertEqual(record['child_returncode'], 0)
        self.assertEqual(bytes.fromhex(record['child_stdout_hex']), b'done\n')
        self.assertEqual(record['proc_stat_error']['type'], 'PermissionError')
        self.assertTrue(record['child_terminal_observed'] and record['child_streams_complete'])

    def test_controller_error_kills_only_owned_child_and_preserves_error(self):
        real_popen = subprocess.Popen
        children = []
        def spawn(*args, **kwargs):
            child = real_popen(*args, **kwargs)
            children.append(child)
            original = child.communicate
            attempts = []
            def communicate(*a, **k):
                attempts.append(1)
                if len(attempts) == 1:
                    raise OSError('fixture controller failure')
                return original(*a, **k)
            child.communicate = communicate
            return child
        record = {}
        with mock.patch.object(subprocess, 'Popen', side_effect=spawn):
            with self.assertRaisesRegex(OSError, 'fixture controller failure'):
                namespace['run_creator'](self.command('import time; time.sleep(20)'), '/',
                                         {'PATH': '/usr/bin:/bin'}, record, timeout=5)
        self.assertEqual(len(children), 1)
        self.assertIsNotNone(children[0].poll())
        self.assertTrue(record['child_terminal_observed'] and record['child_streams_complete'])
        self.assertLess(record['child_returncode'], 0)
        self.assertEqual(record['controller_error']['message'], 'fixture controller failure')

    def test_timeout_terminates_and_collects_owned_child(self):
        record = {}
        with self.assertRaisesRegex(RuntimeError, 'timed out'):
            namespace['run_creator'](self.command('import time; time.sleep(20)'), '/',
                                     {'PATH': '/usr/bin:/bin'}, record, timeout=.03)
        self.assertTrue(record['timed_out'])
        self.assertTrue(record['child_terminal_observed'] and record['child_streams_complete'])
        self.assertLess(record['child_returncode'], 0)

    def test_unresolved_cleanup_is_explicit_failure_with_partial_output(self):
        class FictionalChild:
            pid = 1
            returncode = None
            def poll(self): return None
            def kill(self): self.kill_called = True
            def communicate(self, timeout):
                raise subprocess.TimeoutExpired('fictional', timeout, output=b'partial')
        child = FictionalChild()
        record = {}
        with mock.patch.object(subprocess, 'Popen', return_value=child):
            with self.assertRaisesRegex(RuntimeError, 'unresolved'):
                namespace['run_creator'](['fictional'], '/', {}, record,
                                         timeout=.01, cleanup_timeout=.01)
        self.assertTrue(child.kill_called)
        self.assertFalse(record['child_terminal_observed'])
        self.assertFalse(record['child_streams_complete'])
        self.assertEqual(record['child_cleanup_status'], 'unresolved_terminal_or_streams')
        self.assertEqual(bytes.fromhex(record['child_stdout_hex']), b'partial')
        self.assertEqual(record['cleanup_errors'][0]['operation'], 'collect')


class LayoutTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name) / 'candidate'
        self.root.mkdir()
        for name in ['bin', 'include/python3.11', 'lib/python3.11/site-packages']:
            (self.root / name).mkdir(parents=True)
        for name in ['pyvenv.cfg', 'bin/Activate.ps1', 'bin/activate',
                     'bin/activate.csh', 'bin/activate.fish']:
            (self.root / name).write_text('original fictional fixture\n')
        (self.root / 'lib64').symlink_to('lib', target_is_directory=True)
        for name in ['python', 'python3', 'python3.11']:
            (self.root / 'bin' / name).symlink_to('/fictional/not-executed-python')
        self.record = {}

    def tearDown(self):
        self.tmp.cleanup()

    def inspect(self):
        return namespace['inspect_created_tree'](self.root, self.record, expected_uid=os.getuid())

    def test_expected_shallow_layout_hashes_only_five_files(self):
        entries = self.inspect()
        self.assertEqual(len(entries), 15)
        self.assertEqual(sum('sha256' in e for e in entries), 5)

    def test_configuration_symlink_rejected_before_content(self):
        cfg = self.root / 'pyvenv.cfg'
        cfg.unlink()
        cfg.symlink_to('/fictional/outside-config')
        with mock.patch.dict(namespace, digest=mock.Mock(side_effect=AssertionError('content read'))):
            with self.assertRaisesRegex(RuntimeError, 'name/kind'):
                self.inspect()
            namespace['digest'].assert_not_called()
        self.assertTrue(self.record['created_tree'])

    def test_oversized_expected_file_rejected_before_any_content(self):
        with (self.root / 'bin/activate').open('wb') as f:
            f.truncate(65537)
        with mock.patch.dict(namespace, digest=mock.Mock(side_effect=AssertionError('content read'))):
            with self.assertRaisesRegex(RuntimeError, 'oversized'):
                self.inspect()
            namespace['digest'].assert_not_called()
        self.assertTrue(self.record['created_tree'])

    def test_unknown_directory_rejected_without_descent_or_content(self):
        unknown = self.root / 'unexpected'
        (unknown / 'deep').mkdir(parents=True)
        (unknown / 'deep/outside').symlink_to('/')
        actual_scandir = os.scandir
        visited = []
        def scandir(path):
            visited.append(Path(path))
            self.assertNotIn(unknown, [Path(path), *Path(path).parents])
            return actual_scandir(path)
        with mock.patch.object(os, 'scandir', side_effect=scandir), \
             mock.patch.dict(namespace, digest=mock.Mock(side_effect=AssertionError('content read'))):
            with self.assertRaisesRegex(RuntimeError, 'name/kind'):
                self.inspect()
            namespace['digest'].assert_not_called()
        self.assertEqual(visited, [self.root])
        self.assertTrue(any(e['path'] == 'unexpected' for e in self.record['created_tree']))


if __name__ == '__main__':
    print('Exact helper-source SHA256:', hashlib.sha256(SOURCE.read_bytes()).hexdigest(), flush=True)
    unittest.main(verbosity=2)
