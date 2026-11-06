"""Task54 runner integration: temporary git repos and fake binaries only; no cargo work."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import shlex
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class RunAllTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        base = Path(self.tmp.name)
        self.root, self.bin = base / 'repo', base / 'stubs'
        self.root.mkdir()
        self.bin.mkdir()
        (self.root / 'scripts').mkdir()
        (self.root / 'tools').mkdir()
        shutil.copyfile(ROOT / 'scripts/run_all.sh', self.root / 'scripts/run_all.sh')
        (self.root / 'tools/paper_tables.py').write_text(
            'import pathlib,sys\n'
            'pathlib.Path("table-invocation.txt").write_text(" ".join(sys.argv[1:]))\n')
        self.git('init', '-q')
        self.git('config', 'user.name', 'Synthetic Test')
        self.git('config', 'user.email', 'synthetic@example.invalid')
        self.git('add', '.')
        self.git('commit', '-qm', 'SYNTHETIC fixture, never a measurement')
        # Only expensive external cargo work is stubbed. The runner, git, shell, output
        # validation and receipts remain real. Each binary writes its own fixture JSON.
        cargo = self.bin / 'cargo'
        cargo.write_text('''#!/usr/bin/env python3
import json, os, pathlib, subprocess, sys
if '--version' in sys.argv:
 print('cargo SYNTHETIC 1.0');sys.exit(0)
with open('cargo-invocations.txt', 'a') as f:
 f.write(json.dumps({'args':sys.argv[1:],'nice':os.getpriority(os.PRIO_PROCESS,0)})+'\\n')
args=sys.argv
target_dir=args[args.index('--target-dir')+1] if '--target-dir' in args else os.environ.get('CARGO_TARGET_DIR','target')
base=pathlib.Path(target_dir)
if '--target' in args:base=base/args[args.index('--target')+1]
base=base/'release'
base.mkdir(parents=True, exist_ok=True)
body = """#!/usr/bin/env python3
import json,os,pathlib,subprocess,sys,time
p=pathlib.Path(sys.argv[sys.argv.index('--out')+1])
commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
writer='the '+pathlib.Path(sys.argv[0]).name+' binary (raw output, not assembled)'
doc={'synthetic_fixture':True,'provenance':{'written_by':writer,'git_commit':commit,
 'git_modified_tracked_files':'none','command_line':' '.join(sys.argv),
 'started_unix':time.time()-1,'finished_unix':time.time()}}
if os.environ.get('TEST_NO_PROVENANCE'):doc.pop('provenance')
if os.environ.get('TEST_WRONG_WRITER'):doc['provenance']['written_by']='runner-only metadata'
p.write_text(json.dumps(doc))
"""
for name in ('plain','mnist','iris','mnist_reboot_layout','ablation'):
 p=base/name;p.write_text(body);p.chmod(0o755)
''')
        cargo.chmod(0o755)
        rustc = self.bin / 'rustc'
        rustc.write_text('#!/bin/sh\nprintf "rustc SYNTHETIC 1.0\\nhost: synthetic-host\\n"\n')
        rustc.chmod(0o755)
        self.env = dict(os.environ, PATH=str(self.bin) + os.pathsep + os.environ['PATH'])

    def git(self, *args):
        return subprocess.run(['git', *args], cwd=self.root, check=True,
                              capture_output=True, text=True).stdout.strip()

    def run_runner(self, *args, env=None):
        return subprocess.run(['/bin/bash', 'scripts/run_all.sh', *args], cwd=self.root,
                              env=env or self.env, capture_output=True, text=True)

    def test_missing_input_only_returns_failure_without_build(self):
        # Break caught: early nothing-to-do exit claiming an incomplete run succeeded.
        p = self.run_runner('--quick', '--only', '^q_int784_b32_sub32_cpu$')
        self.assertEqual(p.returncode, 1, p.stdout + p.stderr)
        self.assertFalse((self.root / 'cargo-invocations.txt').exists())

    def test_threads_must_be_positive_and_at_most_24(self):
        # Break caught: oversubscribed builds or invalid env reaching a binary.
        for value in ('0', '25', '-1', 'x', '1.5'):
            with self.subTest(value=value):
                p = self.run_runner('--dry-run', '--threads', value)
                self.assertEqual(p.returncode, 2, p.stdout + p.stderr)

    def test_tables_only_never_builds_or_runs(self):
        # Break caught: forcing experiments just to refresh draft paper tables.
        p = self.run_runner('--tables-only')
        self.assertEqual(p.returncode, 0, p.stderr)
        self.assertTrue((self.root / 'table-invocation.txt').exists())
        self.assertFalse((self.root / 'cargo-invocations.txt').exists())
        self.assertFalse((self.root / 'results/run_all').exists())

    def test_tables_only_can_explicitly_select_new_full_outputs(self):
        # Break caught: silently changing canonical inputs, or ignoring explicit run outputs.
        p = self.run_runner('--tables-only', '--from-run-all', 'full')
        self.assertEqual(p.returncode, 0, p.stderr)
        self.assertIn('--from-run-all full', (self.root / 'table-invocation.txt').read_text())

    def test_untracked_source_is_not_a_clean_head(self):
        # Break caught: compiling untracked sources while claiming a clean commit.
        (self.root / 'tools/new_source.py').write_text('# untracked source\n')
        p = self.run_runner('--quick', '--only', '^q_plain_bc_mlp$')
        self.assertEqual(p.returncode, 2, p.stdout + p.stderr)
        self.assertIn('untracked', p.stderr.lower())

    def test_build_receipt_binds_binary_and_source_and_nice_job_cap(self):
        # Break caught: no-build accepting a stale executable / unbounded un-niced build.
        p = self.run_runner('--quick', '--only', '^q_plain_bc_mlp$', '--threads', '2')
        self.assertEqual(p.returncode, 0, p.stdout + p.stderr)
        inv = json.loads((self.root / 'cargo-invocations.txt').read_text().splitlines()[0])
        self.assertGreaterEqual(inv['nice'], 10)
        self.assertIn('2', inv['args'])
        meta = json.loads((self.root / 'results/run_all/quick/meta/q_plain_bc_mlp.run.json').read_text())
        self.assertEqual(meta['status'], 'done')
        self.assertEqual(len(meta['source_identity']), 64)
        self.assertEqual(len(meta['binary_sha256']), 64)
        (self.root / shlex.split(meta['command'])[0]).write_text('#!/bin/sh\nexit 0\n')
        p = self.run_runner('--quick', '--only', '^q_plain_bc_mlp$', '--no-build', '--allow-dirty')
        self.assertEqual(p.returncode, 1, p.stdout + p.stderr)
        self.assertIn('receipt', p.stderr.lower())

    def test_runner_metadata_cannot_replace_writer_provenance(self):
        # Break caught: successful wrapper status accepting provenance-less JSON.
        env = dict(self.env, TEST_NO_PROVENANCE='1')
        p = self.run_runner('--quick', '--only', '^q_plain_bc_mlp$', env=env)
        self.assertEqual(p.returncode, 1, p.stdout + p.stderr)
        meta = json.loads((self.root / 'results/run_all/quick/meta/q_plain_bc_mlp.run.json').read_text())
        self.assertEqual(meta['status'], 'failed')

    def test_no_build_without_receipt_never_accepts_existing_executable(self):
        # Break caught: metadata claiming HEAD for an unknown preexisting binary.
        path = self.root / 'target/release/plain'
        path.parent.mkdir(parents=True)
        path.write_text('#!/bin/sh\nexit 0\n')
        path.chmod(0o755)
        p = self.run_runner('--quick', '--only', '^q_plain_bc_mlp$', '--no-build', '--allow-dirty')
        self.assertEqual(p.returncode, 1, p.stdout + p.stderr)
        self.assertIn('receipt', p.stderr.lower())

    def test_untracked_source_identity_is_separate_and_invalidates_receipt(self):
        # Break caught: omitting untracked source bytes or calling them tracked changes.
        source = self.root / 'tools/new_source.py'
        source.write_text('# synthetic source version one\n')
        args = ('--quick', '--only', '^q_plain_bc_mlp$', '--allow-dirty')
        p = self.run_runner(*args)
        self.assertEqual(p.returncode, 0, p.stdout + p.stderr)
        meta = json.loads((self.root / 'results/run_all/quick/meta/q_plain_bc_mlp.run.json').read_text())
        self.assertEqual(meta['git_modified_tracked_files'], 'none')
        self.assertIn('tools/new_source.py', meta['git_modified_tracked_and_untracked_files'])
        source.write_text('# synthetic source version two\n')
        p = self.run_runner(*args, '--no-build')
        self.assertEqual(p.returncode, 1, p.stdout + p.stderr)
        self.assertIn('receipt', p.stderr.lower())

    def test_writer_gate_cannot_be_disabled_by_python_optimization(self):
        # Break caught: production validation implemented as removable Python asserts.
        p = self.run_runner('--quick', '--only', '^q_plain_bc_mlp$',
                            env=dict(self.env, TEST_WRONG_WRITER='1', PYTHONOPTIMIZE='1'))
        self.assertEqual(p.returncode, 1, p.stdout + p.stderr)

    def test_fresh_build_ignores_inherited_target_dir_and_stale_default_binary(self):
        # P1: fresh build in another directory must not attest/execute an old default binary.
        old = self.root / 'target/release/plain'
        old.parent.mkdir(parents=True)
        old.write_text('#!/bin/sh\necho stale-binary-executed >&2\nexit 99\n')
        old.chmod(0o755)
        p = self.run_runner('--quick', '--only', '^q_plain_bc_mlp$', '--allow-dirty',
                            env=dict(self.env, CARGO_TARGET_DIR=str(self.root / 'elsewhere')))
        self.assertEqual(p.returncode, 0, p.stdout + p.stderr)
        meta = json.loads((self.root / 'results/run_all/quick/meta/q_plain_bc_mlp.run.json').read_text())
        binary = self.root / shlex.split(meta['command'])[0]
        self.assertNotEqual(binary, old)
        self.assertNotIn('stale-binary-executed', (self.root / 'results/run_all/quick/logs/q_plain_bc_mlp.log').read_text())
        self.assertIn('--target-dir', json.loads((self.root / 'cargo-invocations.txt').read_text().splitlines()[0])['args'])

    def test_cargo_config_changes_invalidate_previous_receipt(self):
        # P2: ignored/untracked Cargo config must participate in source identity.
        p = self.run_runner('--quick', '--only', '^q_plain_bc_mlp$')
        self.assertEqual(p.returncode, 0, p.stdout + p.stderr)
        for name in ('config.toml', 'config'):
            with self.subTest(name=name):
                folder = self.root / '.cargo'
                folder.mkdir(exist_ok=True)
                (folder / name).write_text('[build]\nrustflags = ["--cfg", "synthetic_new_flags"]\n')
                p = self.run_runner('--quick', '--only', '^q_plain_bc_mlp$', '--no-build', '--allow-dirty')
                self.assertEqual(p.returncode, 1, p.stdout + p.stderr)
                self.assertIn('receipt', p.stderr.lower())
                (folder / name).unlink()

    def test_changed_thread_recipe_reruns_with_existing_build(self):
        # P2: a valid two-thread run cannot satisfy a newly requested three-thread run.
        args = ('--quick', '--only', '^q_plain_bc_mlp$')
        p = self.run_runner(*args, '--threads', '2')
        self.assertEqual(p.returncode, 0, p.stdout + p.stderr)
        p = self.run_runner(*args, '--threads', '3', '--no-build', '--allow-dirty')
        self.assertEqual(p.returncode, 0, p.stdout + p.stderr)
        self.assertNotIn('nothing to do', p.stdout)
        meta = json.loads((self.root / 'results/run_all/quick/meta/q_plain_bc_mlp.run.json').read_text())
        self.assertEqual(meta['rayon_num_threads'], '3')
        self.assertIn('--jobs 3', meta['command'])

    def test_metadata_persistence_failure_is_a_failed_run(self):
        # P2: a binary's exit0 is insufficient when its runner record cannot be saved.
        dest = self.root / 'results/run_all/quick/meta/q_plain_bc_mlp.run.json'
        dest.mkdir(parents=True)
        p = self.run_runner('--quick', '--only', '^q_plain_bc_mlp$', '--allow-dirty')
        self.assertEqual(p.returncode, 1, p.stdout + p.stderr)
        self.assertIn('metadata', p.stderr.lower())
        self.assertNotIn(': done (exit 0)', p.stdout)

    def test_rustflags_change_requires_a_fresh_build_receipt(self):
        # Build-environment changes must not masquerade as the source identity alone.
        p = self.run_runner('--quick', '--only', '^q_plain_bc_mlp$')
        self.assertEqual(p.returncode, 0, p.stdout + p.stderr)
        p = self.run_runner('--quick', '--only', '^q_plain_bc_mlp$', '--no-build', '--allow-dirty',
                            env=dict(self.env, RUSTFLAGS='--cfg synthetic_changed_environment'))
        self.assertEqual(p.returncode, 1, p.stdout + p.stderr)
        self.assertIn('receipt', p.stderr.lower())


if __name__ == '__main__':
    unittest.main()
