#!/usr/bin/env python3
"""Synthetic fixtures only: no cargo, GPU, SSH, or training."""
import importlib.util
import json
import os
import shlex
from pathlib import Path
import subprocess
import tempfile
import time
import unittest
from unittest.mock import patch

SCRIPT = Path(__file__).resolve().parents[1] / 'results/quadruped/scripts/post_full_queue.py'
spec = importlib.util.spec_from_file_location('post_full_queue', SCRIPT)
q = importlib.util.module_from_spec(spec)


class QueueTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not SCRIPT.exists():
            raise AssertionError("post-FULL queue implementation is missing")
        spec.loader.exec_module(q)

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.now = time.time()
        self.cfg = {'runs': str(self.root / 'hekit-runs'), 'full_tree': str(self.root / 'full'),
                    'full_commit': 'ff2c2eb', 'not_before_epoch': self.now - 10,
                    'chain_pid': 99999999, 'chain_start_ticks': '123',
                    'chain_log': str(self.root / 'chain10.log'), 'poll_seconds': .01,
                    'max_wait_seconds': .05, 'job_poll_seconds': .01}
        for name in q.FULL_JOBS:
            d = self.root / 'hekit-runs/out' / name
            d.mkdir(parents=True)
            (d / 'post.txt').write_text('JOB_DONE rc=0\n')
            (d / 'meta.json').write_text(json.dumps({'job': name, 'exit_status': 0,
                'workdir': self.cfg['full_tree'], 'start': self.now - 5, 'end': self.now - 1,
                'watchdog_killed': False, 'hekit_synced_commit': ['commit ff2c2eb']}))
        Path(self.cfg['chain_log']).write_text('CHAIN10_DONE\n')

    def test_both_success_and_chain_terminal_required(self):
        self.assertTrue(q.gates(self.cfg))
        Path(self.cfg['chain_log']).unlink()
        self.assertFalse(q.gates(self.cfg))
        Path(self.cfg['chain_log']).write_text('CHAIN10_DONE\n')
        with patch.object(q, 'process_ticks', return_value='123'):
            self.assertFalse(q.gates(self.cfg))

    def test_missing_full_marker_never_released_by_dead_pid(self):
        (self.root / 'hekit-runs/out/R-41-FULL-2/post.txt').unlink()
        self.assertFalse(q.gates(self.cfg))

    def test_failed_and_stale_full_gate_fail_closed(self):
        d = self.root / 'hekit-runs/out/R-41-FULL-1'
        (d / 'post.txt').write_text('JOB_DONE rc=1\n')
        with self.assertRaises(q.QueueError): q.gates(self.cfg)
        (d / 'post.txt').write_text('JOB_DONE rc=0\n')
        meta = json.loads((d / 'meta.json').read_text())
        meta['start'] = self.now - 100
        (d / 'meta.json').write_text(json.dumps(meta))
        with self.assertRaises(q.QueueError): q.gates(self.cfg)

    def test_gate_rejects_wrong_commit_or_reused_pid(self):
        with patch.object(q, 'process_ticks', return_value='456'):
            with self.assertRaises(q.QueueError): q.gates(self.cfg)
        self.cfg['full_commit'] = 'bad'
        with self.assertRaises(q.QueueError): q.gates(self.cfg)

    def test_wait_rechecks_until_chain_finishes(self):
        with patch.object(q, 'gates', side_effect=[False, False, True]) as gate:
            q.wait_gates(self.cfg)
            self.assertEqual(gate.call_count, 3)
        with patch.object(q, 'gates', return_value=False):
            with self.assertRaises(q.QueueError): q.wait_gates(self.cfg)

    def runner_fixture(self, name, overrides=None, rc=0):
        runner = self.root / (name + '-runner.sh')
        script = r'''#!/bin/bash
python3 - "$@" <<'FIXTURE'
import datetime, json, os, pathlib, shlex, sys, time
name, tree, threads, gpu, sep, *command = sys.argv[1:]
out = pathlib.Path(os.environ['HOME']) / 'hekit-runs/out' / name
out.mkdir(parents=True)
logs = out.parent.parent / 'logs'
logs.mkdir(exist_ok=True)
now = time.time()
meta = {'job':name, 'workdir':tree, 'threads_cap':12, 'gpu_job':gpu=='1',
        'nice':10, 'exit_status':RC, 'watchdog_killed':False, 'start':now, 'end':now,
        'command':shlex.join(command), 'hekit_synced_commit':['commit 0e07561', 'full COMMIT', 'dirty_files 0']}
meta.update(OVERRIDES)
(out/'meta.json').write_text(json.dumps(meta))
(out/'cmd.txt').write_text(shlex.join(command))
(out/'post.txt').write_text('JOB_DONE rc=RC\n')
(logs/(name+'.log')).write_text('device error: out of memory' if RC else '')
FIXTURE
'''
        script = script.replace('OVERRIDES', repr(overrides or {})).replace('COMMIT', '0e075618ab6e28b0cf260f98feb7f80ee1ccf1f3').replace('RC', str(rc))
        runner.write_text(script)
        obj = q.Queue(self.cfg | {'runner': str(runner), 'state': str(self.root / 'state')})
        obj.env['HOME'] = str(self.root)
        obj.tree_commits = {str(self.root): '0e075618ab6e28b0cf260f98feb7f80ee1ccf1f3'}
        return obj

    def test_wrapper_zero_is_not_child_success(self):
        obj = self.runner_fixture('fixture', rc=1)
        result = obj.job('fixture', self.root, False, ['false'])
        self.assertEqual(result['rc'], 1)
        self.assertEqual(result['status'], 'failed: device OOM')
        self.assertEqual(result['wrapper_rc'], 0)

    def test_terminal_rejects_wrong_invocation_identity(self):
        cases = [{'job':'WRONG_JOB'}, {'workdir':'/wrong'},
                 {'hekit_synced_commit':['full '+'a'*40]}, {'start':1, 'end':2},
                 {'command':'wrong command'}, {'threads_cap':24}]
        for i, override in enumerate(cases):
            with self.subTest(override=override):
                name = 'wrong-' + str(i)
                obj = self.runner_fixture(name, override)
                with self.assertRaises(q.QueueError): obj.job(name, self.root, False, ['true'])

    def output_fixture(self, gpu=False):
        command = ['target/release/ablation', '--backend', 'cuda' if gpu else 'cpu']
        return command, {'all_checks_pass':True, 'provenance': {
            'written_by':'the ablation binary (raw output, not assembled)',
            'git_commit':'0e075618ab6e28b0cf260f98feb7f80ee1ccf1f3',
            'git_modified_tracked_files':'none', 'command_line':' '.join(command),
            'started_unix':self.now, 'finished_unix':self.now + 1,
            'parallel_feature':True,
            'build':{'debug_assertions':False, 'package_version':'0.0.0',
                     'target_os':'linux', 'target_arch':os.uname().machine,
                     'features':{'parallel':True, 'cuda':True}},
            'gpu': {'name':'fixture', 'free_mib_at_start':3000, 'total_mib':24000} if gpu else None}}

    def test_output_checks_require_actual_pinned_provenance(self):
        for gpu in [False, True]:
            command, data = self.output_fixture(gpu)
            q.validate_output(data, command, data['provenance']['git_commit'], gpu, self.now-1, self.now+2)
            for key, value in [('git_commit','wrong'), ('written_by','wrong'),
                               ('command_line','wrong'), ('started_unix',1),
                               ('build',{}), ('git_modified_tracked_files',' M src.rs')]:
                broken = json.loads(json.dumps(data))
                broken['provenance'][key] = value
                with self.subTest(gpu=gpu,key=key):
                    with self.assertRaises(q.QueueError):
                        q.validate_output(broken, command, data['provenance']['git_commit'], gpu, self.now-1, self.now+2)
            with self.assertRaises(q.QueueError):
                q.validate_output({'all_checks_pass':True}, command, data['provenance']['git_commit'], gpu, self.now-1, self.now+2)

    def test_r52_actual_writer_build_and_commit_schema(self):
        command, data = self.output_fixture()
        command[0] = 'target/release/mnist'
        p = data['provenance']
        p.update({'git_commit':'9305d2e1f4f9999296cd4b4aaf55ea631fbca78e',
                  'written_by':'the mnist binary (raw output, not assembled)',
                  'command_line':' '.join(command)})
        p.pop('gpu')
        p['build']['package_version'] = '1.0.0'
        p['build']['features']['cuda'] = False
        q.validate_output(data, command, p['git_commit'], False, self.now-1, self.now+2)
        p['build']['package_version'] = '0.0.0'
        with self.assertRaises(q.QueueError):
            q.validate_output(data, command, p['git_commit'], False, self.now-1, self.now+2)

    def test_gpu_truthy_placeholder_and_future_finish_rejected(self):
        command, data = self.output_fixture(True)
        data['provenance']['gpu'] = 'truthy placeholder'
        with self.assertRaises(q.QueueError):
            q.validate_output(data, command, data['provenance']['git_commit'], True, self.now-1, self.now+2)
        command, data = self.output_fixture(True)
        data['provenance']['finished_unix'] = self.now + 1000
        with self.assertRaises(q.QueueError):
            q.validate_output(data, command, data['provenance']['git_commit'], True, self.now-1, self.now+2)

    def optimizer(self, results):
        obj = q.Queue(self.cfg | {'state': str(self.root / 'state')})
        calls = []
        def job(name, tree, gpu, command, **kw):
            calls.append((name, command))
            return results.get(name, {'rc': 0, 'status': 'done', 'skip': False})
        obj.job = job
        obj.r48(self.root)
        return calls

    def test_exit3_skip_falls_back_and_matches_cpu_sgd(self):
        calls = self.optimizer({'R-48-2': {'rc': 3, 'skip': True, 'status': 'skipped'}})
        self.assertIn('R-48-2-sgd', [n for n, c in calls])
        cpu = dict(calls)['R-48-3']
        self.assertEqual(cpu[cpu.index('--optimizer') + 1], 'sgd')

    def test_two_gpu_skips_keep_cpu_nag(self):
        calls = self.optimizer({n: {'rc': 3, 'skip': True, 'status': 'skipped'}
                                for n in ['R-48-2', 'R-48-2-sgd']})
        cpu = dict(calls)['R-48-3']
        self.assertEqual(cpu[cpu.index('--optimizer') + 1], 'nag')

    def test_oom_or_exit3_without_skip_never_falls_back(self):
        for result in [{'rc': 1, 'skip': False, 'status': 'failed: device OOM'},
                       {'rc': 3, 'skip': False, 'status': 'failed'}]:
            calls = self.optimizer({'R-48-2': result})
            self.assertNotIn('R-48-2-sgd', dict(calls))
            self.assertIn('R-48-4', dict(calls))

    def test_atomic_duplicate_lock_and_terminal_record(self):
        state = self.root / 'state'
        with q.queue_lock(state):
            with self.assertRaises(q.QueueError):
                with q.queue_lock(state): pass
        with self.assertRaises(q.QueueError):
            with q.queue_lock(state): pass

    def test_full_partial_is_not_success(self):
        (self.root / 'hekit-runs/out/R-41-FULL-2/partial.txt').write_text('partial')
        with self.assertRaises(q.QueueError): q.gates(self.cfg)

    def test_stale_chain_terminal_rejected(self):
        os.utime(self.cfg['chain_log'], (1, 1))
        with self.assertRaises(q.QueueError): q.gates(self.cfg)

    def test_partial_projection_and_own_pid_signal(self):
        text = '\n'.join(f'mnist: epoch 0 update {i} (48 samples): 100.0 s, 1 KS (2 bootstraps), matches prediction true; elapsed {(i+1)*100+20} s, load 1' for i in range(5))
        self.assertEqual(q.partial_projection(text), 2120)
        self.assertIsNone(q.partial_projection(text.splitlines()[0]))
        with patch.object(q, 'descendants', return_value=[]):
            with self.assertRaises(q.QueueError): q.stop_own_tree(100, 200)
        with patch.object(q, 'descendants', side_effect=[[200], [201]]), patch.object(q, 'process_ticks', return_value='123'), patch.object(q.os, 'kill') as kill:
            q.stop_own_tree(100, 200)
            self.assertEqual([call.args[0] for call in kill.call_args_list], [201])

    def test_partial_waits_for_identified_children_and_parent(self):
        with patch.object(q, 'process_ticks', side_effect=['123','123',None,None]), patch.object(q.time, 'sleep') as sleep:
            q.wait_identities_gone([(201,'123'), (200,'123')], timeout=.05, poll=.001)
            self.assertTrue(sleep.called)
        with patch.object(q, 'process_ticks', return_value='123'):
            with self.assertRaises(q.QueueError):
                q.wait_identities_gone([(201,'123')], timeout=.01, poll=.001)
        with patch.object(q, 'process_ticks', return_value='REUSED'), patch.object(q.os, 'kill') as kill:
            q.wait_identities_gone([(201,'123')], timeout=.01, poll=.001)
            kill.assert_not_called()

    def test_manifest_mismatch_rejected(self):
        source = self.root / 'source'
        source.mkdir()
        (source / 'x.npy').write_bytes(b'bad')
        (source / 'data_manifest.json').write_text(json.dumps({'files': [
            {'file': 'x.npy', 'sha256_npy_file': '0' * 64}]}))
        with self.assertRaises(q.QueueError): q.copy_manifest_data(source, self.root / 'dest')
        self.assertFalse((self.root / 'dest').exists())


if __name__ == '__main__': unittest.main()
