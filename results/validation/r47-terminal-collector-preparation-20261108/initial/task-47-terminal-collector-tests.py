"""Synthetic, tiny local operational fixtures; no SSH/GPU/training."""
import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

SOURCE = Path(__file__).with_name('task-47-terminal-collector.remote.py')
spec = importlib.util.spec_from_file_location('r47_collector', SOURCE)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


class Collection(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name).resolve()
        self.runs = self.root / 'runs'
        self.state = self.runs / 'post-full-v1'
        self.tree = self.root / 'clones/hekit-post-full-r47'
        for key, value in [('RUNS', str(self.runs)), ('STATE', str(self.state)), ('TREE', str(self.tree))]:
            p = patch.object(m, key, value); p.start(); self.addCleanup(p.stop)
        self.put(self.state / 'owner.json', {'pid': 1739838, 'started': 1793929784.594468})
        self.put(self.state / 'gate-released.json', {'at': 1794009600.0, 'config': copy.deepcopy(m.EXPECTED_CONFIG)})

    def put(self, path, data):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data) if isinstance(data, dict) else data)

    def terminal(self, rc=0):
        name = 'R-47-3-example'
        cmd = ['target/release/examples/batch_streams']
        self.record = {'job':name, 'workdir':str(self.tree), 'command':cmd, 'rc':rc,
                       'status':'done' if rc==0 else 'failed', 'log':str(self.runs/'logs'/f'{name}.log')}
        self.meta = {'job':name, 'workdir':str(self.tree), 'command':cmd[0], 'exit_status':rc,
                     'watchdog_killed':False, 'start':'2026-11-08T00:00:00+00:00',
                     'end':'2026-11-08T00:01:00+00:00', 'nice':10, 'threads_cap':12,
                     'gpu_job':True, 'hekit_synced_commit':['full bd5cc493db166a282d26b05b1e069f2a0ee629ef','dirty_files 0']}
        self.out = self.runs/'out'/name
        self.state_file = self.state/(name+'.json')
        self.put(self.state_file,self.record)
        self.put(self.out/'meta.json',self.meta)
        self.put(self.out/'cmd.txt',cmd[0]+'\n')
        self.put(self.out/'workdir.txt',str(self.tree)+'\n')
        self.put(self.out/'post.txt',f'JOB_DONE rc={rc}\n')
        self.put(self.runs/'logs'/f'{name}.log','SYNTHETIC LOG; NOT A MEASUREMENT\n')

    def test_missing_gate_refuses_before_job_reads(self):
        (self.state/'gate-released.json').unlink()
        got=m.collect()
        self.assertEqual(got['exit_code'],2)
        self.assertIn('owner.json',got['files'])
        self.assertFalse(any('R-47' in k for k in got['files']))

    def test_owner_mismatch_preserves_raw_and_refuses(self):
        self.put(self.state/'owner.json',{'pid':123,'started':1793929784.594468})
        got=m.collect()
        self.assertEqual(got['exit_code'],2)
        self.assertIn('owner.json',got['files'])

    def test_complete_example_and_missing_terminal_states(self):
        self.terminal()
        got=m.collect()
        self.assertEqual(got['exit_code'],0,got.get('error'))
        self.assertFalse(got['scientific_acceptance'])
        self.assertEqual(got['jobs']['R-47-3-example']['runner_exit'],0)
        self.assertEqual(got['jobs']['R-47-3-tests']['capture_state'],'no_terminal_record_observed')
        self.assertIn('R-47-3-example/program.log',got['files'])
        self.assertIn('R-47-3-example/state.final.json',got['files'])

    def test_failed_child_is_collected_not_called_pass(self):
        self.terminal(rc=1)
        got=m.collect()
        self.assertEqual(got['exit_code'],0,got.get('error'))
        self.assertEqual(got['jobs']['R-47-3-example']['runner_exit'],1)
        self.assertEqual(got['jobs']['R-47-3-example']['queue_status'],'failed')
        self.assertFalse(got['scientific_acceptance'])

    def test_nsys_unavailable_has_no_runner(self):
        self.put(self.state/'R-47-3-nsys.json',{'status':'skipped: nsys unavailable'})
        got=m.collect()
        self.assertEqual(got['exit_code'],0,got.get('error'))
        self.assertEqual(got['jobs']['R-47-3-nsys']['capture_state'],'nsys_unavailable')
        self.assertNotIn('R-47-3-nsys/meta.json',got['files'])

    def test_inconsistent_terminal_keeps_original_bytes(self):
        for field,value in [('exit_status',7),('command','false'),('hekit_synced_commit',['full '+'0'*40,'dirty_files 0'])]:
            with self.subTest(field=field):
                self.terminal()
                self.meta[field]=value
                self.put(self.out/'meta.json',self.meta)
                got=m.collect()
                self.assertEqual(got['exit_code'],2)
                self.assertIn('R-47-3-example/meta.json',got['files'])
                self.assertIn('R-47-3-example/state.json',got['files'])

    def test_inconsistent_post_marker_is_retained(self):
        self.terminal()
        self.put(self.out/'post.txt','JOB_DONE rc=9\n')
        got=m.collect()
        self.assertEqual(got['exit_code'],2)
        self.assertIn('R-47-3-example/post.txt',got['files'])

    def test_duplicate_state_key_is_not_accepted(self):
        self.terminal()
        self.put(self.state_file,'{"job":"R-47-3-example","job":"other"}')
        got=m.collect()
        self.assertEqual(got['exit_code'],2)
        self.assertIn('R-47-3-example/state.json',got['files'])

    def test_aggregate_bound_refuses_before_oversized_read(self):
        with patch.object(m,'TOTAL',256): got=m.collect()
        self.assertEqual(got['exit_code'],2)
        self.assertIn('owner.json',got['files'])
        self.assertNotIn('gate-released.json',got['files'])
        self.assertLessEqual(got['original_bytes_collected'],256)

    def test_changed_terminal_state_retains_both_observations(self):
        self.terminal()
        original=m.read_stable
        calls=0
        def read(path,byte_limit=m.PER_FILE):
            nonlocal calls
            if path==str(self.state_file):
                calls+=1
                if calls==2:
                    self.record['status']='changed'
                    self.put(self.state_file,self.record)
            return original(path,byte_limit)
        with patch.object(m,'read_stable',side_effect=read): got=m.collect()
        self.assertEqual(got['exit_code'],2)
        self.assertIn('R-47-3-example/state.json',got['files'])
        self.assertIn('R-47-3-example/state.final.json',got['files'])


if __name__=='__main__': unittest.main()
