"""Synthetic observation-precondition tests only. Never calls main/Popen/SSH."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

L=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('controller',L/'task-47-terminal-controller.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)


class Precondition(unittest.TestCase):
    def setUp(self):
        temp=tempfile.TemporaryDirectory();self.addCleanup(temp.cleanup)
        self.root=Path(temp.name).resolve()
        p=patch.object(m,'LEDGER',self.root);p.start();self.addCleanup(p.stop)
        self.prefix=self.root/'post-full-observation-SYNTHETIC-NEVER-INVOKE'
        self.obs={'kind':'read_only_job_observation_not_scientific_result',
                  'full2_result_path':'/home/kmihara/hekit-r41full/results/quadruped/mnist196_ni_b48_full_seed0.json',
                  'full2_result_exists':True,
                  'processes':[{'pid':1700979,'expected_start_ticks':'465769819','exists':False}],
                  'queue_entries':['owner.json','gate-released.json','R-47-3-tests.json']}

    def write(self):
        raw=json.dumps(self.obs).encode()
        self.transport={'returncode':0,'stdout_sha256':hashlib.sha256(raw).hexdigest(),
                        'stdout_bytes':len(raw),'source_sha256':m.OBSERVER_SHA}
        Path(str(self.prefix)+'.stdout').write_bytes(raw)
        Path(str(self.prefix)+'.meta.json').write_text(json.dumps(self.transport))

    def test_terminal_test_record_is_sufficient_for_capture_precondition_only(self):
        self.write()
        with patch.object(m.subprocess,'Popen',side_effect=AssertionError('must not launch')):
            self.assertEqual(len(m.load_observation(self.prefix)),2)

    def test_still_live_full_chain_refuses(self):
        self.obs['processes'][0].update(exists=True,start_ticks='465769819',state='S')
        self.write()
        with self.assertRaises(ValueError):m.load_observation(self.prefix)

    def test_r47_pending_refuses(self):
        self.obs['queue_entries']=['owner.json','gate-released.json','post-full-r47-build.json']
        self.write()
        with self.assertRaises(ValueError):m.load_observation(self.prefix)

    def test_queue_terminal_after_r47_build_is_capturable_even_on_failure(self):
        self.obs['queue_entries']=['owner.json','gate-released.json','post-full-r47-build.json','terminal.json']
        self.write()
        self.assertEqual(len(m.load_observation(self.prefix)),2)

    def test_no_gate_or_no_r47_record_refuses(self):
        for entries in [['owner.json','terminal.json','post-full-r47-build.json'],['owner.json','gate-released.json','terminal.json']]:
            with self.subTest(entries=entries):
                self.obs['queue_entries']=entries;self.write()
                with self.assertRaises(ValueError):m.load_observation(self.prefix)

    def test_changed_observation_hash_refuses(self):
        self.write();Path(str(self.prefix)+'.stdout').write_bytes(b'{}')
        with self.assertRaises(ValueError):m.load_observation(self.prefix)


if __name__=='__main__':unittest.main()
