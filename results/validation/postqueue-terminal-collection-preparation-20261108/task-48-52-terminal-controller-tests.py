"""Synthetic pure precondition checks; main/Popen/SSH are never called."""
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
L=Path(__file__).resolve().parent
class Precondition(unittest.TestCase):
    def run_case(self,family,case):
        path=L/f'task-{family[1:]}-terminal-controller.py'
        self.assertTrue(path.is_file(),'controller not implemented')
        spec=importlib.util.spec_from_file_location('controller_'+family,path)
        m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
        with tempfile.TemporaryDirectory() as tmp, patch.object(m.subprocess,'Popen',side_effect=AssertionError('no launch')):
            root=Path(tmp).resolve();prefix=root/'post-full-observation-SYNTHETIC-NEVER-INVOKE'
            obs={'kind':'read_only_job_observation_not_scientific_result','full2_result_path':'/home/kmihara/hekit-r41full/results/quadruped/mnist196_ni_b48_full_seed0.json','full2_result_exists':True,'processes':[{'pid':1700979,'expected_start_ticks':'465769819','exists':False}],'queue_entries':['owner.json','gate-released.json','R-48-4.json' if family=='r48' else 'R-52-2.json']}
            accepted=True
            if case=='queue_failed_before_family_build':obs['queue_entries']=['owner.json','gate-released.json','terminal.json']
            elif case=='pending':obs['queue_entries']=['owner.json','gate-released.json'];accepted=False
            elif case=='other_family':obs['queue_entries']=['owner.json','gate-released.json','R-52-2.json' if family=='r48' else 'R-48-4.json'];accepted=False
            elif case=='no_gate':obs['queue_entries']=['owner.json','terminal.json'];accepted=False
            elif case=='no_result':obs['full2_result_exists']=False;accepted=False
            elif case=='live_chain':obs['processes'][0].update(exists=True,start_ticks='465769819',state='S');accepted=False
            elif case=='unknown_chain':obs['processes']=[];accepted=False
            elif case=='changed_hash':accepted=False
            elif case=='bad_transport':accepted=False
            raw=json.dumps(obs).encode();transport={'returncode':0,'stdout_sha256':hashlib.sha256(raw).hexdigest(),'stdout_bytes':len(raw),'source_sha256':m.OBSERVER_SHA}
            if case=='bad_transport':transport['returncode']=1
            Path(str(prefix)+'.stdout').write_bytes(b'{}' if case=='changed_hash' else raw)
            Path(str(prefix)+'.meta.json').write_text(json.dumps(transport))
            self.assertEqual(m.COMMAND[-1],'nice -n 10 /usr/bin/python3.14 -I -B - '+family)
            self.assertEqual(m.PREFIX.name,f'task-{family[1:]}-terminal-fetch-v1')
            with patch.object(m,'LEDGER',root):
                if accepted:self.assertEqual(m.load_observation(prefix),(raw,json.dumps(transport).encode()))
                else:
                    with self.assertRaises(ValueError):m.load_observation(prefix)
            self.assertEqual(sorted(p.name for p in root.iterdir()),[prefix.name+'.meta.json',prefix.name+'.stdout'])
    def test_fixed_family_preconditions(self):
        for family in ['r48','r52']:
            for case in ['terminal','queue_failed_before_family_build','pending','other_family','no_gate','no_result','live_chain','unknown_chain','changed_hash','bad_transport']:
                with self.subTest(family=family,case=case):self.run_case(family,case)
if __name__=='__main__':unittest.main()
