"""Tiny SYNTHETIC operational fixtures only; never SSH/native/research execution."""
import base64
import copy
import importlib.util
import io
import json
import os
from pathlib import Path
import shlex
import tempfile
import unittest
from contextlib import redirect_stdout
from unittest.mock import patch

SOURCE = Path(__file__).with_name('task-48-52-terminal-collector.remote.py')
m = None
if SOURCE.exists():
    spec = importlib.util.spec_from_file_location('r48_r52_capture', SOURCE)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)

PINS = {'r48': '0e075618ab6e28b0cf260f98feb7f80ee1ccf1f3',
        'r52': '9305d2e1f4f9999296cd4b4aaf55ea631fbca78e'}
TEST_NAMES = ('n16_train_full_keys_cuda.json', 'n16_train_full_keys_cpu.json',
              'n16_train_level7_cuda.json', 'n16_train_level7_cpu.json',
              'small_insecure_cuda.json', 'small_insecure_cpu.json',
              'iris_seal_2020_timing_cuda.json', 'iris_seal_2020_timing_cpu.json',
              'iris_mini_boot_ni_cuda.json', 'iris_mini_boot_ni_cpu.json')


class CollectorTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(m, 'finite R48/R52 collector has not been implemented')
        tmp = tempfile.TemporaryDirectory(prefix='synthetic-r48-r52-')
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name).resolve()
        self.runs = self.root / 'runs'
        self.state = self.runs / 'post-full-v1'
        self.trees = {family: str(self.root / ('clones/hekit-post-full-' + family))
                      for family in ('r48', 'r52')}
        for key, value in (('RUNS', str(self.runs)), ('STATE', str(self.state)), ('TREES', self.trees)):
            p = patch.object(m, key, value); p.start(); self.addCleanup(p.stop)
        self.put(self.state / 'owner.json', {'pid': 1739838, 'started': 1793929784.594468})
        self.gate = {'at': 1794096000.75, 'config': copy.deepcopy(m.EXPECTED_CONFIG)}
        self.put(self.state / 'gate-released.json', self.gate)

    def put(self, path, data):
        path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(data, (dict, list)):
            data = json.dumps(data).encode()
        elif isinstance(data, str):
            data = data.encode()
        path.write_bytes(data)
        return path

    def fixture_command(self, name, opt='nag'):
        # Independent literals from deployed queue; not the collector registry.
        if name == 'post-full-r48-build':
            return False, ['cargo', 'build', '-p', 'hekit-bench', '--features', 'cuda', '--release', '--bins', '--tests'], []
        if name == 'post-full-r52-build':
            return False, ['cargo', 'build', '--release', '-p', 'hekit-bench', '--features', 'parallel', '--bins'], []
        if name == 'R-48-1':
            return True, ['cargo', 'test', '-p', 'hekit-bench', '--features', 'cuda', '--release', '--test', 'mnist_cuda', '--test', 'iris_cuda', '--', '--ignored', '--nocapture', '--test-threads=1'], ['results/quadruped/raw/R-48-1/' + n for n in TEST_NAMES]
        if name.startswith('R-48-'):
            gpu = name in ('R-48-2', 'R-48-2-sgd')
            if name == 'R-48-2-sgd': opt = 'sgd'
            full = name == 'R-48-4'
            out = {'R-48-2': 'ablation_cuda', 'R-48-2-sgd': 'ablation_cuda_sgd',
                   'R-48-3': 'ablation_cpu_l7' + ('_sgd' if opt == 'sgd' else ''),
                   'R-48-4': 'ablation_cpu'}[name]
            out = 'results/quadruped/' + out + '.json'
            cmd = ['target/release/ablation', '--backend', 'cuda' if gpu else 'cpu', '--params', 'n16_train']
            cmd += [] if full else ['--top-level', '7']
            cmd += ['--optimizer', opt, '--iters', '3' if full else '5', '--warmup', '1', '--data', self.trees['r48'] + '/data/mnist', '--out', out]
            return gpu, cmd, [out]
        number = 2 if name == 'R-52-1' else 3
        out = f'results/quadruped/mnist196_emlp{number}_ni_b48_subset1008_seed0.json'
        cmd = ['target/release/mnist', '--mode', 'noninteractive', '--placement', 'when_needed', '--batch', '48', '--faithful', '--blocks', '64' if number == 2 else '128,64', '--hidden', '32', '--dataset-npy', 'results/quadruped/data/reboot_emlp1_full', '--indices', f'results/reboot_emlp{number}_seed0_steps1-21_indices.txt', '--init-weights', f'results/quadruped/raw/reboot_emlp23_full/init_weights_emlp{number}_seed0', '--init-weights-manifest', 'results/quadruped/raw/reboot_emlp23_full/config_values.json', '--epochs', '1', '--transpose', 'reuse', '--out', out]
        return False, cmd, [out]

    def terminal(self, name='R-48-2', rc=0, watchdog=False, status=None, opt='nag', log='SYNTHETIC log, not a measurement\n'):
        family = 'r48' if '48' in name else 'r52'
        gpu, cmd, outputs = self.fixture_command(name, opt)
        tree = self.trees[family]
        skips = [line for line in log.splitlines() if 'SKIP' in line]
        status = status or ('failed: RSS watchdog' if watchdog else 'skipped' if rc == 3 and skips else 'failed' if rc else 'done_with_skips' if skips else 'done')
        record = {'job': name, 'workdir': tree, 'command': cmd, 'rc': rc, 'wrapper_rc': 0,
                  'status': status, 'skip': bool(skips), 'skip_lines': skips,
                  'log': str(self.runs / 'logs' / (name + '.log')), 'evidence_lines': skips}
        meta = {'job': name, 'workdir': tree, 'command': shlex.join(cmd), 'exit_status': rc,
                'watchdog_killed': watchdog, 'start': '2026-11-08T00:00:00+00:00',
                'end': '2026-11-08T00:01:00+00:00', 'wall_s_runner': 60.1,
                'nice': 10, 'threads_cap': 12, 'gpu_job': gpu,
                'env_thread_caps': {n: '12' for n in ('OMP_NUM_THREADS', 'RAYON_NUM_THREADS', 'CARGO_BUILD_JOBS')},
                'hekit_synced_commit': ['commit ' + PINS[family][:7], 'full ' + PINS[family], 'dirty_files 0']}
        state = self.put(self.state / (name + '.json'), record)
        out = self.runs / 'out' / name
        self.put(out / 'meta.json', meta)
        self.put(out / 'cmd.txt', shlex.join(cmd) + ' \n')
        self.put(out / 'workdir.txt', tree + '\n')
        self.put(out / 'post.txt', 'date: 2026-11-08T00:01:00+00:00\nJOB_DONE rc=' + str(rc) + '\n')
        self.put(self.runs / 'logs' / (name + '.log'), log)
        return record, meta, state, out, [Path(tree) / p for p in outputs]

    def collect_ok(self, family):
        got = m.collect(family)
        self.assertEqual(got['exit_code'], 0, got.get('error'))
        self.assertEqual(got['kind'], 'operational_capture_not_scientific_result')
        self.assertIs(got['scientific_acceptance'], False)
        return got

    def raw(self, got, key):
        return base64.b64decode(got['files'][key]['base64'])

    def test_invalid_family_or_cli_arity_refuses_before_reads(self):
        # Break: an arbitrary family/path reaches the filesystem.
        for family in ('r47', '../r48', '', None):
            with self.subTest(family=family), patch.object(m, 'read_stable', side_effect=AssertionError('unexpected read')):
                self.assertEqual(m.collect(family)['exit_code'], 2)
        for args in ([], ['r48', 'r52'], ['--help'], ['/tmp/path']):
            with self.subTest(args=args), patch.object(m, 'read_stable', side_effect=AssertionError('unexpected read')), redirect_stdout(io.StringIO()) as stream:
                self.assertEqual(m.main(args), 2)
                self.assertIs(json.loads(stream.getvalue())['scientific_acceptance'], False)

    def test_missing_states_do_not_read_live_logs_or_outputs(self):
        # Break: missing state causes a speculative read of a growing job.
        path = self.runs / 'logs/R-48-2.log'; path.parent.mkdir(parents=True); os.mkfifo(path)
        got = self.collect_ok('r48')
        self.assertEqual(set(got['jobs']), {'post-full-r48-build','R-48-1','R-48-2','R-48-2-sgd','R-48-3','R-48-4'})
        self.assertTrue(all(v['capture_state'] == 'no_terminal_record_observed' for v in got['jobs'].values()))
        self.assertNotIn('R-48-2/program.log', got['files'])
        self.assertIn('R-48-2/state.json', got['missing_artifacts'])

    def test_all_r48_jobs_and_exact_ten_test_outputs(self):
        # Break: dropping a finite branch, or reading unregistered test output names.
        for name in ('post-full-r48-build','R-48-1','R-48-2','R-48-2-sgd','R-48-3','R-48-4'):
            _, _, _, _, outputs = self.terminal(name)
            for path in outputs: self.put(path, b'{"synthetic":true,"all_checks_pass":false}\n')
        self.put(self.state/'R-48-optimizer.json', {'cpu_optimizer':'nag'})
        extra = Path(self.trees['r48'])/'results/quadruped/raw/R-48-1/unregistered.json'; os.mkfifo(extra)
        got = self.collect_ok('r48')
        binaries = [k for k in got['files'] if '/binary/' in k]
        self.assertEqual(len(binaries), 14)
        self.assertEqual({k.split('/')[-1] for k in binaries if k.startswith('R-48-1/')}, set(TEST_NAMES))
        self.assertTrue(all(v['capture_state']=='terminal_artifacts_captured' for v in got['jobs'].values()))
        self.assertIn('R-48-optimizer.final.json', got['files'])
        self.assertFalse(any('R-47' in k or 'R-52' in k for k in got['files']))

    def test_selected_sgd_cpu_path_only_and_optimizer_retained(self):
        # Break: CPU SGD gets the NAG output or reconstructed optimizer state.
        _, _, _, _, outputs = self.terminal('R-48-3', opt='sgd')
        raw = b'{ "cpu_optimizer" : "sgd" }\n'
        self.put(self.state/'R-48-optimizer.json', raw)
        self.put(outputs[0], b'SYNTHETIC failed JSON bytes retained as-is')
        wrong = outputs[0].with_name('ablation_cpu_l7.json'); os.mkfifo(wrong)
        got = self.collect_ok('r48')
        self.assertEqual(self.raw(got,'R-48-optimizer.json'), raw)
        self.assertEqual(self.raw(got,'R-48-3/binary/ablation_cpu_l7_sgd.json'), b'SYNTHETIC failed JSON bytes retained as-is')

    def test_failed_watchdog_partial_and_failed_output_bytes_are_preserved(self):
        # Break: failed terminal jobs get dropped or their JSON marked accepted.
        for rc, watchdog, status in ((1,False,'failed'),(143,True,'failed: RSS watchdog'),(143,False,'partial'),(0,False,'failed: output checks/provenance')):
            with self.subTest(status=status):
                record, _, state, out, outputs = self.terminal('R-52-1',rc,watchdog,status)
                if status == 'failed: output checks/provenance':
                    record['output_error']='synthetic invalid checks'; self.put(state,record)
                for filename in ('partial.txt','projection.txt','partial-identities.json'):
                    self.put(out/filename, 'SYNTHETIC partial evidence\n')
                binary=b'{"all_checks_pass":false,"synthetic":true}\n'; self.put(outputs[0],binary)
                got=self.collect_ok('r52')
                self.assertEqual(got['jobs']['R-52-1']['runner_exit'],rc)
                self.assertEqual(got['jobs']['R-52-1']['queue_status'],status)
                self.assertEqual(self.raw(got,'R-52-1/binary/'+outputs[0].name),binary)
                for filename in ('partial.txt','projection.txt','partial-identities.json'):
                    self.assertIn('R-52-1/'+filename,got['files'])

    def test_r52_success_build_and_both_fixed_outputs(self):
        # Break: swapping eMLP2/eMLP3 argv/output identities.
        for name in ('post-full-r52-build','R-52-1','R-52-2'):
            _,_,_,_,outputs=self.terminal(name)
            for path in outputs:self.put(path,b'SYNTHETIC binary bytes')
        got=self.collect_ok('r52')
        self.assertEqual(set(got['jobs']),{'post-full-r52-build','R-52-1','R-52-2'})
        self.assertIn('R-52-1/binary/mnist196_emlp2_ni_b48_subset1008_seed0.json',got['files'])
        self.assertIn('R-52-2/binary/mnist196_emlp3_ni_b48_subset1008_seed0.json',got['files'])
        self.assertNotIn('R-48-optimizer.json',got['files'])

    def test_missing_binary_does_not_imply_success_or_not_run(self):
        # Break: absent output becomes invented empty result or a hard acquisition failure.
        self.terminal('R-48-1',rc=101)
        got=self.collect_ok('r48')
        missing=[k for k in got['missing_artifacts'] if k.startswith('R-48-1/binary/')]
        self.assertEqual(len(missing),10)
        self.assertEqual(got['jobs']['R-48-1']['runner_exit'],101)
        self.assertFalse(any('/binary/' in k for k in got['files']))

    def test_optional_terminal_summary_never_accepts_science(self):
        # Break: completed queue summary is promoted to scientific acceptance.
        raw=b'{ "status":"completed_with_failures", "jobs":[], "ended":1794096061 }\n'
        self.put(self.state/'terminal.json',raw)
        got=self.collect_ok('r52')
        self.assertEqual(self.raw(got,'terminal.json'),raw)
        self.assertIn('terminal.final.json',got['files'])

    def test_wrong_owner_gate_and_primitive_types_preserve_refusal_bytes(self):
        # Break: truthy bools, NaN or wrong gate/source association are accepted.
        for filename, obj in (
            ('owner.json',{'pid':True,'started':1793929784.594468}),
            ('owner.json',{'pid':1739838,'started':'1793929784.594468'}),
            ('gate-released.json',{'at':float('nan'),'config':self.gate['config']}),
            ('gate-released.json',{'at':1794096000,'config':{}}),
            ('gate-released.json',{'at':1793929780,'config':self.gate['config']}),
        ):
            with self.subTest(filename=filename,obj=obj):
                path=self.state/filename; before=path.read_bytes(); self.put(path,obj)
                got=m.collect('r48'); self.assertEqual(got['exit_code'],2); self.assertIn(filename,got['files'])
                self.assertFalse(any('/state.json' in k for k in got['files']))
                path.write_bytes(before)

    def test_bad_terminal_identity_argv_types_and_times_refuse_before_output(self):
        # Break: plausible unrelated command/path/pin or incoherent terminal status reaches output read.
        mutations=[('record','rc',True),('record','wrapper_rc','0'),('record','job','R-48-3'),
                   ('record','command',['sh','-c','anything']),('record','log','/tmp/log'),
                   ('record','skip','false'),('record','status','running'),
                   ('meta','exit_status',4),('meta','nice',True),('meta','threads_cap',12.0),
                   ('meta','gpu_job',1),('meta','watchdog_killed','false'),
                   ('meta','hekit_synced_commit',['full '+PINS['r48'],'dirty_files 1']),
                   ('meta','start','2026-11-07T23:59:59+00:00'),('meta','end','2026-11-07T23:00:00+00:00'),
                   ('meta','start','2026-11-08T00:00:00'),('meta','end','2099-01-01T00:00:00+00:00'),
                   ('meta','wall_s_runner',8000),('meta','env_thread_caps',{})]
        for kind, field, value in mutations:
            with self.subTest(kind=kind,field=field):
                record,meta,state,out,outputs=self.terminal()
                obj=record if kind=='record' else meta; obj[field]=value
                self.put(state if kind=='record' else out/'meta.json',obj)
                got=m.collect('r48'); self.assertEqual(got['exit_code'],2)
                self.assertIn('R-48-2/state.json',got['files'])
                self.assertFalse(any('/binary/' in k for k in got['files']))

    def test_cmd_workdir_post_mismatch_retained_before_log_reads(self):
        # Break: runner ancillary identity disagreement ignored before potentially live data read.
        for filename,text in (('cmd.txt','false\n'),('workdir.txt','/tmp/other\n'),
                              ('post.txt','date: 2026-11-08T00:01:00+00:00\nJOB_DONE rc=4\n'),
                              ('post.txt','date: 2026-11-08T00:01:00+00:00\nJOB_DONE rc=0\nJOB_DONE rc=0\n')):
            with self.subTest(filename=filename,text=text):
                _,_,_,out,_=self.terminal(); self.put(out/filename,text)
                got=m.collect('r48'); self.assertEqual(got['exit_code'],2)
                self.assertIn('R-48-2/'+filename,got['files'])
                self.assertNotIn('R-48-2/program.log',got['files'])

    def test_malformed_duplicate_and_recursive_operational_json_retained(self):
        # Break: parsing failure loses the raw offending file and earlier evidence.
        for raw in (b'{', b'{"job":"R-48-2","job":"other"}', b'{"n":NaN}', b'{"x":'+b'['*5000+b'0'+b']'*5000+b'}'):
            with self.subTest(raw=raw[:20]):
                _,_,state,_,_=self.terminal(); self.put(state,raw)
                got=m.collect('r48'); self.assertEqual(got['exit_code'],2)
                self.assertEqual(self.raw(got,'R-48-2/state.json'),raw)
                self.assertIn('owner.json',got['files'])

    def test_each_reread_mutation_preserves_both_observations(self):
        # Break: owner/gate/job/meta/optimizer/queue terminal changes escape final comparison.
        for target,key in (('owner.json','owner.final.json'),('gate-released.json','gate-released.final.json'),
                           ('R-48-optimizer.json','R-48-optimizer.final.json'),('terminal.json','terminal.final.json'),
                           ('R-48-2.json','R-48-2/state.final.json'),('meta.json','R-48-2/meta.final.json')):
            with self.subTest(target=target):
                self.put(self.state/'owner.json',{'pid':1739838,'started':1793929784.594468})
                self.put(self.state/'gate-released.json',self.gate)
                _,_,_,out,_=self.terminal()
                self.put(self.state/'R-48-optimizer.json',{'cpu_optimizer':'nag'})
                self.put(self.state/'terminal.json',{'status':'completed','jobs':[],'ended':1794096061})
                path=out/'meta.json' if target=='meta.json' else self.state/target
                original=m.read_stable; count=0
                def read(p,byte_limit=m.PER_FILE):
                    nonlocal count
                    if p==str(path):
                        count+=1
                        if count==2:path.write_bytes(path.read_bytes()+b'\n')
                    return original(p,byte_limit)
                with patch.object(m,'read_stable',side_effect=read):got=m.collect('r48')
                self.assertEqual(got['exit_code'],2)
                self.assertIn(key,got['files'])
                self.assertIn('owner.json',got['files'])

    def test_symlink_fifo_and_bounds_refuse_without_overread(self):
        # Break: unsafe special files block/read, or bounds don't count rereads.
        _,_,_,out,_=self.terminal()
        log=self.runs/'logs/R-48-2.log'; original=log.read_bytes()
        for kind in ('fifo','symlink'):
            log.unlink()
            if kind=='fifo':os.mkfifo(log)
            else:log.symlink_to(out/'cmd.txt')
            got=m.collect('r48'); self.assertEqual(got['exit_code'],2)
            self.assertNotIn('R-48-2/program.log',got['files'])
            log.unlink(); log.write_bytes(original)
        with patch.object(m,'PER_FILE',64):got=m.collect('r48')
        self.assertEqual(got['exit_code'],2)
        first=self.collect_ok('r48'); total=first['original_bytes_collected']
        with patch.object(m,'TOTAL',total-1):got=m.collect('r48')
        self.assertEqual(got['exit_code'],2)
        self.assertLessEqual(got['original_bytes_collected'],total-1)
        self.assertIn('owner.json',got['files'])

    def test_ancestor_symlink_and_wrong_owner_rejected_by_shared_reader(self):
        # Break: final-only nofollow or wrong UID permits acquisition.
        path=self.put(self.root/'real/file',b'SYNTHETIC')
        (self.root/'alias').symlink_to(path.parent,target_is_directory=True)
        with self.assertRaises((ValueError,OSError)):m.read_stable(str(self.root/'alias/file'))
        with patch.object(m.os,'getuid',return_value=os.getuid()+1):
            with self.assertRaises(ValueError):m.read_stable(str(path))

    def test_no_log_status_inconsistency_or_optimizer_mismatch(self):
        # Break: queue status/log contradictions or CPU optimizer mismatch silently accepted.
        for which in ('skip','watchdog','status','optimizer'):
            record,meta,state,out,_=self.terminal('R-48-3')
            self.put(self.state/'R-48-optimizer.json',{'cpu_optimizer':'sgd' if which=='optimizer' else 'nag'})
            if which=='skip':self.put(self.runs/'logs/R-48-3.log','SKIP: SYNTHETIC\n')
            if which=='watchdog':self.put(self.runs/'logs/R-48-3.log','WATCHDOG: SYNTHETIC\n')
            if which=='status':record['status']='failed'; self.put(state,record)
            got=m.collect('r48'); self.assertEqual(got['exit_code'],2,which)


    def test_gate_config_integer_identity_cannot_be_float(self):
        # Break: Python numeric equality hides a changed original config primitive type.
        gate=copy.deepcopy(self.gate)
        gate['config']['chain_pid']=1700979.0
        self.put(self.state/'gate-released.json',gate)
        got=m.collect('r48')
        self.assertEqual(got['exit_code'],2)
        self.assertIn('gate-released.json',got['files'])
        self.assertFalse(any('/state.json' in k for k in got['files']))

    def test_skip_oom_and_wrapper_status_are_operational_only(self):
        # Break: valid queue skip/OOM branches are rejected, or wrapper rc replaces child rc.
        for rc,status,log in ((3,'skipped','SKIP: SYNTHETIC capacity\n'),
                              (0,'done_with_skips','SKIP: SYNTHETIC test branch\n'),
                              (1,'failed: device OOM','Device error: SYNTHETIC out of memory\n')):
            with self.subTest(status=status):
                record,_,state,_,_=self.terminal('R-48-2',rc=rc,status=status,log=log)
                record['wrapper_rc']=-15; self.put(state,record)
                got=self.collect_ok('r48')
                self.assertEqual(got['jobs']['R-48-2']['runner_exit'],rc)
                self.assertEqual(got['jobs']['R-48-2']['wrapper_exit'],-15)
                self.assertEqual(got['jobs']['R-48-2']['queue_status'],status)
                self.assertIn('R-48-2/binary/ablation_cuda.json',got['missing_artifacts'])

    def test_main_valid_family_serializes_raw_failure_evidence(self):
        # Break: CLI drops the preserved operational payload on refusal.
        self.put(self.state/'owner.json',b'{broken SYNTHETIC JSON')
        with redirect_stdout(io.StringIO()) as output:
            code=m.main(['r52'])
        got=json.loads(output.getvalue())
        self.assertEqual(code,2)
        self.assertEqual(self.raw(got,'owner.json'),b'{broken SYNTHETIC JSON')
        self.assertIs(got['scientific_acceptance'],False)

    def test_missing_required_runner_metadata_keeps_state_without_binary_reads(self):
        # Break: an incomplete terminal runner yields success or touches scientific output.
        _,_,_,out,_=self.terminal('R-52-2')
        (out/'meta.json').unlink()
        got=m.collect('r52')
        self.assertEqual(got['exit_code'],2)
        self.assertIn('R-52-2/state.json',got['files'])
        self.assertFalse(any('/binary/' in k for k in got['files']))


if __name__=='__main__':unittest.main()
