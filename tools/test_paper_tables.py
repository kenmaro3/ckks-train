"""Task54 tables: FULL fixtures are SYNTHETIC, never measurements."""
import contextlib
import copy
import io
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import paper_tables as tables
from test_rq1_summary import synthetic_full

ROOT = Path(__file__).resolve().parents[1]


def archived_overlay(root, source=ROOT):
    """Link read-only archived inputs; new fixture/output paths stay in the temp tree."""
    (root / 'docs').symlink_to(source / 'docs', target_is_directory=True)
    for base in ('results', 'results/quadruped', 'results/quadruped/raw'):
        dest = root / base
        dest.mkdir(parents=True, exist_ok=True)
        for p in (source / base).iterdir():
            q = dest / p.name
            if (q.exists() or p.name in ('run_all', 'tables', 'R-41-FULL-1', 'R-41-FULL-2', 'R-47-3', 'R-52', 'R-48',
                                         'mnist196_emlp2_ni_b48_subset1008_seed0.json',
                                         'mnist196_emlp3_ni_b48_subset1008_seed0.json',
                                         'ablation_cuda.json', 'ablation_cuda_sgd.json', 'ablation_cpu_l7.json',
                                         'ablation_cpu_l7_sgd.json', 'ablation_cpu.json')):
                continue
            if p.is_file() or (p.is_dir() and base.endswith('/raw')) or p.name == 'reference':
                q.symlink_to(p, target_is_directory=p.is_dir())


def put_full(root, batch=64, mutate=None):
    doc = synthetic_full(batch)
    # Match only the fixture's machine/thread metadata to the historical timing reference.
    doc['provenance']['machine']['host'] = 'kmihara-Z790-S01'
    if mutate:
        mutate(doc)
    key = 'FULL-1' if batch == 64 else 'FULL-2'
    path = root / tables.RUN[key][0][0]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc))
    return path


def rows(table):
    return [[c.text for c in item] for kind, item in table.rows if kind == 'row']


R47_DIR = 'results/quadruped/raw/R-47-3/R-47-3-example'
R47_PIN = 'bd5cc493db166a282d26b05b1e069f2a0ee629ef'
R47_WD = '/home/kmihara/hekit-post-full-clones/hekit-post-full-r47'
R47_COMMAND = 'target/release/examples/batch_streams'
R47_LOG = """batch_streams: n16_train, d = 64, level 5, 4 products, 30 reps; CPU setup 1.2 s
pool release threshold: default (268435456 bytes)
[default] sequential x4, 1 stream:   median   40.000 ms  min   38.000  sd  1.000  (10.000 ms per product)
[default] batch x4, 2 streams:    median   30.000 ms  min   28.000  sd  2.000
[default] batch x4, 4 streams:    median   20.000 ms  min   18.000  sd  3.000
pool release threshold: max (18446744073709551615 bytes)
[max] sequential x4, 1 stream:   median   32.000 ms  min   30.000  sd  0.500  (8.000 ms per product)
[max] batch x4, 2 streams:    median   24.000 ms  min   22.000  sd  1.500
[max] batch x4, 4 streams:    median   16.000 ms  min   14.000  sd  2.500
"""


def put_r47(root, rc=0, watchdog=False, log=R47_LOG):
    """Tiny SYNTHETIC original-format runner/queue records; never measurements."""
    dest = root / R47_DIR
    dest.mkdir(parents=True, exist_ok=True)
    meta = {
        'job': 'R-47-3-example', 'workdir': R47_WD, 'command': R47_COMMAND,
        'hekit_synced_commit': ['commit bd5cc49', 'full ' + R47_PIN, 'dirty_files 0'],
        'threads_cap': 12, 'env_thread_caps': {name: '12' for name in
            ('OMP_NUM_THREADS', 'RAYON_NUM_THREADS', 'CARGO_BUILD_JOBS')},
        'nice': 10, 'gpu_job': True, 'start': '2026-11-08T10:00:00+09:00',
        'end': '2026-11-08T10:00:10+09:00', 'wall_s_runner': 10.1,
        'exit_status': rc, 'watchdog_killed': watchdog,
    }
    state = {'job': 'R-47-3-example', 'rc': rc, 'wrapper_rc': 0,
             'status': 'failed: RSS watchdog' if watchdog else 'failed' if rc else 'done',
             'skip': False, 'skip_lines': [], 'command': [R47_COMMAND],
             'workdir': R47_WD, 'log': '/home/kmihara/hekit-runs/logs/R-47-3-example.log',
             'evidence_lines': []}
    originals = {'program.log': log, 'cmd.txt': R47_COMMAND + ' \n',
                 'meta.json': json.dumps(meta), 'state.json': json.dumps(state),
                 'post.txt': 'date: 2026-11-08T10:00:10+09:00\nJOB_DONE rc=' + str(rc) + '\n'}
    for name, text in originals.items():
        (dest / name).write_text(text)
    return dest


R52_PIN = '9305d2e1f4f9999296cd4b4aaf55ea631fbca78e'
R52_WD = '/home/kmihara/hekit-post-full-clones/hekit-post-full-r52'


def put_r52(root, number=2, rc=0, watchdog=False, partial=False, output=True):
    """Tiny SYNTHETIC P52-shaped data and original-format records, never measurements."""
    key = f'R-52-{number-1}'
    rel = f'results/quadruped/mnist196_emlp{number}_ni_b48_subset1008_seed0.json'
    archive = root / 'results/quadruped/raw/R-52' / key
    archive.mkdir(parents=True, exist_ok=True)
    raw = root / 'results/quadruped/raw/reboot_emlp23_full'
    cfg = json.loads((ROOT / tables.REBOOT_23_CFG).read_bytes())
    if raw.is_symlink():
        raw.unlink()
    raw.mkdir(exist_ok=True)
    for model in (2, 3):
        seed = cfg['models'][f'emlp{model}']['per_seed']['0']
        for entry in seed['init_weights']:
            dest = raw / entry['file']
            dest.parent.mkdir(exist_ok=True)
            dest.write_bytes(b'SYNTHETIC weight payload')
            entry['sha256'] = hashlib.sha256(dest.read_bytes()).hexdigest()
        (raw / seed['indices_file']).write_bytes((ROOT / f'results/reboot_emlp{model}_seed0_steps1-21_indices.txt').read_bytes())
    (raw / 'config_values.json').write_text(json.dumps(cfg))
    data = root / 'results/quadruped/data/reboot_emlp1_full'
    data.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((ROOT / 'results/quadruped/data/reboot_emlp1_full/data_manifest.json').read_bytes())
    for item in manifest['files']:
        dest = data / item['file']
        dest.write_bytes(b'SYNTHETIC data payload')
        item['sha256_npy_file'] = hashlib.sha256(dest.read_bytes()).hexdigest()
    (data / 'data_manifest.json').write_text(json.dumps(manifest))
    argv = ['target/release/mnist', '--mode', 'noninteractive', '--placement', 'when_needed',
            '--batch', '48', '--faithful', '--blocks', '64' if number == 2 else '128,64',
            '--hidden', '32', '--dataset-npy', 'results/quadruped/data/reboot_emlp1_full',
            '--indices', f'results/reboot_emlp{number}_seed0_steps1-21_indices.txt',
            '--init-weights', f'results/quadruped/raw/reboot_emlp23_full/init_weights_emlp{number}_seed0',
            '--init-weights-manifest', 'results/quadruped/raw/reboot_emlp23_full/config_values.json',
            '--epochs', '1', '--transpose', 'reuse', '--out', rel]
    start = 1794099600.0  # 2026-11-08T10:00:00+09:00
    blocks = [64] if number == 2 else [128, 64]
    dims = [[196, 64, 16], [64, 32, 16]] if number == 2 else [[196,128,16], [128,64,16], [64,32,16]]
    boots = len(blocks)
    def counter(mult):
        return {'rotations': (60+69*boots)*mult, 'relins': (40+40*boots)*mult,
                'sse_switches': 2*boots*mult, 'key_switches': (100+111*boots)*mult,
                'bootstraps': boots*mult, 'ct_ct_tensor': 2*mult, 'ct_pt_mults': 3*mult, 'rescales': 4*mult}
    doc = {
      'provenance': {'written_by': 'the mnist binary (raw output, not assembled)', 'git_commit': R52_PIN,
        'git_modified_tracked_files': 'none', 'command_line': ' '.join(argv),
        'build': {'debug_assertions': False, 'package_version': '1.0.0', 'target_os': 'linux',
                  'target_arch': 'x86_64', 'features': {'parallel': True, 'cuda': False}},
        'parallel_feature': True, 'machine': {'host': 'synthetic-host', 'cpu': 'synthetic CPU', 'os': 'Linux'},
        'threads': 'rayon (feature parallel), RAYON_NUM_THREADS = 12, 32 logical CPUs',
        'started_unix': start+1, 'finished_unix': start+9},
      'all_checks_pass': rc == 0,
      'checks': {k: True for k in ('every_update_equals_prediction', 'twin_weights_within_2^-10',
                                  'test_acc_diff_within_0.005', 'handoff_err_within_2^-10')},
      'task': 'Task 52: ReBoot eMLP-2/3 block structure, non-interactive (bootstrapping)',
      'setup': {'params':'n16_boot', 'log_n':16, 'batch':48, 'seed':32, 'transpose_keys':'reuse', 'boot_ks':111,
                'init_weights': argv[argv.index('--init-weights')+1], 'biases':False,
                'eval_config': {'hoist':True, 'bsgs':True, 'lazy_relin':True, 'default':True}, 'lr_reference_on_mean_gradient':0.024,
                'lr_schedule':'constant'},
      'plan': {'stage_dims':dims, 'layout_d':256, 'lanes':64, 'micro_batches_per_update':1,
               'placement':'when_needed', 'updates':21, 'boot_ks_assumed':111,
               'start_level':20, 'bootstrap_output_level':21, 'levels_per_stage_step':7,
               'stage_nonboot_ks_per_update':[50,50] if number == 2 else [34,33,33], 'nonboot_ks_per_update':100,
               'weight_bootstraps_per_update_avg':0.0, 'handoff_bootstraps_per_update':boots,
               'velocity_reference_shift_rotations':10, 'predicted_ks_per_sample':counter(21)['key_switches']/1008,
               'boot_schedules':[[0]*21 for _ in dims]},
      'measured': {'updates':21, 'samples':1008, 'total':counter(21), 'per_update_first':counter(1),
                   'ks_per_sample_incl_bootstraps':counter(21)['key_switches']/1008,
                   'predicted_ks_this_run':counter(21)['key_switches'], 'measured_over_predicted':1.0,
                   'every_update_equals_prediction':True, 'handoff_bootstraps':boots*21},
      'time_s': {'keygen_total':1.0, 'train_total':2.0, 'bootstrap_total':1.0, 'wall_total':5.0,
                 'per_update':{'min':0.1,'median':0.1,'mean':0.1,'max':0.1}},
      'twin': {'max_abs_diff':0.0, 'log2_max_abs_diff':None,
               'max_abs_handoff_activation_per_block':[0.5]*len(blocks),
               'handoff_bootstraps':[{'slots_compared':1008*w, 'max_abs_input_twin':0.5, 'inputs_above_1':0,
                 'precondition_hit':False, 'max_abs_err_vs_twin':0.0, 'log2_max_abs_err_vs_twin':None} for w in blocks]},
      'evaluation':{'test_samples':10000, 'test_acc_encrypted_trained':0.9, 'test_acc_twin':0.9, 'test_acc_diff':0.0},
      'memory':{'peak_rss_bytes':1000},
    }
    if rc:
        doc['checks']['handoff_err_within_2^-10'] = False
    dest = root / rel
    if dest.is_symlink():
        dest.unlink()
    if output:
        dest.write_text(json.dumps(doc))
    elif dest.exists():
        dest.unlink()
    meta = {'job':key, 'host':'synthetic-host', 'workdir':R52_WD, 'command':' '.join(argv),
            'hekit_synced_commit':['commit 9305d2e','full '+R52_PIN,'dirty_files 0'],
            'threads_cap':12, 'env_thread_caps':{name:'12' for name in ('OMP_NUM_THREADS','RAYON_NUM_THREADS','CARGO_BUILD_JOBS')},
            'nice':10,'gpu_job':False,'start':'2026-11-08T10:00:00+09:00','end':'2026-11-08T10:00:10+09:00',
            'wall_s_runner':10.0,'exit_status':rc,'watchdog_killed':watchdog}
    state = {'job':key,'rc':rc,'wrapper_rc':0,'status':'failed: RSS watchdog' if watchdog else 'partial' if partial else 'failed' if rc else 'done',
             'skip':False,'skip_lines':[],'command':argv,'workdir':R52_WD,'log':f'/home/kmihara/hekit-runs/logs/{key}.log',
             'evidence_lines':[]}
    if output and not rc:
        state['output_sha256'] = hashlib.sha256(dest.read_bytes()).hexdigest()
    for name,text in {'meta.json':json.dumps(meta),'state.json':json.dumps(state),'cmd.txt':' '.join(argv)+' \n',
                      'post.txt':'date: 2026-11-08T10:00:10+09:00\nJOB_DONE rc='+str(rc)+'\n',
                      'program.log':'SYNTHETIC terminal log\n'}.items():
        (archive/name).write_text(text)
    if partial:
        limit = 12240.0 if number == 2 else 23040.0
        (archive/'partial.txt').write_text(f'partial: after >=5 updates; projected {2*limit:.0f} > {limit} s\n')
        (archive/'projection.txt').write_text(f'after 5 updates: {2*limit:.0f} s / 21 updates; limit {limit} s\n')
        (archive/'partial-identities.json').write_text('[[122, null], [123, "456"]]')
    return dest, archive


def refresh_r52_digest(dest, archive):
    state=json.loads((archive/'state.json').read_text())
    state['output_sha256']=hashlib.sha256(dest.read_bytes()).hexdigest()
    (archive/'state.json').write_text(json.dumps(state))


R48_PIN = '0e075618ab6e28b0cf260f98feb7f80ee1ccf1f3'
R48_WD = '/home/kmihara/hekit-post-full-clones/hekit-post-full-r48'


def put_r48_optimizer(root, optimizer):
    path = root / 'results/quadruped/raw/R-48/R-48-optimizer.json'
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({'cpu_optimizer': optimizer}))
    return path


def put_r48(root, job='R-48-2', optimizer='nag', rc=0, watchdog=False, skip=False, output=None, oom=False):
    """Tiny SYNTHETIC actual-P48-shaped ablation and original terminal records."""
    if output is None:
        output = rc == 0
    gpu = job in ('R-48-2', 'R-48-2-sgd')
    full = job == 'R-48-4'
    if job == 'R-48-2-sgd':
        optimizer = 'sgd'
    name = ('ablation_cuda' + ('_sgd' if optimizer == 'sgd' else '') if gpu else
            'ablation_cpu' if full else 'ablation_cpu_l7' + ('_sgd' if optimizer == 'sgd' else ''))
    rel = f'results/quadruped/{name}.json'
    argv = ['target/release/ablation', '--backend', 'cuda' if gpu else 'cpu', '--params', 'n16_train',
            *([] if full else ['--top-level', '7']), '--optimizer', optimizer, '--iters', '3' if full else '5',
            '--warmup', '1', '--data', R48_WD + '/data/mnist', '--out', rel]
    doc = json.loads((ROOT / 'results/ablation_cpu_smoke_small.json').read_bytes())
    doc['provenance'].update(written_by='the ablation binary (raw output, not assembled)',git_commit=R48_PIN,
        git_modified_tracked_files='none',command_line=' '.join(argv),parallel_feature=True,
        build={'debug_assertions':False,'package_version':'0.0.0','target_os':'linux','target_arch':'x86_64',
               'features':{'parallel':True,'cuda':True}},
        machine={'cpu':'synthetic CPU','host':'synthetic-host','os':'Linux','logical_cpus':32,'memory_bytes':'1000'},
        threads='rayon (feature parallel), RAYON_NUM_THREADS = 12, 32 logical CPUs',
        load_start='1.0 2.0 3.0 1/100 1',load_end='2.0 3.0 4.0 1/100 1',
        started_unix=1794099601.0,finished_unix=1794099609.0,
        gpu={'name':'SYNTHETIC GPU','free_mib_at_start':3000.0,'total_mib':24000.0} if gpu else None)
    doc['backend']['options'].update(backend='cuda' if gpu else 'cpu',
        eval_config={'hoist':True,'bsgs':True,'lazy_relin':True,'default':True},
        top_level=None if full else 7,matvec='sequential',gpu_pool='default (256 MiB release threshold)')
    doc['backend']['device'] = 'SYNTHETIC GPU' if gpu else 'cpu'
    doc['setup'].update(params='n16_train',log_n=16,max_level=31,start_level=31 if full else 7,
        level_capped=not full,dims=[784,32,10],layout_d=1024,lanes_b=32,optimizer=optimizer,
        lr=0.001,momentum=0.9,seed=48,data='MNIST training images (first in file order)',
        transpose_keys='reuse (mlp_key_pool)',matvec_dims_per_step=[32,16,16],
        warmup_iterations=1,timed_iterations=3 if full else 5)
    for i,row in enumerate(doc['configs']):
        row['counter_per_iteration']['rotations'] += 400
        row['counter_per_iteration']['key_switches'] += 400
        row['closed_form']['rotations'] += 400
        median = 1.0 + i/10
        row.update(step_s=[median+j/10 for j in ([-1,0,1] if full else [-2,-1,0,1,2])],
                   step_s_median=median,refresh_s_median=0.2,iteration_s_median=median+0.2,
                   step_time_over_default=median,step_speedup_over_all_off=1.7/median)
    doc['machine'].update(threads='rayon (parallel feature), RAYON_NUM_THREADS=12',wall_total_s=9.0)
    dest=root/rel
    if dest.is_symlink():dest.unlink()
    if output:dest.write_text(json.dumps(doc))
    elif dest.exists():dest.unlink()
    archive=root/'results/quadruped/raw/R-48'/job
    archive.mkdir(parents=True,exist_ok=True)
    need = dict(doc['backend']['gpu_need_estimate'])
    need['total_mib'] = 2456.0 if optimizer == 'nag' else 2072.0
    log = (f'ablation: SKIP ablation: needs {need["total_mib"]:.1f} MiB of GPU memory ({json.dumps(need)}), '
           'only 1000.0 MiB free; GPU run unmeasured\n' if skip else
           'device error: synthetic out of memory\n' if oom else 'SYNTHETIC ablation terminal log\n')
    meta={'job':job,'host':'synthetic-host','workdir':R48_WD,'command':' '.join(argv),
          'hekit_synced_commit':['commit 0e07561','full '+R48_PIN,'dirty_files 0'],
          'threads_cap':12,'env_thread_caps':{name:'12' for name in ('OMP_NUM_THREADS','RAYON_NUM_THREADS','CARGO_BUILD_JOBS')},
          'nice':10,'gpu_job':gpu,'start':'2026-11-08T10:00:00+09:00','end':'2026-11-08T10:00:10+09:00',
          'wall_s_runner':10.0,'exit_status':rc,'watchdog_killed':watchdog}
    state={'job':job,'rc':rc,'wrapper_rc':0,'status':'failed: RSS watchdog' if watchdog else
           'failed: device OOM' if oom and rc else 'skipped' if rc==3 and skip else 'failed' if rc else 'done',
           'skip':skip,'skip_lines':log.splitlines() if skip else [],'command':argv,'workdir':R48_WD,
           'log':f'/home/kmihara/hekit-runs/logs/{job}.log','evidence_lines':[]}
    if output and rc==0:state['output_sha256']=hashlib.sha256(dest.read_bytes()).hexdigest()
    for name,text in {'program.log':log,'meta.json':json.dumps(meta),'state.json':json.dumps(state),
                      'cmd.txt':' '.join(argv)+' \n','post.txt':'date: 2026-11-08T10:00:10+09:00\nJOB_DONE rc='+str(rc)+'\n'}.items():
        (archive/name).write_text(text)
    return dest,archive


class PaperTablesTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        archived_overlay(self.root)

    def build(self, fn):
        store = tables.Store(self.root)
        return fn(store, tables.Inputs(store))

    def test_interactive_reference_never_populates_ni_d_or_ways(self):
        # Break caught: labeling the independent interactive run as NI-derived way d.
        t = self.build(tables.table_rq1)
        r = next(r for r in rows(t) if r[0] == 'interactive')
        self.assertEqual(r[9:14], ['—'] * 5)
        self.assertTrue(any('interactive' in n.text.lower() and '0.564' in n.text for n in t.notes))

    def test_synthetic_full_rows_and_accuracy_gate(self):
        # Break caught: untested populated FULL branch, shifted columns or wrong B48 label.
        put_full(self.root)
        put_full(self.root, 48)
        t = self.build(tables.table_rq1)
        one, two = [r for r in rows(t) if r[0] in ('FULL-1', 'FULL-2')]
        self.assertEqual(one[9:13], ['0.234', '0.389', '0.399', '0.405'])
        self.assertEqual(two[3], '48 in 64')
        acc = rows(self.build(tables.table_accuracy))
        self.assertEqual(next(r for r in acc if r[0].startswith('hekit encrypted, B = 48'))[-1], 'PASS')
        wall = rows(self.build(tables.table_wallclock))
        self.assertEqual(next(r for r in wall if r[0] == 'FULL-2')[2], '48 in 64')
        self.assertEqual(next(r for r in wall if r[0] == 'FULL-1')[6], 'not measured')

    def test_present_full1_accuracy_renders_pdftex_safe_non_gate(self):
        # Break caught: populated B64 accuracy emits unsupported raw U+2260 to pdfLaTeX.
        put_full(self.root)
        with contextlib.redirect_stderr(io.StringIO()), contextlib.redirect_stdout(io.StringIO()) as output:
            rc = tables.main(['--root', str(self.root)])
        self.assertEqual(rc, 0, output.getvalue())
        tex = (self.root / 'paper/tables/accuracy_vs_reboot.tex').read_text()
        row = next(line for line in tex.splitlines()
                   if line.startswith('hekit encrypted, B = 64, WN &'))
        self.assertIn('& 89.00 &', row)
        self.assertIn(r'not the gate (B $\neq$ 48)', row)
        self.assertNotIn('≠', tex)
        md = (self.root / 'results/tables/accuracy_vs_reboot.md').read_text()
        self.assertIn('not the gate (B ≠ 48)', md)

    def test_full_parameter_error_is_sourced_diagnostic_not_accuracy_gate(self):
        # Distinct synthetic error catches a copied/hardcoded historical value.
        def error_fixture(doc):
            doc['twin'].update(max_abs_diff=0.125, log2_max_abs_diff=-3.0,
                               max_abs_w_diff_per_layer=[0.125, 0.0625],
                               max_abs_b_diff_per_layer=[0.0, 0.0])
        path = put_full(self.root, mutate=error_fixture)
        table = self.build(tables.table_accuracy)
        notes = [n for n in table.notes if 'final parameter error' in n.text]
        self.assertEqual(len(notes), 1)  # Missing FULL-2 must not inherit FULL-1.
        self.assertIn('FULL-1 final parameter error: 0.125', notes[0].text)
        self.assertIn('not a per-bootstrap precision or accuracy gate', notes[0].text)
        self.assertEqual(notes[0].sources,
                         [(path.relative_to(self.root).as_posix(), 'twin.max_abs_diff')])
        full1 = next(r for r in rows(table) if r[0].startswith('hekit encrypted, B = 64'))
        self.assertEqual(full1[-1], 'not the gate (B ≠ 48)')

    def test_each_consumer_rejects_malformed_full(self):
        # Break caught: wall-clock accepting truncated/invalid FULL despite RQ1 validation.
        put_full(self.root, mutate=lambda d: d['training'].update(samples=1))
        for fn in (tables.table_rq1, tables.table_accuracy, tables.table_wallclock):
            with self.subTest(table=fn.__name__), self.assertRaises(tables.TableError):
                self.build(fn)

    def test_failed_full_writes_no_partial_files(self):
        # Break caught: writing early tables before discovering malformed later input.
        p = put_full(self.root)
        p.write_text('{"provenance":')
        with contextlib.redirect_stderr(io.StringIO()), contextlib.redirect_stdout(io.StringIO()):
            rc = tables.main(['--root', str(self.root)])
        self.assertEqual(rc, 2)
        self.assertFalse((self.root / 'paper').exists())
        self.assertFalse((self.root / 'results/tables').exists())

    def test_same_cpu_timing_rejects_thread_mismatch(self):
        # Break caught: creating a 12-thread ratio from a 1-thread hekit result.
        put_full(self.root, mutate=lambda d: d['provenance'].update(
            threads='rayon (feature parallel), RAYON_NUM_THREADS = 1, 32 logical CPUs'))
        with self.assertRaises(tables.TableError):
            self.build(tables.table_wallclock)

    def test_zero_samples_fail_as_table_error(self):
        # Break caught: divide-by-zero escaping validation in ordinary NI rows.
        s = tables.Store(self.root)
        rel = tables.RUN['F1'][0][0]
        d = copy.deepcopy(s.json(rel))
        d['training']['samples'] = 0
        s.cache[rel] = d
        with self.assertRaises(tables.TableError):
            tables.table_rq1(s, tables.Inputs(s))

    def test_twin_comparison_uses_the_same_seed_subset(self):
        # Break caught: comparing one available twin against all ten ReBoot reference seeds.
        store = tables.Store(self.root)
        inp = tables.Inputs(store)
        original = inp.simple
        inp.simple = lambda canonical, run_id: (
            None if run_id.startswith('plain196_emlp1_s') and not run_id.endswith('_s0')
            else original(canonical, run_id))
        t = tables.table_accuracy(store, inp)
        r = next(r for r in rows(t) if r[0].startswith('hekit plaintext twin (ReBoot weights'))
        self.assertEqual(r[5], '90.41 (n = 1)')

    def test_inline_caveat_table_is_readable_prose_in_tex(self):
        # Break caught: inline Markdown pipes leaking unchanged into a PDF note.
        t = self.build(tables.table_ablation)
        tex = tables.render_tex(t, 'SYNTHETIC TEST rendering of archived input')
        notes = tex.split(r'\begin{minipage}', 1)[1]
        self.assertNotIn('|---|', notes)
        self.assertNotIn('| form |', notes)
        for label in ('Definition of', 'hoist changes no', 'gradient'):
            self.assertIn(label, notes)
        self.assertIn('default BSGS', notes)
        self.assertIn('13', notes)

    def test_quick_tables_do_not_validate_or_label_smokes_as_full(self):
        # Break caught: treating q_ subset IDs as absent or full training results.
        p = self.root / 'results/run_all/quick/q_ni196_wn_b64_sub128_s0.json'
        p.parent.mkdir(parents=True)
        d = synthetic_full()
        d['training'].update(samples=128, updates=2)
        p.write_text(json.dumps(d))
        s = tables.Store(self.root)
        inp = tables.Inputs(s, 'quick')
        self.assertIsNone(inp.path('FULL-1'))
        with contextlib.redirect_stderr(io.StringIO()), contextlib.redirect_stdout(io.StringIO()) as output:
            rc = tables.main(['--root', str(self.root), '--from-run-all', 'quick'])
        self.assertEqual(rc, 0, output.getvalue())
        self.assertFalse((self.root / 'paper').exists())
        quick = (self.root / 'results/run_all/quick/tables/md/quick_coverage.md').read_text()
        self.assertIn('q_ni196_wn_b64_sub128_s0', quick)
        self.assertIn('pipeline', quick.lower())
        self.assertNotIn('PASS', quick)

    def test_compatibility_entrypoint_regenerates_identical_canonical_tables(self):
        # Break caught: legacy tools/tables.py path missing or diverging from the authority.
        p = subprocess.run([sys.executable, str(ROOT / 'tools/tables.py'), '--root', str(self.root)],
                           capture_output=True, text=True)
        self.assertEqual(p.returncode, 0, p.stderr)
        self.assertTrue((self.root / 'paper/tables/rq1_ks.tex').is_file())

    def test_new_full_directory_rejects_runner_only_and_annotated_provenance(self):
        # Break caught: using wrapper metadata or historical annotations as a new writer.
        p = self.root / 'results/run_all/full/ni196_wn_b64_full_s0.json'
        p.parent.mkdir(parents=True)
        for provenance in ({}, {'written_by': 'scripts/run_all.sh', 'git_commit': '1'*40},
                           {'written_by': 'the mnist binary (raw output, not assembled)',
                            'git_commit': '1'*40, 'note': 'hand-added'}):
            with self.subTest(provenance=provenance):
                p.write_text(json.dumps({'provenance': provenance}))
                s = tables.Store(self.root)
                with self.assertRaises(tables.TableError):
                    tables.Inputs(s, 'full').path('FULL-1')

    def test_ablation_gpu_cpu_pair_requires_the_same_optimizer_level_layout_and_batch(self):
        # Break caught: presenting NAG CPU against SGD GPU, or unlike layouts/levels.
        base = json.loads((ROOT / 'results/ablation_cpu_smoke_small.json').read_text())
        for field, value in (('optimizer', 'sgd'), ('start_level', 6), ('layout_d', 512), ('lanes_b', 16)):
            with self.subTest(field=field):
                gpu, cpu = copy.deepcopy(base), copy.deepcopy(base)
                gpu['backend']['device'] = 'cuda'
                for doc in (gpu, cpu):
                    doc['setup'].update(params='n16_train', start_level=7)
                    doc['backend']['options']['top_level'] = 7
                gpu['setup'][field] = value
                for name, doc in (('ablation_cuda.json', gpu), ('ablation_cpu_l7.json', cpu)):
                    (self.root / 'results/quadruped' / name).write_text(json.dumps(doc))
                with self.assertRaises(tables.TableError):
                    self.build(tables.table_ablation)


    def test_r47_absent_keeps_existing_generated_bytes(self):
        # Break caught: changing any historical table when canonical R47 originals are absent.
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(tables.main(['--root', str(self.root)]), 0)
        # Frozen bytes at BASE 1a59ecea; independent of a future accepted R47 archive
        # and its regenerated tables. The real-repository --check covers every output.
        expected = {
            'paper/tables/gpu_vs_cpu_ops.tex': 'c9104367559251ee43275272c219c2e632b59225c1d56eda7170422afe3043cf',
            'results/tables/gpu_vs_cpu_ops.md': 'af5f19dd3f4cc993b43643a49c3bcbc0600854ddee638adfa402f4fffd6ff242',
        }
        for rel, digest in expected.items():
            self.assertEqual(hashlib.sha256((self.root / rel).read_bytes()).hexdigest(), digest)
        gpu_sources = ''.join(line for line in
            (self.root / 'results/tables/sources.tsv').read_text().splitlines(keepends=True)
            if line.startswith('gpu_vs_cpu_ops\t'))
        self.assertEqual(hashlib.sha256(gpu_sources.encode()).hexdigest(),
                         '4b72fd7a51b2d88682c675aff653c736a8dac33418de08079f4d1cafb5207c80')

    def test_r47_success_six_rows_keep_statistics_and_actual_line_sources(self):
        # Break caught: leaving a pending row, mixing thresholds, losing min/SD or using old ratios.
        dest = put_r47(self.root)
        before = {p.name: p.read_bytes() for p in dest.iterdir()}
        t = self.build(tables.table_gpu)
        r47 = [r for r in rows(t) if 'R-47-3' in r[0]]
        self.assertEqual(len(r47), 6)
        self.assertEqual([r[2] for r in r47], ['40.000', '30.000', '20.000', '32.000', '24.000', '16.000'])
        self.assertTrue(all(r[3:] == ['—', '—', '—', 'none published', '—'] for r in r47))
        self.assertIn('default', r47[0][0])
        self.assertIn('max', r47[3][0])
        self.assertIn('Criterion', t.title)
        self.assertIn('example', t.title)
        md = tables.render_md(t, 'synthetic/')
        self.assertNotIn('pending (R-47-3)', md)
        self.assertIn('min 38.000 ms; SD 1.000 ms; 30 within-run repetitions', md)
        self.assertIn('not 30 independent seeds', md)
        self.assertIn('not end-to-end training', md)
        self.assertIn('not GPU peak', md)
        sources = '\n'.join(tables.source_lines(t))
        for n in (3, 4, 5, 7, 8, 9):
            self.assertIn(R47_DIR + '/program.log\tline ' + str(n), sources)
        self.assertIn(R47_DIR + '/meta.json\texit_status', sources)
        self.assertIn(R47_DIR + '/post.txt\tline 2', sources)
        self.assertIn('capture-time SHA256', md)
        self.assertIn('not historical execution proof', md)
        for name, raw in before.items():
            self.assertIn(hashlib.sha256(raw).hexdigest(), md)
        new_rows = [item for kind, item in t.rows if kind == 'row' and 'R-47-3' in item[0].text]
        self.assertIn((R47_DIR + '/program.log', 'line 1'), new_rows[0][1].sources)
        self.assertEqual(before, {p.name: p.read_bytes() for p in dest.iterdir()})

    def test_r47_failure_discards_partial_success_timings(self):
        # Break caught: accepting timing lines printed before a later assertion/watchdog failure.
        for rc, watchdog in ((101, False), (143, True), (0, True), (3, False)):
            with self.subTest(rc=rc, watchdog=watchdog):
                put_r47(self.root, rc, watchdog, '\n'.join(R47_LOG.splitlines()[:4]) + '\npanic: synthetic failure\n')
                t = self.build(tables.table_gpu)
                new = [r for r in rows(t) if 'R-47-3' in r[0]]
                self.assertEqual(len(new), 1)
                self.assertIn('failed/unmeasured', new[0][2])
                self.assertNotIn('40.000', tables.render_md(t, 'synthetic/'))
                self.assertNotIn('Memory SKIP', tables.render_md(t, 'synthetic/'))
                self.assertIn(R47_DIR + '/meta.json', '\n'.join(tables.source_lines(t)))

    def test_r47_rejects_partial_originals_and_alias_only_is_pending(self):
        # Break caught: silently selecting an alias or historical data for a partial canonical job.
        alias = self.root / 'results/quadruped/batch_streams_r47_3.log'
        # Overlay may link a future alias; remove the temp link before fixture writes.
        if alias.is_symlink():
            alias.unlink()
        alias.write_text(R47_LOG)
        self.assertTrue(any('pending (R-47-3)' in r for r in rows(self.build(tables.table_gpu))))
        dest = put_r47(self.root)
        for name in ('program.log', 'meta.json', 'post.txt', 'cmd.txt', 'state.json'):
            with self.subTest(name=name):
                original = (dest / name).read_bytes()
                (dest / name).unlink()
                with self.assertRaises(tables.TableError):
                    self.build(tables.table_gpu)
                (dest / name).write_bytes(original)

    def test_r47_rejects_wrong_identity_or_inconsistent_terminal_provenance(self):
        # Break caught: accepting another pin/job/command or untrustworthy terminal records.
        mutations = [
            ('meta.json', 'job', 'R-47-3-nsys'),
            ('meta.json', 'hekit_synced_commit', ['commit bd5cc49', 'full ' + 'a'*40, 'dirty_files 0']),
            ('meta.json', 'hekit_synced_commit', ['commit bd5cc49', 'full ' + R47_PIN, 'dirty_files 1']),
            ('meta.json', 'hekit_synced_commit', ['commit bd5cc49', 'full ' + R47_PIN, 'dirty_files 0', 'dirty_files 1']),
            ('meta.json', 'workdir', '/tmp/other'), ('meta.json', 'command', 'nsys ' + R47_COMMAND),
            ('meta.json', 'threads_cap', 1), ('meta.json', 'nice', 0), ('meta.json', 'gpu_job', False),
            ('meta.json', 'env_thread_caps', {}), ('meta.json', 'start', '2026-11-08T10:00:00'),
            ('meta.json', 'end', '2026-11-08T09:59:00+09:00'),
            ('meta.json', 'wall_s_runner', 1000), ('meta.json', 'watchdog_killed', 'false'),
            ('meta.json', 'exit_status', 1), ('state.json', 'rc', 1),
            ('state.json', 'status', 'failed'), ('state.json', 'command', ['other']),
            ('state.json', 'workdir', '/tmp/other'), ('state.json', 'job', 'R-47-3-nsys'),
            ('state.json', 'wrapper_rc', '0'), ('state.json', 'skip', True),
        ]
        for name, key, value in mutations:
            with self.subTest(name=name, key=key, value=value):
                dest = put_r47(self.root)
                doc = json.loads((dest / name).read_text())
                doc[key] = value
                (dest / name).write_text(json.dumps(doc))
                with self.assertRaises(tables.TableError):
                    self.build(tables.table_gpu)
        for name, text in [('cmd.txt', 'nsys ' + R47_COMMAND), ('post.txt', 'JOB_DONE rc=1\n'),
                           ('post.txt', 'JOB_DONE rc=0\nJOB_DONE rc=0\n'),
                           ('meta.json', '{'), ('state.json', '{}')]:
            with self.subTest(name=name, text=text):
                dest = put_r47(self.root)
                (dest / name).write_text(text)
                with self.assertRaises(tables.TableError):
                    self.build(tables.table_gpu)

    def test_r47_success_rejects_wrong_incomplete_duplicate_or_invalid_timings(self):
        # Break caught: plausible metrics from malformed logs or the separate five-rep profile.
        invalid = [
            R47_LOG.replace('30 reps', '5 reps'), R47_LOG.replace('d = 64', 'd = 32'),
            R47_LOG + R47_LOG.splitlines()[0] + '\n',
            R47_LOG + R47_LOG.splitlines()[3] + '\n',
            R47_LOG.replace(R47_LOG.splitlines()[4] + '\n', ''),
            R47_LOG.replace('268435456 bytes', '0 bytes'),
            R47_LOG.replace('[max] batch x4, 4', '[max] batch x4, 3'),
            R47_LOG.replace('30.000 ms', 'nan ms'),
            R47_LOG.replace('30.000 ms', 'inf ms'),
            R47_LOG.replace('30.000 ms', '-30.000 ms'),
            R47_LOG.replace('30.000 ms', '0.000 ms'),
            R47_LOG.replace('min   28.000', 'min   38.000'),
            R47_LOG.replace('sd  2.000', 'sd -2.000'),
            R47_LOG.replace('(10.000 ms per product)', '(11.000 ms per product)'),
            R47_LOG + 'SKIP: synthetic capacity\n',
            R47_LOG + '[other] batch x4, 4 streams: median 1.000 ms min 1.000 sd 0.000\n',
            R47_LOG + 'WATCHDOG: tree RSS 100 kB > 90 kB, killing own job tree\n',
        ]
        for case, log in enumerate(invalid):
            with self.subTest(case=case):
                put_r47(self.root, log=log)
                with self.assertRaises(tables.TableError):
                    self.build(tables.table_gpu)

    def test_r47_malformed_input_writes_no_generated_files_and_quick_ignores_it(self):
        # Break caught: a late R47 parse failure leaving plausible partial paper output.
        dest = put_r47(self.root, log=R47_LOG.replace('30 reps', '5 reps'))
        with contextlib.redirect_stderr(io.StringIO()), contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(tables.main(['--root', str(self.root)]), 2)
            self.assertFalse((self.root / 'paper').exists())
            self.assertEqual(tables.main(['--root', str(self.root), '--from-run-all', 'quick']), 0)
        self.assertTrue(dest.is_dir())

    def test_r47_compatibility_entrypoint_renders_same_success_table(self):
        # Break caught: exposing the R47 data only through one CLI, or dropping it in full mode.
        put_r47(self.root)
        for args in ([], ['--from-run-all', 'full']):
            p = subprocess.run([sys.executable, str(ROOT / 'tools/tables.py'), '--root', str(self.root),
                                '--sources', 'gpu_vs_cpu_ops', *args], capture_output=True, text=True)
            self.assertEqual(p.returncode, 0, p.stderr)
            self.assertIn(R47_DIR + '/program.log\tline 9', p.stdout)

    def test_r47_overlay_excludes_future_archives_from_fixture_writes(self):
        # Break caught: future actual R47 archive symlinked into a fixture then overwritten.
        with tempfile.TemporaryDirectory() as archived, tempfile.TemporaryDirectory() as fixture:
            archived = Path(archived)
            fixture = Path(fixture)
            for directory in ('docs', 'results/quadruped/raw/R-47-3'):
                (archived / directory).mkdir(parents=True, exist_ok=True)
            sentinel = archived / 'results/quadruped/raw/R-47-3/original.log'
            sentinel.write_text('SYNTHETIC original')
            archived_overlay(fixture, source=archived)
            dest = put_r47(fixture)
            self.assertFalse((fixture / 'results/quadruped/raw/R-47-3').is_symlink())
            self.assertNotEqual(dest.resolve(), archived / R47_DIR)
            self.assertEqual(sentinel.read_text(), 'SYNTHETIC original')


    def test_r52_success_models_validate_original_schema_and_seed_handoff_sources(self):
        # Break caught: raw two-number intake, wrong block schema, or hidden seed/precondition scope.
        for number in (2,3):
            dest, archive = put_r52(self.root, number)
            doc=json.loads(dest.read_text())
            doc['twin']['handoff_bootstraps'][0].update(max_abs_input_twin=1.2,inputs_above_1=2,precondition_hit=True)
            doc['twin']['max_abs_handoff_activation_per_block'][0]=1.2
            dest.write_text(json.dumps(doc)); refresh_r52_digest(dest,archive)
        t=self.build(tables.table_accuracy)
        new=[r for r in rows(t) if r[0].startswith('hekit encrypted eMLP-')]
        self.assertEqual(len(new),2)
        for r in new:
            self.assertEqual(r[2],'1 (baseline seed 0; RNG 32)')
            self.assertEqual(r[3:5],['90.00','90.00'])
        md=tables.render_md(t,'synthetic/')
        self.assertIn('21 updates; 1008 samples',md)
        self.assertIn('inputs above 1: 2',md)
        self.assertIn('not independent seeds',md)
        self.assertIn('historical input-byte proof',md)
        src='\n'.join(tables.source_lines(t))
        self.assertIn('twin.handoff_bootstraps[0].slots_compared',src)
        self.assertIn('raw/R-52/R-52-1/post.txt\tline 2',src)
        self.assertIn('setup.seed',src)

    def test_r52_failed_watchdog_partial_never_publish_saved_success_accuracy(self):
        # Break caught: P52 writes failed-check JSON before exiting nonzero.
        for rc,watchdog,partial,output in ((1,False,False,True),(143,True,False,True),
                                          (0,True,False,True),(1,False,False,False),(143,False,True,False)):
            with self.subTest(rc=rc,watchdog=watchdog,partial=partial):
                dest,archive=put_r52(self.root,rc=rc,watchdog=watchdog,partial=partial,output=output)
                t=self.build(tables.table_accuracy)
                r=next(r for r in rows(t) if r[0].startswith('hekit encrypted eMLP-2,'))
                self.assertIn('partial/unmeasured' if partial else 'failed/unmeasured',r[3])
                self.assertEqual(r[4],'—')
                self.assertNotEqual(r[7],'twin check')
                for name in ('partial.txt','projection.txt','partial-identities.json'):
                    if (archive/name).exists(): (archive/name).unlink()

    def test_r52_rejects_missing_partial_or_inconsistent_original_terminal(self):
        # Break caught: binary-only success or incomplete/contradictory terminal association.
        for name in ('program.log','meta.json','post.txt','cmd.txt','state.json'):
            dest,archive=put_r52(self.root)
            (archive/name).unlink()
            with self.subTest(name=name),self.assertRaises(tables.TableError): self.build(tables.table_accuracy)
        for name,key,value in [('meta.json','hekit_synced_commit',['full '+'a'*40]),('meta.json','threads_cap',1),
            ('meta.json','nice',0),('meta.json','gpu_job',True),('meta.json','exit_status',1),
            ('meta.json','start','2026-11-08T10:00:00'),('meta.json','workdir','/tmp/wrong'),
            ('state.json','rc',1),('state.json','status','failed'),('state.json','output_sha256','a'*64),
            ('state.json','command',['other']),('state.json','job','R-52-2')]:
            dest,archive=put_r52(self.root)
            d=json.loads((archive/name).read_text());d[key]=value;(archive/name).write_text(json.dumps(d))
            with self.subTest(name=name,key=key),self.assertRaises(tables.TableError): self.build(tables.table_accuracy)
        dest,archive=put_r52(self.root,output=False)
        with self.assertRaises(tables.TableError): self.build(tables.table_accuracy)
        dest,archive=put_r52(self.root,partial=True,rc=143,output=False)
        (archive/'projection.txt').unlink()
        with self.assertRaises(tables.TableError): self.build(tables.table_accuracy)

    def test_r52_success_rejects_false_checks_wrong_provenance_scope_and_handoff(self):
        # Break caught: a plausible accuracy overrides failed scientific/provenance/coverage fields.
        mutations=[
          ('all_checks_pass',False),('checks.handoff_err_within_2^-10',False),
          ('provenance.git_commit','a'*40),('provenance.written_by','runner'),
          ('provenance.git_modified_tracked_files',' M tracked.rs'),('provenance.command_line','mnist'),
          ('provenance.build.features.cuda',True),('provenance.threads','RAYON_NUM_THREADS = 1'),
          ('provenance.finished_unix',1794099700),('setup.seed',0),('setup.params','small'),
          ('setup.init_weights','other'),('setup.eval_config.hoist',1),('plan.layout_d',256.0),('plan.stage_dims',[[196,128,16],[128,64,16],[64,32,16]]),
          ('measured.updates',20),('measured.samples',960),('measured.total.key_switches',0),
          ('measured.predicted_ks_this_run',0),('measured.handoff_bootstraps',0),
          ('twin.max_abs_diff',0.01),('twin.log2_max_abs_diff',-10),
          ('twin.handoff_bootstraps.0.slots_compared',0),('twin.handoff_bootstraps.0.max_abs_err_vs_twin',0.01),
          ('twin.handoff_bootstraps.0.precondition_hit',True),('twin.handoff_bootstraps.0.inputs_above_1',999999),
          ('evaluation.test_acc_twin',0.8),('evaluation.test_acc_encrypted_trained',1.1),
        ]
        for field,value in mutations:
            dest,archive=put_r52(self.root);d=json.loads(dest.read_text());node=d
            parts=field.split('.')
            for part in parts[:-1]: node=node[int(part)] if isinstance(node,list) else node[part]
            node[parts[-1]]=value;dest.write_text(json.dumps(d));refresh_r52_digest(dest,archive)
            with self.subTest(field=field),self.assertRaises(tables.TableError): self.build(tables.table_accuracy)

    def test_r52_queue_output_validation_failure_is_unmeasured_even_with_child_zero(self):
        # Break caught: the queue can reject written output/provenance after a child exits zero.
        dest,archive=put_r52(self.root)
        state=json.loads((archive/'state.json').read_text())
        state.update(status='failed: output checks/provenance',output_error='output checks/provenance do not match pinned invocation')
        del state['output_sha256'];(archive/'state.json').write_text(json.dumps(state))
        r=next(r for r in rows(self.build(tables.table_accuracy)) if r[0].startswith('hekit encrypted eMLP-2,'))
        self.assertIn('failed/unmeasured',r[3]);self.assertEqual(r[4],'—')
        state['output_sha256']=hashlib.sha256(dest.read_bytes()).hexdigest()
        (archive/'state.json').write_text(json.dumps(state))
        with self.assertRaises(tables.TableError):self.build(tables.table_accuracy)

    def test_r52_partial_rounding_and_queue_rejection_preserve_original_control_flow(self):
        # Break caught: rounded projected time can equal the limit although the decision exceeded it;
        # queue output validation may subsequently replace the partial/watchdog status after child0.
        dest,archive=put_r52(self.root,rc=0,partial=True,output=True)
        (archive/'partial.txt').write_text('partial: after >=5 updates; projected 12240 > 12240.0 s\n')
        (archive/'projection.txt').write_text('after 5 updates: 12240 s / 21 updates; limit 12240.0 s\n')
        r=next(r for r in rows(self.build(tables.table_accuracy)) if r[0].startswith('hekit encrypted eMLP-2,'))
        self.assertIn('partial/unmeasured',r[3])
        state=json.loads((archive/'state.json').read_text())
        state.update(status='failed: output checks/provenance',output_error='stale output file')
        del state['output_sha256'];(archive/'state.json').write_text(json.dumps(state))
        r=next(r for r in rows(self.build(tables.table_accuracy)) if r[0].startswith('hekit encrypted eMLP-2,'))
        self.assertIn('partial/unmeasured',r[3])

    def test_r52_projection_only_preserves_success_failure_watchdog_for_both_jobs(self):
        # Break caught: projection is written before deciding to stop; it is not stop evidence.
        for number in (2, 3):
            for rc, watchdog in ((0, False), (1, False), (143, True)):
                for projected in (10000, 50000):
                    with self.subTest(number=number, rc=rc, watchdog=watchdog, projected=projected):
                        dest, archive = put_r52(self.root, number, rc=rc, watchdog=watchdog)
                        limit = 12240.0 if number == 2 else 23040.0
                        projection = f'after 5 updates: {projected} s / 21 updates; limit {limit} s\n'
                        (archive / 'projection.txt').write_text(projection)
                        t = self.build(tables.table_accuracy)
                        r = next(r for r in rows(t) if r[0].startswith(f'hekit encrypted eMLP-{number},'))
                        if rc:
                            self.assertIn('failed/unmeasured', r[3])
                            self.assertEqual(r[4], '—')
                        else:
                            self.assertEqual(r[3:5], ['90.00', '90.00'])
                            self.assertEqual(r[7], 'twin check')
                        self.assertNotIn('partial/unmeasured', r[3])
                        sources = '\n'.join(tables.source_lines(t))
                        self.assertIn(f'raw/R-52/R-52-{number - 1}/projection.txt\tline 1', sources)
                        self.assertIn(hashlib.sha256(projection.encode()).hexdigest(), tables.render_md(t, 'synthetic/'))
                        self.assertEqual((archive / 'projection.txt').read_text(), projection)

    def test_r52_projection_and_stop_evidence_must_each_match_original_format(self):
        # Break caught: fixing projection-only intake must not allow malformed projection or incomplete stops.
        def read_job(number):
            store = tables.Store(self.root)
            return tables.r52_input(store, tables.Inputs(store), f'R-52-{number - 1}')

        for number in (2, 3):
            limit = 12240.0 if number == 2 else 23040.0
            for projection in ('synthetic malformed\n',
                               f'after 4 updates: 10000 s / 21 updates; limit {limit} s\n',
                               'after 5 updates: 10000 s / 21 updates; limit 1 s\n'):
                with self.subTest(number=number, projection=projection):
                    _, archive = put_r52(self.root, number)
                    (archive / 'projection.txt').write_text(projection)
                    with self.assertRaises(tables.TableError):
                        read_job(number)
            for stop_file in ('partial.txt', 'partial-identities.json'):
                with self.subTest(number=number, stop_file=stop_file):
                    _, archive = put_r52(self.root, number, rc=143, partial=True, output=False)
                    (archive / stop_file).unlink()
                    with self.assertRaises(tables.TableError):
                        read_job(number)
                    for name in ('partial.txt', 'projection.txt', 'partial-identities.json'):
                        if (archive / name).exists():
                            (archive / name).unlink()
            _, archive = put_r52(self.root, number)
            state = json.loads((archive / 'state.json').read_text())
            state['status'] = 'partial'
            (archive / 'state.json').write_text(json.dumps(state))
            (archive / 'projection.txt').write_text(f'after 5 updates: 50000 s / 21 updates; limit {limit} s\n')
            with self.assertRaises(tables.TableError):
                read_job(number)
            (archive / 'projection.txt').unlink()

    def test_r52_nonzero_errors_and_registered_input_prefix_validation(self):
        # Break caught: demanding null for every log2 error, or accepting a different input prefix.
        dest,archive=put_r52(self.root);d=json.loads(dest.read_text())
        d['twin']['max_abs_diff']=2**-12;d['twin']['log2_max_abs_diff']=-12.0
        d['twin']['handoff_bootstraps'][0].update(max_abs_err_vs_twin=2**-12,log2_max_abs_err_vs_twin=-12.0)
        dest.write_text(json.dumps(d));refresh_r52_digest(dest,archive)
        self.build(tables.table_accuracy)
        indices=self.root/'results/reboot_emlp2_seed0_steps1-21_indices.txt'
        content=indices.read_text()
        if indices.is_symlink():indices.unlink()
        indices.write_text('0 '*1008)
        with self.assertRaises(tables.TableError):self.build(tables.table_accuracy)
        indices.write_text(content)
        # Late child failure invalidates even a fully successful saved result object.
        meta=json.loads((archive/'meta.json').read_text());meta['exit_status']=1
        (archive/'meta.json').write_text(json.dumps(meta))
        state=json.loads((archive/'state.json').read_text());state.update(rc=1,status='failed')
        (archive/'state.json').write_text(json.dumps(state))
        (archive/'post.txt').write_text('date: 2026-11-08T10:00:10+09:00\nJOB_DONE rc=1\n')
        r=next(r for r in rows(self.build(tables.table_accuracy)) if r[0].startswith('hekit encrypted eMLP-2,'))
        self.assertIn('failed/unmeasured',r[3])

    def test_r52_registered_input_hash_failure_is_not_success(self):
        # Break caught: consuming result identities against mismatched supported input/manifest hashes.
        dest,archive=put_r52(self.root)
        weight=self.root/'results/quadruped/raw/reboot_emlp23_full/init_weights_emlp2_seed0/linear_fwd_00.npy'
        weight.write_bytes(b'SYNTHETIC corrupt weight')
        with self.assertRaises(tables.TableError):self.build(tables.table_accuracy)
        dest,archive=put_r52(self.root)
        data=self.root/'results/quadruped/data/reboot_emlp1_full/train_labels.npy'
        data.write_bytes(b'SYNTHETIC corrupt data')
        with self.assertRaises(tables.TableError):self.build(tables.table_accuracy)

    def test_r52_run_all_alias_requires_same_terminal_originals_and_bytes(self):
        # Break caught: generic new_result bypass or substituting an unrelated canonical original.
        dest,archive=put_r52(self.root)
        alias=self.root/'results/run_all/full/ni196_emlp2_b48_sub1008_s0.json'
        alias.parent.mkdir(parents=True);alias.write_bytes(dest.read_bytes())
        s=tables.Store(self.root);t=tables.table_accuracy(s,tables.Inputs(s,'full'))
        self.assertTrue(any('1 (baseline seed 0; RNG 32)' in r for r in rows(t)))
        d=json.loads(alias.read_text());d['evaluation']['test_acc_twin']=0.89;alias.write_text(json.dumps(d))
        s=tables.Store(self.root)
        with self.assertRaises(tables.TableError):tables.table_accuracy(s,tables.Inputs(s,'full'))
        alias.write_bytes(dest.read_bytes());dest.unlink()
        s=tables.Store(self.root);tables.table_accuracy(s,tables.Inputs(s,'full'))
        (archive/'post.txt').unlink()
        s=tables.Store(self.root)
        with self.assertRaises(tables.TableError):tables.table_accuracy(s,tables.Inputs(s,'full'))

    def test_explicit_tex_note_override_keeps_markdown_default_and_sources(self):
        # Break caught: a manuscript-only summary must not replace the inspectable original note/source index.
        table = tables.Table('synthetic_note', 'SYNTHETIC note test', ['Value'])
        table.row(['synthetic'])
        full = 'SYNTHETIC full hash ' + 'a'*64 + '; not historical execution proof'
        table.note(full, [('synthetic/input', 'raw bytes')], tex_text='SYNTHETIC concise manuscript note')
        table.note('SYNTHETIC ordinary default note', [('synthetic/input', 'field')])
        md = tables.render_md(table, 'synthetic/')
        tex = tables.render_tex(table, 'synthetic/')
        self.assertIn(full, md)
        self.assertNotIn('concise manuscript', md)
        self.assertIn('SYNTHETIC concise manuscript note', tex)
        self.assertNotIn('a'*64, tex)
        self.assertIn('SYNTHETIC ordinary default note', md)
        self.assertIn('SYNTHETIC ordinary default note', tex)
        self.assertIsNone(table.notes[1].tex_text)
        self.assertEqual(tables.source_lines(table), [
            'synthetic_note\tnote 1\t\t\tsynthetic/input\traw bytes\t',
            'synthetic_note\tnote 2\t\t\tsynthetic/input\tfield\t'])

    def test_r52_tex_summarizes_only_hash_inventory_and_md_keeps_full_hashes(self):
        # Break caught: long source-path/SHA inventories clip the paper; all acceptance caveats must remain.
        digests = []
        for number in (2, 3):
            dest, _ = put_r52(self.root, number)
            digests.append(hashlib.sha256(dest.read_bytes()).hexdigest())
        table = self.build(tables.table_accuracy)
        md, tex = tables.render_md(table, 'synthetic/'), tables.render_tex(table, 'synthetic/')
        overridden = [note for note in table.notes if note.tex_text is not None]
        self.assertEqual(len(overridden), 2)
        self.assertTrue(all('captured original/source SHA256' in note.text for note in overridden))
        for digest in digests:
            self.assertIn(digest, md)
            self.assertNotIn(digest, tex)
        self.assertIn('not historical execution proof', md)
        self.assertIn('results/tables/accuracy_vs_reboot.md', overridden[0].tex_text)
        self.assertIn('results/tables/sources.tsv', overridden[0].tex_text)
        self.assertIn('captured/current bytes', tex)
        self.assertIn('not historical execution or consumed-input proof', tex)
        for caveat in ('21 updates; 1008 samples', 'RNG32', 'not independent seeds',
                       'compared 64512 valid-sample/width slots', 'precondition hit false',
                       'not historical input-byte proof'):
            self.assertIn(caveat, tex)
        sources = tables.source_lines(table)
        for note in overridden:
            note.tex_text = None
        self.assertEqual(sources, tables.source_lines(table))
        self.assertTrue(all(note.tex_text is None for note in self.build(tables.table_gpu).notes))
        # The same concise inventory treatment applies to an unmeasured failed original.
        put_r52(self.root, 2, rc=1, output=False)
        failed = self.build(tables.table_accuracy)
        self.assertEqual(len([note for note in failed.notes if note.tex_text is not None]), 2)

    def test_r52_error_atomicity_quick_and_both_entrypoints(self):
        # Break caught: partial paper writes, quick contamination, or entrypoint divergence.
        dest,archive=put_r52(self.root);(archive/'post.txt').write_text('JOB_DONE rc=1\n')
        with contextlib.redirect_stderr(io.StringIO()),contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(tables.main(['--root',str(self.root)]),2)
            self.assertFalse((self.root/'paper').exists())
            self.assertEqual(tables.main(['--root',str(self.root),'--from-run-all','quick']),0)
        put_r52(self.root)
        for entry in ('paper_tables.py','tables.py'):
            p=subprocess.run([sys.executable,str(ROOT/'tools'/entry),'--root',str(self.root),'--sources','accuracy_vs_reboot'],capture_output=True,text=True)
            self.assertEqual(p.returncode,0,p.stderr)
            self.assertIn('baseline seed 0; RNG 32',p.stdout)

    def test_r52_future_archive_and_result_overlay_isolation(self):
        # Break caught: a future real R52 raw/result symlink overwritten by synthetic fixture writes.
        with tempfile.TemporaryDirectory() as source,tempfile.TemporaryDirectory() as fixture:
            source,fixture=Path(source),Path(fixture)
            for rel in ('docs','results/quadruped/raw/R-52'):(source/rel).mkdir(parents=True,exist_ok=True)
            result=source/'results/quadruped/mnist196_emlp2_ni_b48_subset1008_seed0.json'
            result.write_text('SYNTHETIC original');sentinel=source/'results/quadruped/raw/R-52/original'
            sentinel.write_text('SYNTHETIC original');archived_overlay(fixture,source)
            self.assertFalse((fixture/'results/quadruped/raw/R-52').exists())
            self.assertFalse((fixture/'results/quadruped'/result.name).exists())
            (fixture/'results/quadruped'/result.name).write_text('SYNTHETIC replacement')
            self.assertEqual(result.read_text(),'SYNTHETIC original');self.assertEqual(sentinel.read_text(),'SYNTHETIC original')


    def test_r48_registered_nag_pair_l31_context_and_sources(self):
        # Break caught: output fields render without original terminal/workload/iteration context.
        for job in ('R-48-2','R-48-3','R-48-4'):put_r48(self.root,job)
        put_r48_optimizer(self.root,'nag')
        table=self.build(tables.table_ablation)
        measured=[r for r in rows(table) if r[0].startswith(('GPU, n16_train','CPU, n16_train'))]
        self.assertEqual(len(measured),24)
        md=tables.render_md(table,'synthetic/')
        self.assertIn('5 timed iterations',md);self.assertIn('3 timed iterations',md)
        self.assertIn('within seed 48',md);self.assertIn('not independent seeds',md)
        self.assertIn('load_start',md);self.assertIn('load_end',md)
        self.assertIn('sum of step and refresh medians',md)
        self.assertIn('not GPU peak',md);self.assertIn('not NI bootstrap',md)
        src='\n'.join(tables.source_lines(table))
        self.assertIn('R-48/R-48-2/post.txt\tline 2',src)
        self.assertIn('setup.timed_iterations',src)
        self.assertTrue(any('upper bound' in n.text for n in table.notes))

    def test_r48_conditional_sgd_both_skip_and_failed_sgd_choices(self):
        # Break caught: unlike CPU/GPU optimizers, hidden NAG skip, or failed SGD treated as successful.
        for sgd_rc,sgd_skip,cpu_opt,watchdog in ((0,False,'sgd',False),(3,True,'nag',False),
                                                   (1,False,'sgd',False),(3,True,'nag',True)):
            with self.subTest(sgd_rc=sgd_rc):
                put_r48(self.root,rc=3,skip=True)
                put_r48(self.root,'R-48-2-sgd',rc=sgd_rc,skip=sgd_skip,watchdog=watchdog)
                put_r48(self.root,'R-48-3',optimizer=cpu_opt)
                put_r48_optimizer(self.root,cpu_opt)
                table=self.build(tables.table_ablation)
                gpu=[r for r in rows(table) if r[0].startswith('GPU, n16_train')]
                self.assertEqual(len(gpu),8 if sgd_rc==0 else 1)
                if sgd_rc:self.assertIn('unmeasured',gpu[0][7])
                self.assertTrue(any('R-48-2' in n.text and 'capacity SKIP' in n.text for n in table.notes))
                # Remove prior selected CPU variant in this isolated synthetic iteration.
                for rel in ('ablation_cpu_l7.json','ablation_cpu_l7_sgd.json'):
                    path=self.root/'results/quadruped'/rel
                    if path.exists():path.unlink()

    def test_r48_cpu_only_uses_optimizer_record_without_manufacturing_gpu(self):
        # Break caught: a standalone CPU record implies an unseen GPU run or requires one unnecessarily.
        for optimizer in ('nag','sgd'):
            dest,archive=put_r48(self.root,'R-48-3',optimizer=optimizer)
            record=put_r48_optimizer(self.root,optimizer)
            table=self.build(tables.table_ablation)
            gpu=next(r for r in rows(table) if r[0].startswith('GPU, n16_train'))
            self.assertIn('pending (R-48-2)',gpu)
            self.assertEqual(len([r for r in rows(table) if r[0].startswith('CPU, n16_train, level 7')]),8)
            record.unlink()
            with self.assertRaises(tables.TableError):self.build(tables.table_ablation)
            record.write_text('{"cpu_optimizer":"other"}')
            with self.assertRaises(tables.TableError):self.build(tables.table_ablation)
            dest.unlink()

    def test_r48_preceding_skip_and_missing_optional_members_remain_unmeasured_or_pending(self):
        # Break caught: later optimizer/CPU/SGD absence hides the actual preceding capacity skip.
        put_r48(self.root,rc=3,skip=True)
        table=self.build(tables.table_ablation)
        gpu=next(r for r in rows(table) if r[0].startswith('GPU,'))
        cpu=next(r for r in rows(table) if r[0].startswith('CPU, n16_train, level 7'))
        self.assertIn('capacity SKIP/unmeasured',gpu[7]);self.assertIn('pending (R-48-3)',cpu)
        source='\n'.join(tables.source_lines(table))
        self.assertIn('R-48/R-48-2/program.log\tline 1',source)

    def test_r48_oom_watchdog_late_failure_and_output_rejection_are_unmeasured(self):
        # Break caught: printed/saved success JSON overrides a failed original terminal.
        for rc,watchdog,oom in ((1,False,True),(143,True,False),(1,False,False)):
            put_r48(self.root,rc=rc,watchdog=watchdog,oom=oom,output=True)
            table=self.build(tables.table_ablation)
            gpu=[r for r in rows(table) if r[0].startswith('GPU, n16_train')]
            self.assertEqual(len(gpu),1);self.assertIn('failed/unmeasured',gpu[0][7])
        dest,archive=put_r48(self.root)
        state=json.loads((archive/'state.json').read_text())
        state.update(status='failed: output checks/provenance',output_error='synthetic output rejection')
        del state['output_sha256'];(archive/'state.json').write_text(json.dumps(state))
        self.assertIn('failed/unmeasured',next(r for r in rows(self.build(tables.table_ablation)) if r[0].startswith('GPU,'))[7])

    def test_r48_branch_conflicts_and_missing_nag_trigger_fail_closed(self):
        # Break caught: selecting SGD from mere presence, or contradictory predecessor/CPU choice.
        put_r48(self.root,'R-48-2-sgd')
        with self.assertRaises(tables.TableError):self.build(tables.table_ablation)
        put_r48(self.root) # NAG succeeds: SGD was never selected.
        with self.assertRaises(tables.TableError):self.build(tables.table_ablation)
        put_r48(self.root,rc=3,skip=True)
        put_r48(self.root,'R-48-3',optimizer='nag');put_r48_optimizer(self.root,'nag')
        with self.assertRaises(tables.TableError):self.build(tables.table_ablation)

    def test_r48_capacity_skip_requires_the_actual_producer_marker(self):
        # Break caught: a generic SKIP substring is queue metadata, not the producer's capacity diagnostic.
        _,archive=put_r48(self.root,rc=3,skip=True)
        log='SYNTHETIC unrelated SKIP comment\n';(archive/'program.log').write_text(log)
        state=json.loads((archive/'state.json').read_text());state['skip_lines']=log.splitlines()
        (archive/'state.json').write_text(json.dumps(state))
        with self.assertRaises(tables.TableError):self.build(tables.table_ablation)

    def test_r48_wrong_terminal_and_binary_provenance_fail_without_partial_outputs(self):
        # Break caught: canonical path bypasses provenance, or malformed input leaves partial generated files.
        for name,key,value in [('meta.json','threads_cap',1),('meta.json','hekit_synced_commit',['full '+'a'*40]),
             ('meta.json','gpu_job',False),('state.json','rc',1),('state.json','output_sha256','a'*64),
             ('state.json','command',['other']),('meta.json','start','2026-11-08T10:00:00')]:
            dest,archive=put_r48(self.root);d=json.loads((archive/name).read_text());d[key]=value
            (archive/name).write_text(json.dumps(d))
            with self.subTest(name=name,key=key),self.assertRaises(tables.TableError):self.build(tables.table_ablation)
        dest,archive=put_r48(self.root);(archive/'post.txt').unlink()
        with contextlib.redirect_stderr(io.StringIO()),contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(tables.main(['--root',str(self.root)]),2)
            self.assertFalse((self.root/'paper').exists())
            self.assertEqual(tables.main(['--root',str(self.root),'--from-run-all','quick']),0)

    def test_r48_rejects_bad_binary_flags_counts_stats_and_protocol(self):
        # Break caught: len8/top-level success masks duplicate rows or contradictory actual fields.
        mutations=[('provenance.written_by','runner'),('provenance.git_commit','a'*40),
          ('provenance.build.features.cuda',False),('provenance.machine.host','other'),
          ('provenance.finished_unix',1794099700),('provenance.gpu',None),
          ('setup.lr',0.002),('setup.dims',[196,32,10]),('setup.timed_iterations',3),
          ('configs.0.hoist',1),('configs.0.closed_form_matches',False),
          ('configs.0.counter_constant_over_iterations',False),('configs.0.counts_equal_hoist_on_counterpart',False),
          ('configs.4.weights_bit_identical_to_hoist_on_counterpart',False),
          ('configs.0.counter_per_iteration.key_switches',0),('configs.0.closed_form.rotations',0),
          ('configs.0.step_s',[0,1,2,3,4]),('configs.0.step_s_median',99),
          ('configs.0.step_time_over_default',2),('configs.0.iteration_s_median',99),
          ('configs.0.log2_max_abs_weight_diff_vs_default',-10),('protocol','other')]
        for field,value in mutations:
            dest,archive=put_r48(self.root);d=json.loads(dest.read_text());node=d;parts=field.split('.')
            for part in parts[:-1]:node=node[int(part)] if isinstance(node,list) else node[part]
            node[parts[-1]]=value;dest.write_text(json.dumps(d));refresh_r52_digest(dest,archive)
            with self.subTest(field=field),self.assertRaises(tables.TableError):self.build(tables.table_ablation)
        dest,archive=put_r48(self.root);d=json.loads(dest.read_text());d['configs']=[d['configs'][0]]*8
        dest.write_text(json.dumps(d));refresh_r52_digest(dest,archive)
        with self.assertRaises(tables.TableError):self.build(tables.table_ablation)

    def test_r48_synthetic_pixel_fallback_is_visible_invalid_workload_without_times(self):
        # Break caught: P48's synthetic-pixel fallback silently becomes real-MNIST measured training.
        dest,archive=put_r48(self.root);d=json.loads(dest.read_text())
        d['setup']['data']='synthetic uniform pixels (MNIST IDX files not found)'
        dest.write_text(json.dumps(d));refresh_r52_digest(dest,archive)
        table=self.build(tables.table_ablation);gpu=[r for r in rows(table) if r[0].startswith('GPU,')]
        self.assertEqual(len(gpu),1);self.assertIn('invalid-workload/unmeasured',gpu[0][7])
        self.assertTrue(any('synthetic uniform pixels' in note.text for note in table.notes))

    def test_r48_full_alias_same_original_and_both_entrypoints(self):
        # Break caught: generic run_all aliases select historical/unassociated bytes.
        dest,archive=put_r48(self.root)
        alias=self.root/'results/run_all/full/ablation_cuda_l7.json';alias.parent.mkdir(parents=True)
        alias.write_bytes(dest.read_bytes())
        store=tables.Store(self.root);tables.table_ablation(store,tables.Inputs(store,'full'))
        alias.write_text('{}')
        store=tables.Store(self.root)
        with self.assertRaises(tables.TableError):tables.table_ablation(store,tables.Inputs(store,'full'))
        alias.write_bytes(dest.read_bytes());(archive/'post.txt').unlink()
        store=tables.Store(self.root)
        with self.assertRaises(tables.TableError):tables.table_ablation(store,tables.Inputs(store,'full'))
        put_r48(self.root)
        for entry in ('paper_tables.py','tables.py'):
            p=subprocess.run([sys.executable,str(ROOT/'tools'/entry),'--root',str(self.root),'--sources','rq3_ablation'],capture_output=True,text=True)
            self.assertEqual(p.returncode,0,p.stderr)
            self.assertIn('R-48/R-48-2/post.txt',p.stdout)

    def test_r48_wrong_cpu_candidate_without_terminal_is_rejected_atomically(self):
        # Break caught: CPU detection sees both variants, but a missing selected variant used to hide the wrong one.
        for selected, wrong in (('nag', 'sgd'), ('sgd', 'nag')):
            put_r48_optimizer(self.root, selected)
            filename = 'ablation_cpu_l7' + ('_sgd' if wrong == 'sgd' else '') + '.json'
            candidate = self.root / 'results/quadruped' / filename
            raw = b'{"SYNTHETIC":"wrong CPU variant without original terminal"}'
            candidate.write_bytes(raw)
            expected = self.root / 'results/quadruped' / ('ablation_cpu_l7' + ('_sgd' if selected == 'sgd' else '') + '.json')
            self.assertFalse(expected.exists())
            self.assertFalse((self.root / 'results/quadruped/raw/R-48/R-48-3').exists())
            with self.subTest(selected=selected, consumer='table'):
                with self.assertRaises(tables.TableError):
                    self.build(tables.table_ablation)
            for entry in ('paper_tables.py', 'tables.py'):
                with self.subTest(selected=selected, consumer=entry):
                    p = subprocess.run([sys.executable, str(ROOT / 'tools' / entry), '--root', str(self.root),
                                        '--sources', 'rq3_ablation'], capture_output=True, text=True)
                    self.assertEqual(p.returncode, 2, p.stdout)
                    self.assertEqual(p.stdout, '')
            with self.subTest(selected=selected, consumer='generate'):
                with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                    self.assertEqual(tables.main(['--root', str(self.root)]), 2)
                self.assertFalse((self.root / 'paper').exists())
                self.assertFalse((self.root / 'results/tables').exists())
            self.assertEqual(candidate.read_bytes(), raw)
            candidate.unlink()

    def test_r48_cpu_alias_only_selected_bytes_and_failed_original_stay_supported(self):
        # Break caught: rejecting a legitimate selected alias while moving the wrong-candidate guard.
        alias = self.root / 'results/run_all/full/ablation_cpu_l7.json'
        alias.parent.mkdir(parents=True)
        for optimizer in ('nag', 'sgd'):
            original, _ = put_r48(self.root, 'R-48-3', optimizer=optimizer)
            put_r48_optimizer(self.root, optimizer)
            alias.write_bytes(original.read_bytes())
            original.unlink()
            store = tables.Store(self.root)
            table = tables.table_ablation(store, tables.Inputs(store, 'full'))
            cpu = [r for r in rows(table) if r[0].startswith('CPU, n16_train, level 7')]
            gpu = next(r for r in rows(table) if r[0].startswith('GPU,'))
            self.assertEqual(len(cpu), 8)
            self.assertIn('pending (R-48-2)', gpu)
            alias.unlink()
            put_r48(self.root, 'R-48-3', optimizer=optimizer, rc=1, output=False)
            table = self.build(tables.table_ablation)
            cpu = [r for r in rows(table) if r[0].startswith('CPU, n16_train, level 7')]
            self.assertEqual(len(cpu), 1)
            self.assertIn('failed/unmeasured', cpu[0][7])

    def test_r48_future_archive_and_binary_overlay_write_isolation(self):
        # Break caught: fixtures overwrite future actual R48 originals through symlinks.
        with tempfile.TemporaryDirectory() as source,tempfile.TemporaryDirectory() as fixture:
            source,fixture=Path(source),Path(fixture)
            for rel in ('docs','results/quadruped/raw/R-48'):(source/rel).mkdir(parents=True,exist_ok=True)
            names=('ablation_cuda.json','ablation_cuda_sgd.json','ablation_cpu_l7.json','ablation_cpu_l7_sgd.json','ablation_cpu.json')
            for name in names:(source/'results/quadruped'/name).write_text('SYNTHETIC original')
            archived_overlay(fixture,source)
            self.assertFalse((fixture/'results/quadruped/raw/R-48').exists())
            for name in names:
                path=fixture/'results/quadruped'/name
                self.assertFalse(path.exists());path.write_text('SYNTHETIC replacement')
                self.assertEqual((source/'results/quadruped'/name).read_text(),'SYNTHETIC original')


if __name__ == '__main__':
    unittest.main()
