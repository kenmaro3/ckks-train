"""RQ1 aggregation tests. All full-length result fixtures here are SYNTHETIC."""
import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / 'tools/rq1_summary.py'
REFERENCE = ROOT / 'results/quadruped/raw/reboot_emlp1_full/config_values.json'


def synthetic_full(batch=64):
    """SYNTHETIC full-length raw-schema fixture, never a measurement."""
    updates = 750 if batch == 64 else 1000
    schedule = [0, 1] * 375 if batch == 64 else [1] * 1000
    boots = 1500 if batch == 64 else 4000
    nonboot = 720000 if batch == 64 else 960000
    total = nonboot + boots * 111
    command = (
        './target/release/mnist --mode noninteractive --placement when_needed '
        f'--batch {batch} --faithful --dataset-npy results/quadruped/data/reboot_emlp1_full '
        '--indices results/quadruped/raw/reboot_emlp1_full/train_indices.txt '
        '--init-weights results/quadruped/raw/reboot_emlp1_full/init_weights_seed0 '
        '--init-weights-manifest results/quadruped/raw/reboot_emlp1_full/config_values.json '
        '--lr-schedule results/quadruped/raw/reboot_emlp1_full/config_values.json '
        '--lr-key lr_per_step --epochs 1 --transpose reuse'
        + (' --lr-schedule-batch 48' if batch == 64 else ''))
    return {
        'synthetic_fixture': True,
        'provenance': {
            'written_by': 'the mnist binary (raw output, not assembled)',
            'git_commit': '1' * 40, 'git_modified_tracked_files': 'none',
            'command_line': command, 'started_unix': 1793880000.0,
            'finished_unix': 1793980000.0,
            'machine': {'cpu': 'SYNTHETIC CPU', 'host': 'SYNTHETIC host',
                        'os': 'SYNTHETIC Linux', 'logical_cpus': 32,
                        'memory_bytes': '132479885312'},
            'threads': 'rayon (feature parallel), RAYON_NUM_THREADS = 12, 32 logical CPUs'},
        'network': '196 -> 32 (z^2 + z) -> 16 (linear), MSE, NAG (biases SGD)',
        'setup': {'params': 'n16_boot', 'log_n': 16, 'slots': 32768, 'boot_ks': 111,
                  'placement': 'when_needed', 'batch': batch,
                  'lanes_per_ciphertext': 64, 'micro_batches_per_update': 1,
                  'layout_d': 256, 'transpose_keys': 'reuse'},
        'training': {'epochs': 1, 'subset': 48000, 'updates': updates, 'samples': 48000,
                     'lr': 0.0005 * batch, 'momentum': 0.9, 'weight_decay': 0,
                     'seed': 32,
                     'lr_schedule': 'results/quadruped/raw/reboot_emlp1_full/config_values.json'},
        'prediction': {'predicted_ks_per_sample': total / 48000,
                       'bootstraps_per_update_avg': boots / updates,
                       'boot_schedule': schedule},
        'measured': {'total': {'rotations': nonboot - 80 * updates + 69 * boots,
                               'relins': 80 * updates + 40 * boots,
                               'sse_switches': 2 * boots, 'key_switches': total,
                               'bootstraps': boots},
                     'ks_per_sample_incl_bootstraps': total / 48000,
                     'ks_per_sample_without_bootstraps': nonboot / 48000,
                     'bootstraps_per_update': boots / updates,
                     'bootstraps_per_sample': boots / 48000,
                     'weight_booted': schedule[:], 'every_update_equals_prediction': True,
                     'predicted_ks_this_run': total, 'measured_over_predicted': 1},
        'time_s': {'train_total': 70000, 'epoch_measured': 70000,
                   'wall_total': 70100, 'bootstrap_time_total': 20000},
        'twin': {'max_abs_diff': 0.00005, 'log2_max_abs_diff': -14.287712379549449,
                 'max_abs_w_diff_per_layer': [0.00005, 0.00003],
                 'max_abs_b_diff_per_layer': [0.00001, 0.00002]},
        'evaluation': {'test_samples': 10000, 'test_acc_encrypted_trained':
                       0.89 if batch == 64 else 0.9042, 'test_acc_twin': 0.9041,
                       'test_acc_init': 0.15, 'test_mse_encrypted_trained': 0.02,
                       'test_mse_twin': 0.02}}


class SummaryTests(unittest.TestCase):
    def test_missing_full_results_are_unmeasured_without_creating_files(self):
        # Break caught: reporting absent measurements as results, or writing artifacts.
        with tempfile.TemporaryDirectory() as tmp:
            one, two = Path(tmp) / 'full1.json', Path(tmp) / 'full2.json'
            run = subprocess.run([sys.executable, str(SCRIPT), '--full-1', str(one),
                                  '--full-2', str(two), '--reboot-config', str(REFERENCE)],
                                 capture_output=True, text=True)
            self.assertEqual(run.returncode, 0, run.stderr)
            self.assertIn('FULL-1: unmeasured', run.stdout)
            self.assertIn('FULL-2: unmeasured', run.stdout)
            self.assertIn('Overall RQ1: pending', run.stdout)
            self.assertEqual(list(Path(tmp).iterdir()), [])

    def run_summary(self, one=None, two=None, reference=REFERENCE):
        with tempfile.TemporaryDirectory() as tmp:
            paths = [Path(tmp) / 'full1.json', Path(tmp) / 'full2.json']
            for path, doc in zip(paths, (one, two)):
                if doc is not None:
                    path.write_text(json.dumps(doc))
            before = {p: p.read_bytes() for p in paths if p.exists()}
            run = subprocess.run([sys.executable, str(SCRIPT), '--full-1', str(paths[0]),
                                  '--full-2', str(paths[1]), '--reboot-config', str(reference)],
                                 capture_output=True, text=True)
            self.assertEqual({p: p.read_bytes() for p in before}, before)
            return run

    def test_recomputes_four_ratios_from_counts_and_uses_reboot_b48(self):
        # Break caught: using rounded cached ratios, hekit B, or interactive counts.
        run = self.run_summary(synthetic_full(), synthetic_full(48))
        self.assertEqual(run.returncode, 0, run.stderr)
        for value in ('0.234276', '0.389271', '0.399324', '0.405405'):
            self.assertIn(value, run.stdout)
        self.assertIn('boot-excluded NI ablation', run.stdout)
        self.assertIn('not a separately executed interactive run', run.stdout)
        self.assertIn('502 (model)', run.stdout)
        self.assertIn('37 (model)', run.stdout)
        self.assertIn('48000 samples (4800 × 10 passes)', run.stdout)
        self.assertIn('binary epochs=1', run.stdout)
        self.assertIn('FULL-1 accuracy (separate): 0.890000', run.stdout)
        self.assertIn('FULL-2 accuracy gate: PASS', run.stdout)
        self.assertIn('Overall RQ1: pending', run.stdout)
        self.assertIn('ReBoot full CPU timing: unmeasured', run.stdout)
        self.assertIn('Security comparability: unverified', run.stdout)
        self.assertIn(hashlib.sha256(json.dumps(synthetic_full()).encode()).hexdigest(), run.stdout)
        self.assertIn('SYNTHETIC', run.stdout)

    def test_accuracy_gate_uses_reference_values_not_seed_average_or_hardcoded_value(self):
        # Break caught: copying 0.9041 or the ten-seed mean instead of reading seed0.
        ref = json.loads(REFERENCE.read_text())
        ref['per_seed']['0']['twin']['test_acc_argmax10'] = 0.80
        ref['twin_test_acc_over_seeds']['values'][0] = 0.80
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / 'config_values.json'
            p.write_text(json.dumps(ref))
            run = self.run_summary(two=synthetic_full(48), reference=p)
        self.assertEqual(run.returncode, 0, run.stderr)
        self.assertIn('seed-0 reference accuracy: 0.800000', run.stdout)
        self.assertIn('FULL-2 accuracy gate: FAIL', run.stdout)

    def test_accuracy_gate_includes_half_percentage_point_boundary(self):
        full = synthetic_full(48)
        for accuracy, gate in [(0.9091, 'PASS'), (0.8991, 'PASS'), (0.9092, 'FAIL')]:
            with self.subTest(accuracy=accuracy):
                full['evaluation']['test_acc_encrypted_trained'] = accuracy
                run = self.run_summary(two=full)
                self.assertEqual(run.returncode, 0, run.stderr)
                self.assertIn('FULL-2 accuracy gate: ' + gate, run.stdout)

    def test_encryption_rng_seed_does_not_replace_initial_weight_seed(self):
        full = synthetic_full()
        full['training']['seed'] = 987
        run = self.run_summary(one=full)
        self.assertEqual(run.returncode, 0, run.stderr)
        self.assertIn('initial weights: seed 0 (command + released config)', run.stdout)
        full['provenance']['command_line'] = full['provenance']['command_line'].replace(
            'init_weights_seed0', 'init_weights_seed1')
        run = self.run_summary(one=full)
        self.assertNotEqual(run.returncode, 0)
        self.assertIn('init-weights', run.stderr)
        self.assertEqual(run.stdout, '')

    def test_invalid_or_incomplete_full_inputs_fail_closed(self):
        # Break caught: trusting partial, stale, counter-inconsistent or nonfinite results.
        mutations = [
            ('writer', lambda d: d['provenance'].update(written_by='assembled')),
            ('commit', lambda d: d['provenance'].update(git_commit='unknown')),
            ('timestamps', lambda d: d['provenance'].update(finished_unix=1)),
            ('machine', lambda d: d['provenance']['machine'].update(cpu='unknown')),
            ('network', lambda d: d.update(network='784 -> 32 -> 16')),
            ('N', lambda d: d['setup'].update(log_n=15)),
            ('samples', lambda d: d['training'].update(samples=47999)),
            ('updates', lambda d: d['training'].update(updates=749)),
            ('batch', lambda d: d['setup'].update(batch=32)),
            ('old-lanes', lambda d: d['setup'].update(lanes_per_ciphertext=32)),
            ('old-micro', lambda d: d['setup'].update(micro_batches_per_update=2)),
            ('old-layout', lambda d: d['setup'].update(layout_d=1024)),
            ('schedule', lambda d: d['training'].pop('lr_schedule')),
            ('command-schedule', lambda d: d['provenance'].update(command_line=
                d['provenance']['command_line'].replace('--lr-schedule ', '--wrong '))),
            ('seed-indices', lambda d: d['provenance'].update(command_line=
                d['provenance']['command_line'].replace('train_indices.txt', 'train_indices_seed1.txt'))),
            ('counter', lambda d: d['measured']['total'].update(key_switches=1)),
            ('bool-counter', lambda d: d['measured']['total'].update(bootstraps=True)),
            ('nonboot', lambda d: d['measured'].update(ks_per_sample_without_bootstraps=1)),
            ('prediction', lambda d: d['measured'].update(every_update_equals_prediction=False)),
            ('predicted-count', lambda d: d['measured'].update(predicted_ks_this_run=1)),
            ('boot-schedule', lambda d: d['prediction']['boot_schedule'].pop()),
            ('booted', lambda d: d['measured']['weight_booted'].__setitem__(0, 1)),
            ('accuracy', lambda d: d['evaluation'].update(test_acc_encrypted_trained=float('nan'))),
            ('null-accuracy', lambda d: d['evaluation'].update(test_acc_encrypted_trained=None)),
            ('MSE', lambda d: d['evaluation'].update(test_mse_twin=float('inf'))),
            ('time', lambda d: d['time_s'].update(train_total=0)),
            ('twin', lambda d: d['twin'].update(max_abs_diff=None)),
        ]
        for name, mutate in mutations:
            with self.subTest(name=name):
                full = synthetic_full()
                mutate(full)
                run = self.run_summary(one=full)
                self.assertNotEqual(run.returncode, 0, name)
                self.assertEqual(run.stdout, '', name)
                self.assertIn('error:', run.stderr)
                self.assertNotIn('Traceback', run.stderr)

    def test_full2_requires_every_update_bootstrap_for_partial_lanes(self):
        full = synthetic_full(48)
        full['prediction']['boot_schedule'][0] = 0
        full['measured']['weight_booted'][0] = 0
        run = self.run_summary(two=full)
        self.assertNotEqual(run.returncode, 0)

    def test_real_raw_subset_is_not_promoted_to_full_measurement(self):
        raw = ROOT / 'results/quadruped/raw/R-41-F2/mnist_emlp1_ni_when_needed_b64_faithful.json'
        run = subprocess.run([sys.executable, str(SCRIPT), '--full-1', str(raw),
                              '--full-2', str(raw.parent / 'missing.json'),
                              '--reboot-config', str(REFERENCE)], capture_output=True, text=True)
        self.assertNotEqual(run.returncode, 0)
        self.assertIn('196', run.stderr)
        self.assertEqual(run.stdout, '')

    def test_malformed_json_fails_without_partial_markdown(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'truncated.json'
            path.write_text('{"provenance":')
            run = subprocess.run([sys.executable, str(SCRIPT), '--full-1', str(path),
                                  '--full-2', str(path.parent / 'missing'), '--reboot-config',
                                  str(REFERENCE)], capture_output=True, text=True)
        self.assertNotEqual(run.returncode, 0)
        self.assertIn('error:', run.stderr)
        self.assertEqual(run.stdout, '')

    def test_missing_reference_preserves_pending_accuracy(self):
        run = self.run_summary(reference=REFERENCE.parent / 'absent.json')
        self.assertEqual(run.returncode, 0, run.stderr)
        self.assertIn('ReBoot released config: unmeasured', run.stdout)
        self.assertIn('FULL-2 accuracy gate: pending', run.stdout)

    def test_invalid_reference_is_not_silently_used(self):
        for section, key, value in [('config_values', 'batch_size', 64),
                                    ('config_values', 'input_resize', 28),
                                    ('config_values', 'epochs', 1)]:
            with self.subTest(key=key), tempfile.TemporaryDirectory() as tmp:
                ref = json.loads(REFERENCE.read_text())
                ref[section][key] = value
                path = Path(tmp) / 'config_values.json'
                path.write_text(json.dumps(ref))
                run = self.run_summary(reference=path)
                self.assertNotEqual(run.returncode, 0)
                self.assertIn('error:', run.stderr)

    def test_real_raw_binary_parser_regression(self):
        # Break caught: assuming encryption seed=0 or rejecting valid optional null diagnostics.
        from tools import rq1_summary
        raw = ROOT / 'results/quadruped/raw/R-41-F2/mnist_emlp1_ni_when_needed_b64_faithful.json'
        parsed = rq1_summary.read_binary(raw)
        self.assertEqual(parsed.data['measured']['total']['key_switches'], 18028)
        self.assertEqual(parsed.data['measured']['total']['bootstraps'], 36)
        self.assertEqual(parsed.sha256, hashlib.sha256(raw.read_bytes()).hexdigest())

    def test_truncated_nested_schemas_fail_with_clear_cli_error(self):
        for section in ('provenance', 'setup', 'training', 'prediction', 'measured',
                        'time_s', 'twin', 'evaluation'):
            for value in ({}, None, []):
                with self.subTest(section=section, value=value):
                    full = synthetic_full()
                    full[section] = value
                    run = self.run_summary(one=full)
                    self.assertNotEqual(run.returncode, 0)
                    self.assertIn('error:', run.stderr)
                    self.assertNotIn('Traceback', run.stderr)
                    self.assertEqual(run.stdout, '')

    def test_reference_model_settings_and_declared_training_must_agree(self):
        # Break caught: mixing a different activation or optimizer with the accuracy reference.
        for key, value in [('momentum', 0.8), ('lr', 0.001), ('weight_decay', 0.001)]:
            with self.subTest(key=key):
                full = synthetic_full()
                full['training'][key] = value
                run = self.run_summary(one=full)
                self.assertNotEqual(run.returncode, 0)
                self.assertIn(key, run.stderr)
        with tempfile.TemporaryDirectory() as tmp:
            ref = json.loads(REFERENCE.read_text())
            ref['config_values']['activation'] = 'x^2'
            path = Path(tmp) / 'config_values.json'
            path.write_text(json.dumps(ref))
            run = self.run_summary(reference=path)
            self.assertNotEqual(run.returncode, 0)
            self.assertIn('activation', run.stderr)

    def test_existing_m0_model_fields_must_match_recomputed_ratios(self):
        # Break caught: allowing inconsistent cached model ratios to survive aggregation.
        full = synthetic_full()
        model = {'source': 'SYNTHETIC M0 model fixture',
                 'ks_per_sample_without_bootstraps': 37}
        for key, heboot, rbboot, own, reboot, same in [
            ('a_own_bootstrap_costs', 111, 502, 18.46875, 78.83333333333333, 68.375),
            ('b_both_502', 502, 502, 30.6875, 78.83333333333333, 68.375),
            ('c_both_hekit', 111, 111, 18.46875, 46.25, 43.9375)]:
            model[key] = {'hekit_boot_ks': heboot, 'reboot_boot_ks': rbboot,
                          'hekit_ks_per_sample': own, 'reboot_ks_per_sample_b48': reboot,
                          'ratio_vs_reboot_b48': own / reboot,
                          'reboot_ks_per_sample_same_b': same,
                          'ratio_vs_reboot_same_b': own / same}
        for direct in (True, False):
            full['rq1'] = {'m0_model': model} if direct else {'by_reboot_side': {'m0_model': model}}
            self.assertEqual(self.run_summary(one=full).returncode, 0)
            for field in ('hekit_ks_per_sample', 'reboot_ks_per_sample_b48',
                          'ratio_vs_reboot_b48', 'reboot_ks_per_sample_same_b',
                          'ratio_vs_reboot_same_b', 'hekit_boot_ks', 'reboot_boot_ks'):
                with self.subTest(direct=direct, field=field):
                    old = model['a_own_bootstrap_costs'][field]
                    model['a_own_bootstrap_costs'][field] = 999
                    run = self.run_summary(one=full)
                    self.assertNotEqual(run.returncode, 0)
                    self.assertIn('m0_model', run.stderr)
                    self.assertEqual(run.stdout, '')
                    model['a_own_bootstrap_costs'][field] = old

    def test_nonfinite_optional_metrics_are_rejected(self):
        full = synthetic_full()
        full['time_s']['per_update'] = {'mean': float('inf')}
        run = self.run_summary(one=full)
        self.assertNotEqual(run.returncode, 0)
        self.assertEqual(run.stdout, '')
        self.assertIn('finite', run.stderr)

    def test_duplicate_json_keys_fail_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'duplicate.json'
            path.write_text('{"config_values": {}, "config_values": {}}')
            run = self.run_summary(reference=path)
        self.assertNotEqual(run.returncode, 0)
        self.assertIn('duplicate', run.stderr)

    def test_declared_provenance_limits_are_visible(self):
        run = self.run_summary(one=synthetic_full())
        self.assertEqual(run.returncode, 0, run.stderr)
        self.assertIn('declared provenance', run.stdout)
        self.assertIn('built binary', run.stdout)
        self.assertIn('input file contents', run.stdout)


if __name__ == '__main__':
    unittest.main()
