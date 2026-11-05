"""Read-only Task 42 RQ1 aggregation from raw binary JSON and released values.

Usage: python3 tools/rq1_summary.py --full-1 PATH --full-2 PATH --reboot-config PATH
Missing inputs remain unmeasured; malformed or inconsistent inputs fail closed.
Only Markdown is written, to stdout. This does not finalize the M3 milestone.
"""
import argparse
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import re
import shlex
import sys


class SummaryError(ValueError):
    """An input cannot support the requested comparison."""


def require(condition, message):
    if not condition:
        raise SummaryError(message)


def number(value, label, *, minimum=0, integer=False):
    require(type(value) in (int, float) and (type(value) is int or math.isfinite(value)),
            f'{label}: expected finite number')
    require(not integer or type(value) is int, f'{label}: expected integer')
    require(value >= minimum, f'{label}: must be >= {minimum}')
    return value


def text_value(value, label):
    require(isinstance(value, str) and value.strip() and value.strip() != 'unknown',
            f'{label}: missing or unknown')
    return value


def close(actual, expected, label):
    number(actual, label, minimum=-math.inf)
    require(math.isclose(actual, expected, rel_tol=1e-9, abs_tol=1e-9),
            f'{label}: {actual!r} disagrees with recomputed {expected!r}')


@dataclass(frozen=True)
class Source:
    path: Path
    sha256: str | None
    data: dict | None


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, f'duplicate JSON key: {key}')
        result[key] = value
    return result


def finite_tree(value, label='JSON'):
    if isinstance(value, dict):
        for key, child in value.items():
            finite_tree(child, label + '.' + key)
    elif isinstance(value, list):
        for child in value:
            finite_tree(child, label + '[]')
    elif type(value) is float:
        require(math.isfinite(value), f'{label}: expected finite number')


def read_source(path):
    path = Path(path)
    try:
        raw = path.read_bytes()
    except FileNotFoundError:
        return Source(path, None, None)
    except OSError as exc:
        raise SummaryError(f'{path}: {exc}') from exc
    try:
        data = json.loads(raw, object_pairs_hook=unique_object)
        finite_tree(data)
    except (ValueError, UnicodeError) as exc:
        raise SummaryError(f'{path}: invalid JSON: {exc}') from exc
    require(isinstance(data, dict), f'{path}: expected JSON object')
    return Source(path, hashlib.sha256(raw).hexdigest(), data)


def validate_binary(data):
    """Validate the measured NI schema, including existing 784-input raw rows."""
    p = data['provenance']
    require(p['written_by'] == 'the mnist binary (raw output, not assembled)',
            'provenance.written_by: expected raw mnist binary writer')
    require(isinstance(p['git_commit'], str) and
            re.fullmatch(r'[0-9a-f]{40}', p['git_commit']) is not None,
            'provenance.git_commit: expected full git commit')
    text_value(p['git_modified_tracked_files'], 'provenance.git_modified_tracked_files')
    text_value(p['command_line'], 'provenance.command_line')
    start = number(p['started_unix'], 'provenance.started_unix', minimum=1)
    finish = number(p['finished_unix'], 'provenance.finished_unix', minimum=1)
    require(finish > start, 'provenance timestamps: finish must follow start')
    for field in ('cpu', 'host', 'os'):
        text_value(p['machine'][field], 'provenance.machine.' + field)
    number(p['machine']['logical_cpus'], 'provenance.machine.logical_cpus', minimum=1, integer=True)
    text_value(p['threads'], 'provenance.threads')
    setup, training, measured = data['setup'], data['training'], data['measured']
    require(setup['boot_ks'] == 111, 'setup.boot_ks: expected measured 111')
    require(setup['placement'] == 'when_needed', 'setup.placement: expected when_needed')
    samples = number(training['samples'], 'training.samples', minimum=1, integer=True)
    updates = number(training['updates'], 'training.updates', minimum=1, integer=True)
    counts = measured['total']
    for key in ('rotations', 'relins', 'sse_switches', 'key_switches', 'bootstraps'):
        number(counts[key], 'measured.total.' + key, integer=True)
    for key, value in counts.items():
        number(value, 'measured.total.' + key, integer=True)
    ks, boots = counts['key_switches'], counts['bootstraps']
    require(ks == counts['rotations'] + counts['relins'] + counts['sse_switches'],
            'measured.total.key_switches: counters do not sum')
    require(counts['sse_switches'] == 2 * boots and counts['rotations'] >= 69 * boots
            and counts['relins'] >= 40 * boots, 'measured.total: inconsistent bootstrap counters')
    nonboot = ks - 111 * boots
    require(nonboot >= 0, 'measured.total: negative boot-excluded count')
    for key, expected in (
        ('ks_per_sample_incl_bootstraps', ks / samples),
        ('ks_per_sample_without_bootstraps', nonboot / samples),
        ('bootstraps_per_update', boots / updates),
        ('bootstraps_per_sample', boots / samples),
        ('measured_over_predicted', 1)):
        close(measured[key], expected, 'measured.' + key)
    prediction = data['prediction']
    schedule = prediction['boot_schedule']
    require(isinstance(schedule, list) and len(schedule) == updates and
            all(type(x) is int and x in (0, 1) for x in schedule),
            'prediction.boot_schedule: expected one 0/1 per update')
    require(measured['weight_booted'] == schedule, 'measured.weight_booted: schedule mismatch')
    require(boots == 4 * sum(schedule), 'measured.total.bootstraps: schedule mismatch')
    close(prediction['bootstraps_per_update_avg'], boots / updates,
          'prediction.bootstraps_per_update_avg')
    require(measured['every_update_equals_prediction'] is True,
            'measured.every_update_equals_prediction: prediction checks did not pass')
    number(measured['predicted_ks_this_run'], 'measured.predicted_ks_this_run', integer=True)
    require(measured['predicted_ks_this_run'] == ks, 'measured.predicted_ks_this_run: mismatch')
    # Partial batches in the older raw files have a different denominator in this
    # prediction field; their exact run prediction above remains authoritative.
    number(prediction['predicted_ks_per_sample'], 'prediction.predicted_ks_per_sample', minimum=1)
    times = data['time_s']
    for key in ('train_total', 'epoch_measured', 'wall_total', 'bootstrap_time_total'):
        number(times[key], 'time_s.' + key, minimum=0 if key == 'bootstrap_time_total' else 1e-12)
    require(times['bootstrap_time_total'] <= times['train_total'] <= times['wall_total'],
            'time_s: bootstrap/train/wall times out of order')
    require(times['wall_total'] <= finish - start + 1, 'time_s.wall_total: exceeds provenance duration')
    evaluation = data['evaluation']
    number(evaluation['test_samples'], 'evaluation.test_samples', minimum=1, integer=True)
    for key in ('test_acc_encrypted_trained', 'test_acc_twin', 'test_acc_init'):
        require(number(evaluation[key], 'evaluation.' + key) <= 1, f'evaluation.{key}: exceeds 1')
    for key in ('test_mse_encrypted_trained', 'test_mse_twin'):
        number(evaluation[key], 'evaluation.' + key)
    twin = data['twin']
    number(twin['max_abs_diff'], 'twin.max_abs_diff')
    number(twin['log2_max_abs_diff'], 'twin.log2_max_abs_diff', minimum=-math.inf)
    for key in ('max_abs_w_diff_per_layer', 'max_abs_b_diff_per_layer'):
        require(isinstance(twin[key], list) and len(twin[key]) == 2, f'twin.{key}: expected two layers')
        for value in twin[key]:
            number(value, 'twin.' + key)
    validate_cached_m0(data)
    return data


def read_binary(path):
    """Read and validate raw NI JSON; missing files return a source with data=None."""
    source = read_source(path)
    if source.data is not None:
        try:
            validate_binary(source.data)
        except (KeyError, TypeError, SummaryError) as exc:
            raise SummaryError(f'{source.path}: {exc}') from exc
    return source


def validate_reference(data):
    cfg, seed = data['config_values'], data['per_seed']['0']
    for key, expected in (('batch_size', 48), ('epochs', 10), ('subsample_K', 4800),
                          ('input_resize', 14), ('hidden', [32]), ('bias', False),
                          ('activation', 'x^2 + x')):
        require(cfg[key] == expected, f'ReBoot config_values.{key}: expected {expected!r}')
    text_value(cfg['lr_schedule'], 'ReBoot config_values.lr_schedule')
    for key, expected in (('steps', 1000), ('epochs', 10), ('steps_per_epoch', 100),
                          ('batch_size', 48), ('train_samples_drawn', 4800),
                          ('input_features', 196), ('output_width', 16), ('test_samples', 10000)):
        require(seed[key] == expected, f'ReBoot per_seed.0.{key}: expected {expected}')
    require(seed['seeds']['model_and_epoch_shuffle_seed'] == 0, 'ReBoot initial model seed must be 0')
    require(seed['indices_file'] == 'train_indices.txt', 'ReBoot seed-0 indices mismatch')
    weights = seed['init_weights']
    require(isinstance(weights, list) and len(weights) == 2, 'ReBoot seed-0 weights: expected two layers')
    for weight, name, shape in zip(weights, ('W1.npy', 'W2.npy'), ([196, 32], [32, 16])):
        require(weight['file'] == f'init_weights_seed0/{name}' and weight['shape'] == shape,
                'ReBoot seed-0 weight filename/shape mismatch')
        require(re.fullmatch(r'[0-9a-f]{64}', weight['sha256']) is not None,
                'ReBoot seed-0 weight SHA256 missing')
    lr = seed['lr_per_step']
    require(isinstance(lr, list) and len(lr) == 1000, 'ReBoot seed-0 schedule: expected 1000 steps')
    for value in lr:
        number(value, 'ReBoot seed-0 lr_per_step', minimum=1e-300)
    accuracy = number(seed['twin']['test_acc_argmax10'], 'ReBoot seed-0 accuracy')
    require(accuracy <= 1, 'ReBoot seed-0 accuracy exceeds 1')
    return accuracy


def command_options(command):
    try:
        args = shlex.split(command)
    except ValueError as exc:
        raise SummaryError(f'provenance.command_line: {exc}') from exc
    options = {}
    index = 1
    while index < len(args):
        key = args[index]
        require(key.startswith('--') and key not in options, 'command: invalid or duplicate option ' + key)
        if index + 1 < len(args) and not args[index + 1].startswith('--'):
            options[key] = args[index + 1]
            index += 2
        else:
            options[key] = True
            index += 1
    return options


def validate_full(data, batch, reference=None):
    require(data['network'].startswith('196 -> 32 (z^2 + z) -> 16 (linear)'),
            'network: FULL requires 196 -> 32 -> 16')
    setup, train = data['setup'], data['training']
    for key, expected in (('log_n', 16), ('slots', 32768), ('params', 'n16_boot'),
                          ('batch', batch), ('lanes_per_ciphertext', 64),
                          ('micro_batches_per_update', 1), ('layout_d', 256),
                          ('transpose_keys', 'reuse')):
        require(setup[key] == expected, f'setup.{key}: FULL expected {expected}')
    if reference is not None:
        cfg = reference['config_values']
        close(train['momentum'], cfg['momentum'], 'training.momentum')
        close(train['lr'], cfg['lr_base'] * batch, 'training.lr')
        close(train['weight_decay'], cfg['weight_decay']['fwd'], 'training.weight_decay')
        close(train['weight_decay'], cfg['weight_decay']['local'], 'training.weight_decay')
    updates = 750 if batch == 64 else 1000
    for key, expected in (('epochs', 1), ('subset', 48000), ('samples', 48000), ('updates', updates)):
        require(train[key] == expected, f'training.{key}: FULL expected {expected}')
    require(train['lr_schedule'] != 'constant', 'training.lr_schedule: missing full schedule')
    require(data['evaluation']['test_samples'] == 10000, 'evaluation.test_samples: expected 10000')
    close(data['time_s']['epoch_measured'], data['time_s']['train_total'], 'time_s.epoch_measured')
    close(data['prediction']['predicted_ks_per_sample'],
          data['measured']['total']['key_switches'] / 48000, 'prediction.predicted_ks_per_sample')
    expected_schedule = [0, 1] * 375 if batch == 64 else [1] * 1000
    require(data['prediction']['boot_schedule'] == expected_schedule,
            'prediction.boot_schedule: not the registered FULL schedule')
    options = command_options(data['provenance']['command_line'])
    for key, expected in (('--mode', 'noninteractive'), ('--placement', 'when_needed'),
                          ('--batch', str(batch)), ('--faithful', True), ('--epochs', '1'),
                          ('--transpose', 'reuse'), ('--lr-key', 'lr_per_step')):
        require(options.get(key) == expected, f'command {key}: expected {expected}')
    paths = {
        '--dataset-npy': 'reboot_emlp1_full',
        '--indices': 'reboot_emlp1_full/train_indices.txt',
        '--init-weights': 'reboot_emlp1_full/init_weights_seed0',
        '--init-weights-manifest': 'reboot_emlp1_full/config_values.json',
        '--lr-schedule': 'reboot_emlp1_full/config_values.json',
    }
    for key, suffix in paths.items():
        value = options.get(key)
        require(isinstance(value, str) and (value == suffix or value.endswith('/' + suffix)),
                f'command {key}: expected released seed-0 source {suffix}')
    require(options['--init-weights-manifest'] == options['--lr-schedule'] == train['lr_schedule'],
            'command/training lr_schedule and init-weights-manifest disagree')
    require(options.get('--lr-schedule-batch', '48') == '48', 'command --lr-schedule-batch: expected 48')
    if batch == 64:
        require(options.get('--lr-schedule-batch') == '48', 'command --lr-schedule-batch: missing alignment')
    return data


def ratios(data):
    counts = data['measured']['total']
    samples = data['training']['samples']
    boots = counts['bootstraps']
    nonboot = counts['key_switches'] - boots * 111
    return [
        ('a: own bootstrap costs', (nonboot + boots * 111) / samples, 37 + 4 * 502 / 48),
        ('b: both 502 (model)', (nonboot + boots * 502) / samples, 37 + 4 * 502 / 48),
        ('c: both 111', (nonboot + boots * 111) / samples, 37 + 4 * 111 / 48),
        ('d: boot-excluded NI ablation', nonboot / samples, 37),
    ]


def validate_cached_m0(data):
    rq1 = data.get('rq1', {})
    require(isinstance(rq1, dict), 'rq1: expected object')
    sides = rq1.get('by_reboot_side', {})
    require(isinstance(sides, dict), 'rq1.by_reboot_side: expected object')
    models = [container['m0_model'] for container in (rq1, sides) if 'm0_model' in container]
    keys = ('a_own_bootstrap_costs', 'b_both_502', 'c_both_hekit')
    for model in models:
        require(isinstance(model, dict), 'rq1.m0_model: expected object')
        text_value(model['source'], 'rq1.m0_model.source')
        close(model['ks_per_sample_without_bootstraps'], 37, 'rq1.m0_model.nonboot')
        for key, (_, own, baseline), (heboot, rbboot) in zip(
                keys, ratios(data), ((111, 502), (502, 502), (111, 111))):
            way = model[key]
            same_batch_baseline = 37 + 4 * rbboot / data['setup']['batch']
            for field, expected in (
                ('hekit_boot_ks', heboot), ('reboot_boot_ks', rbboot),
                ('hekit_ks_per_sample', own), ('reboot_ks_per_sample_b48', baseline),
                ('ratio_vs_reboot_b48', own / baseline),
                ('reboot_ks_per_sample_same_b', same_batch_baseline),
                ('ratio_vs_reboot_same_b', own / same_batch_baseline)):
                close(way[field], expected, 'rq1.m0_model.' + key + '.' + field)


def md(value):
    return str(value).replace('`', "'").replace('|', r'\|').replace('\n', ' ')


def render_summary(full1, full2, reboot_config):
    """Validate all inputs before returning any Markdown."""
    reference = read_source(reboot_config)
    ref_accuracy = None
    if reference.data is not None:
        try:
            ref_accuracy = validate_reference(reference.data)
        except (KeyError, TypeError, SummaryError) as exc:
            raise SummaryError(f'{reference.path}: {exc}') from exc
    sources = [read_binary(full1), read_binary(full2)]
    for source, batch in zip(sources, (64, 48)):
        if source.data is not None:
            try:
                validate_full(source.data, batch, reference.data)
            except (KeyError, TypeError, AttributeError, SummaryError) as exc:
                raise SummaryError(f'{source.path}: {exc}') from exc
    lines = ['# RQ1 full-length evidence (Task 42 preparation)', '',
             'Overall RQ1: pending. This summary cannot establish an unconditional pass.', '',
             'Checks validate declared provenance and internal consistency. A recorded git HEAD '
             'does not attest the built binary; command paths do not prove input file contents.', '',
             'ReBoot full CPU timing: unmeasured (no 784-input subset timing is reused).',
             'Security comparability: unverified (requires matched estimator evidence and distributions).',
             'ReBoot B=48, 4 bootstraps/update; nonboot KS/sample = 37 (model); '
             'bootstrap KS = 502 (model). hekit bootstrap KS = 111 (measured).',
             'd is the measured NI count with bootstrap KS removed, '
             'not a separately executed interactive run.', '']
    lines += [f'ReBoot released config: {"values-only reference" if reference.data else "unmeasured"}',
              f'- Input: `{md(reference.path)}`; SHA256: `{reference.sha256 or "unmeasured"}`']
    if ref_accuracy is not None:
        lines.append(f'- seed-0 reference accuracy: {ref_accuracy:.6f} (per_seed.0.twin.test_acc_argmax10)')
    for index, source in enumerate(sources, 1):
        lines += ['', f'## FULL-{index}: {"measured raw binary output" if source.data else "unmeasured"}',
                  f'- Input: `{md(source.path)}`; SHA256: `{source.sha256 or "unmeasured"}`']
        if source.data is None:
            continue
        data = source.data
        p, counts, train = data['provenance'], data['measured']['total'], data['training']
        if data.get('synthetic_fixture'):
            lines.append('- SYNTHETIC fixture; not a real measurement.')
        lines += [f'- Writer: {md(p["written_by"])}; commit `{p["git_commit"]}`; tracked changes: '
                  f'`{md(p["git_modified_tracked_files"])}`.',
                  f'- Machine: {md(p["machine"]["host"])} / {md(p["machine"]["cpu"])} / '
                  f'{md(p["machine"]["os"])}; threads: {md(p["threads"])}.',
                  f'- Unix timestamps: {p["started_unix"]} → {p["finished_unix"]}.',
                  f'- Command: `{md(p["command_line"])}`',
                  '- initial weights: seed 0 (command + released config)' if reference.data else
                  '- initial weights: command declares seed 0; released config unmeasured.',
                  '- training.seed is the encryption RNG seed; it is not the initial-weight seed.',
                  f'- 196 → 32 → 16; N=2^16; B={data["setup"]["batch"]}; '
                  f'{train["updates"]} updates; 48000 samples (4800 × 10 passes).',
                  '- binary epochs=1 traverses flattened indices; train_total covers all 10 passes.',
                  f'- Training time: {data["time_s"]["train_total"]:.6f} s; '
                  f'{data["time_s"]["train_total"] / 48000:.6f} s/sample; '
                  f'{data["time_s"]["train_total"] / 10:.6f} s/pass (arithmetic mean).',
                  f'- Measured integer counts: {counts["key_switches"]} KS, '
                  f'{counts["bootstraps"]} bootstraps; '
                  f'{counts["key_switches"] - 111 * counts["bootstraps"]} nonboot KS; '
                  f'{counts["bootstraps"] / train["updates"]:.6f} bootstraps/update.',
                  '- Schedule and every-update prediction checks: verified.',
                  '', '| Way | hekit KS/sample | ReBoot KS/sample (model) | Ratio | ≤0.70 KS gate |',
                  '|---|---:|---:|---:|---|']
        for name, own, reboot in ratios(data):
            ratio = own / reboot
            lines.append(f'| {name} | {own:.6f} | {reboot:.6f} | {ratio:.6f} | '
                         f'{"PASS" if ratio <= 0.70 else "FAIL"} |')
        evaluation = data['evaluation']
        lines += ['', f'- FULL-{index} accuracy ({"separate" if index == 1 else "measured"}): '
                  f'{evaluation["test_acc_encrypted_trained"]:.6f}; twin: '
                  f'{evaluation["test_acc_twin"]:.6f}; maximum weight/bias error: '
                  f'{data["twin"]["max_abs_diff"]:.8g}.']
    lines.append('')
    if sources[1].data is None or ref_accuracy is None:
        lines.append('FULL-2 accuracy gate: pending (needs full result and released seed-0 values).')
    else:
        delta = sources[1].data['evaluation']['test_acc_encrypted_trained'] - ref_accuracy
        passed = abs(delta) <= 0.005 + 1e-12
        lines.append(f'FULL-2 accuracy gate: {"PASS" if passed else "FAIL"}; '
                     f'difference {delta * 100:+.6f} percentage points; tolerance ±0.5 percentage point.')
    return '\n'.join(lines) + '\n'


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--full-1', required=True, help='raw FULL-1 binary JSON (B64)')
    parser.add_argument('--full-2', required=True, help='raw FULL-2 binary JSON (B48 in 64 lanes)')
    parser.add_argument('--reboot-config', required=True, help='released values-only config_values.json')
    args = parser.parse_args(argv)
    try:
        result = render_summary(args.full_1, args.full_2, args.reboot_config)
    except SummaryError as exc:
        print(f'error: {exc}', file=sys.stderr)
        return 1
    print(result, end='')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
