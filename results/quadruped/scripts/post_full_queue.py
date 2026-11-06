#!/usr/bin/env python3
"""One-shot post-FULL queue. Python stdlib; only run_job owns resource-heavy children.

Usage: post_full_queue.sh --config /absolute/config.json
Required configuration is documented in docs/post-full-queue.md.
The atomic state directory is deliberately never reused, even after interruption.
"""
import argparse
from contextlib import contextmanager
from datetime import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import shlex
import signal
import subprocess
import sys
import time

FULL_JOBS = ('R-41-FULL-1', 'R-41-FULL-2')
PINS = {'r47': 'bd5cc49', 'r48': '0e07561', 'r52': '9305d2e'}
FULL_PINS = {'r47': 'bd5cc493db166a282d26b05b1e069f2a0ee629ef',
             'r48': '0e075618ab6e28b0cf260f98feb7f80ee1ccf1f3',
             'r52': '9305d2e1f4f9999296cd4b4aaf55ea631fbca78e'}
THREADS = 12


class QueueError(RuntimeError):
    pass


def atomic_json(path, value):
    path = Path(path)
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(json.dumps(value, indent=2) + '\n')
    tmp.replace(path)


def epoch(value):
    return float(value) if isinstance(value, (int, float)) else datetime.fromisoformat(value).timestamp()


def process_ticks(pid):
    try:
        return Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()[19]
    except FileNotFoundError:
        return None


def gates(cfg):
    """Missing evidence means wait; contradictory/stale terminal evidence means stop."""
    ended = []
    for name in FULL_JOBS:
        out = Path(cfg['runs']) / 'out' / name
        post, meta = out / 'post.txt', out / 'meta.json'
        if (out / 'partial.txt').exists():
            raise QueueError(f'{name}: FULL was partial')
        if not post.exists() or not meta.exists():
            return False
        lines = post.read_text().splitlines()
        if not lines or not re.fullmatch(r'JOB_DONE rc=\d+', lines[-1]):
            return False  # run_job writes post snapshot before final marker
        m = json.loads(meta.read_text())
        if lines[-1] != 'JOB_DONE rc=0' or m['exit_status'] != 0 or m.get('watchdog_killed'):
            raise QueueError(f'{name}: FULL did not succeed')
        start, end = epoch(m['start']), epoch(m['end'])
        commits = ' '.join(m.get('hekit_synced_commit', []))
        if (m['job'] != name or m['workdir'] != cfg['full_tree'] or
                not re.search(r'\b(?:commit|full) ' + re.escape(cfg['full_commit']) + r'[0-9a-f]*\b', commits) or
                start < cfg['not_before_epoch'] or end < start or end > time.time() + 60 or
                post.stat().st_mtime < cfg['not_before_epoch']):
            raise QueueError(f'{name}: stale or mismatched FULL evidence')
        ended.append(end)
    ticks = process_ticks(int(cfg['chain_pid']))
    if ticks is not None:
        if ticks != str(cfg['chain_start_ticks']):
            raise QueueError('chain PID was reused; operator review required')
        return False
    terminal = Path(cfg['chain_log'])
    if not terminal.exists():
        return False
    lines = terminal.read_text().splitlines()
    if terminal.stat().st_mtime < max(ended + [cfg['not_before_epoch']]):
        raise QueueError('stale chain terminal evidence')
    return bool(lines and lines[-1] == 'CHAIN10_DONE')


def wait_gates(cfg):
    deadline = time.monotonic() + cfg.get('max_wait_seconds', 5 * 86400)
    while not gates(cfg):
        if time.monotonic() >= deadline:
            raise QueueError('FULL/chain gate timeout; no work started')
        time.sleep(min(cfg.get('poll_seconds', 30), max(0, deadline - time.monotonic())))


@contextmanager
def queue_lock(state):
    state = Path(state)
    state.parent.mkdir(parents=True, exist_ok=True)
    try:
        state.mkdir()  # exclusive atomic ownership; never automatically remove a stale lock
    except FileExistsError:
        raise QueueError(f'queue already claimed: {state}')
    atomic_json(state / 'owner.json', {'pid': os.getpid(), 'started': time.time()})
    try:
        yield
    except BaseException as exc:
        atomic_json(state / 'terminal.json', {'status': 'failed', 'error': str(exc), 'ended': time.time()})
        raise


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def copy_manifest_data(source, destination):
    source, destination = Path(source), Path(destination)
    manifest = source / 'data_manifest.json'
    files = json.loads(manifest.read_text())['files']
    for entry in files:
        name = entry['file']
        if Path(name).name != name or (source / name).is_symlink() or digest(source / name) != entry['sha256_npy_file']:
            raise QueueError(f'data SHA256 mismatch or unsafe path: {name}')
    destination.mkdir(parents=True, exist_ok=True)
    shutil.copy2(manifest, destination / manifest.name)
    for entry in files:
        target = destination / entry['file']
        shutil.copy2(source / entry['file'], target)
        if digest(target) != entry['sha256_npy_file']:
            raise QueueError(f'copied data mismatch: {target}')


def descendants(pid):
    """Linux direct child walk; used exclusively for this runner's own time tree."""
    children = []
    for path in Path('/proc').glob('[0-9]*/stat'):
        try:
            fields = path.read_text().rsplit(')', 1)[1].split()
            if int(fields[1]) == pid:
                child = int(path.parent.name)
                children.extend(descendants(child))
                children.append(child)
        except (FileNotFoundError, ProcessLookupError):
            pass
    return children


def partial_projection(text):
    rows = re.findall(r'^mnist: epoch .* update .*\): ([0-9.]+) s,.*elapsed ([0-9]+) s', text, re.M)
    if len(rows) < 5:
        return None
    times = [float(row[0]) for row in rows[:5]]
    return float(rows[4][1]) - sum(times) + sum(times) / 5 * 21


def stop_own_tree(runner_pid, time_pid):
    original = process_ticks(time_pid)
    if not original or time_pid not in descendants(runner_pid):
        raise QueueError('partial watcher refuses non-child PID')
    victims = [(p, process_ticks(p)) for p in descendants(time_pid)] + [(time_pid, original)]
    if process_ticks(time_pid) != original:
        raise QueueError('partial watcher time PID identity changed')
    for victim, ticks in victims[:-1]:
        if ticks and process_ticks(victim) == ticks:
            try:
                os.kill(victim, signal.SIGTERM)
            except ProcessLookupError:
                pass
    return victims


def wait_identities_gone(identities, timeout=60, poll=0.25):
    """Wait for every captured process identity, including natural time-parent exit.

    A reused PID is no longer the captured identity and is never signalled here.
    Timeout stops the queue: no following build or training may start.
    """
    deadline = time.monotonic() + timeout
    while any(process_ticks(pid) == ticks for pid, ticks in identities if ticks):
        if time.monotonic() >= deadline:
            raise QueueError('partial tree still exists after bounded termination wait; no next job')
        time.sleep(min(poll, max(0, deadline - time.monotonic())))


def validate_terminal(metadata, out, name, tree, command, commit, gpu, before, after):
    start, end = epoch(metadata['start']), epoch(metadata['end'])
    synced = metadata.get('hekit_synced_commit', [])
    if (metadata.get('job') != name or metadata.get('workdir') != str(tree) or
            'full ' + commit not in synced or 'dirty_files 0' not in synced or
            metadata.get('threads_cap') != THREADS or metadata.get('gpu_job') is not gpu or
            metadata.get('nice') != 10 or shlex.split(metadata.get('command', '')) != list(map(str, command)) or
            not math.isfinite(start) or not math.isfinite(end) or
            start < math.floor(before) or end < start or end > math.ceil(after)):
        raise QueueError(f'{name}: terminal identity does not match actual invocation')
    for filename in ('post.txt', 'meta.json', 'cmd.txt'):
        path = out / filename
        if not path.exists() or not before <= path.stat().st_mtime <= after + 1:
            raise QueueError(f'{name}: missing/stale terminal file {filename}')
    if shlex.split((out / 'cmd.txt').read_text()) != list(map(str, command)):
        raise QueueError(f'{name}: command record does not match actual invocation')


def validate_output(data, command, commit, gpu, before, after):
    """Only fields emitted by the pinned provenance.rs and binary writers are used."""
    if not isinstance(data, dict) or not isinstance(data.get('provenance'), dict):
        raise QueueError('output lacks provenance object')
    p = data['provenance']
    binary = Path(str(command[0])).name
    if binary not in ('ablation', 'mnist'):
        raise QueueError('unexpected JSON writer command')
    expected_commit = FULL_PINS['r48' if binary == 'ablation' else 'r52']
    expected_build = {'debug_assertions': False,
        'package_version': '0.0.0' if binary == 'ablation' else '1.0.0',
        'target_os': 'linux', 'target_arch': os.uname().machine,
        'features': {'parallel': True, 'cuda': binary == 'ablation'}}
    start, end = p.get('started_unix'), p.get('finished_unix')
    if (data.get('all_checks_pass') is not True or commit != expected_commit or
            p.get('git_commit') != commit or p.get('git_modified_tracked_files') != 'none' or
            p.get('written_by') != f'the {binary} binary (raw output, not assembled)' or
            p.get('command_line') != ' '.join(map(str, command)) or
            p.get('build') != expected_build or p.get('parallel_feature') is not True or
            not isinstance(start, (int, float)) or not isinstance(end, (int, float)) or
            not math.isfinite(start) or not math.isfinite(end) or not before <= start <= end <= after):
        raise QueueError('output checks/provenance do not match pinned invocation')
    if binary == 'ablation':
        info = p.get('gpu')
        if gpu:
            if (not isinstance(info, dict) or not isinstance(info.get('name'), str) or not info['name'] or
                    not all(isinstance(info.get(k), (int, float)) and math.isfinite(info[k]) and info[k] > 0
                            for k in ('free_mib_at_start', 'total_mib'))):
                raise QueueError('output lacks actual GPU provenance')
        elif 'gpu' not in p or info is not None:
            raise QueueError('CPU ablation GPU provenance must be null')


class Queue:
    def __init__(self, cfg):
        self.cfg = cfg
        self.runs = Path(cfg['runs'])
        self.state = Path(cfg.get('state', self.runs / 'post-full-v1'))
        self.runner = Path(cfg.get('runner', self.runs / 'run_job.sh'))
        self.state.mkdir(parents=True, exist_ok=True)
        self.records = []
        self.tree_commits = {}
        self.env = os.environ.copy()
        self.env.update({k: str(THREADS) for k in ('OMP_NUM_THREADS', 'RAYON_NUM_THREADS',
            'CARGO_BUILD_JOBS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS')})
        self.env['PATH'] = str(Path.home() / '.cargo/bin') + ':/usr/local/cuda-13.1/bin:' + self.env.get('PATH', '')
        self.env['LD_LIBRARY_PATH'] = '/usr/local/cuda-13.1/lib64:' + self.env.get('LD_LIBRARY_PATH', '')

    def disk(self, directory):
        free = shutil.disk_usage(directory).free
        if free < self.cfg.get('min_free_bytes', 12 * 1024 ** 3):
            raise QueueError(f'insufficient disk before clone/build: {free} bytes')

    def clone(self, family):
        parent = Path(self.cfg['clone_root'])
        parent.mkdir(parents=True, exist_ok=True)
        self.disk(parent)
        dest = parent / f'hekit-post-full-{family}'
        if dest.exists():
            raise QueueError(f'refusing existing clone: {dest}')
        bundle = Path(self.cfg['bundle'])
        if digest(bundle) != self.cfg['bundle_sha256']:
            raise QueueError('bundle SHA256 mismatch')
        # Nice applies even to clone/checkouts; training/builds go through run_job below.
        subprocess.run(['nice', '-n', '10', 'git', 'clone', '--no-checkout', str(bundle), str(dest)], check=True, env=self.env)
        subprocess.run(['nice', '-n', '10', 'git', '-C', str(dest), 'checkout', '--detach', PINS[family]], check=True, env=self.env)
        commit = subprocess.check_output(['git', '-C', str(dest), 'rev-parse', 'HEAD'], text=True).strip()
        if commit != FULL_PINS[family]:
            raise QueueError('incorrect checkout')
        self.tree_commits[str(dest)] = commit
        (dest / '.synced-commit').write_text(f'commit {PINS[family]}\nfull {commit}\ndirty_files 0\n')
        return dest

    def job(self, name, tree, gpu, command, output=None, partial=None, env_extra=None):
        out = self.runs / 'out' / name
        commit = self.tree_commits.get(str(tree))
        if commit not in FULL_PINS.values():
            raise QueueError(f'{name}: worktree was not created at a registered pin')
        if out.exists():
            raise QueueError(f'refusing pre-existing job evidence: {name}')
        env = self.env | {'SYNCED_COMMIT_FILE': str(tree / '.synced-commit')} | (env_extra or {})
        before = time.time()
        process = subprocess.Popen(['bash', str(self.runner), name, str(tree), str(THREADS), str(int(gpu)), '--', *map(str, command)], env=env)
        partial_stopped = False
        projection = None
        log = self.runs / 'logs' / (name + '.log')
        while process.poll() is None:
            if partial and not partial_stopped and log.exists():
                if projection is None:
                    projection = partial_projection(log.read_text())
                    if projection is not None:
                        (out / 'projection.txt').write_text(f'after 5 updates: {projection:.0f} s / 21 updates; limit {partial} s\n')
                if projection is not None and projection > partial:
                    pid_file = out / 'time.pid'
                    if pid_file.exists():
                        pid = int(pid_file.read_text())
                        victims = stop_own_tree(process.pid, pid)
                        atomic_json(out / 'partial-identities.json', victims)
                        (out / 'partial.txt').write_text(f'partial: after >=5 updates; projected {projection:.0f} > {partial} s\n')
                        partial_stopped = True
                        wait_identities_gone(victims, self.cfg.get('partial_stop_timeout_seconds', 60))
            time.sleep(self.cfg.get('job_poll_seconds', 5))
        wrapper_rc = process.wait()
        after = time.time()
        post, meta = out / 'post.txt', out / 'meta.json'
        if not post.exists() or not meta.exists():
            raise QueueError(f'{name}: runner missing terminal provenance (wrapper rc={wrapper_rc})')
        lines = post.read_text().splitlines()
        m = re.fullmatch(r'JOB_DONE rc=(\d+)', lines[-1] if lines else '')
        metadata = json.loads(meta.read_text())
        if not m or int(m[1]) != metadata['exit_status']:
            raise QueueError(f'{name}: inconsistent/stale terminal provenance')
        validate_terminal(metadata, out, name, tree, command, commit, gpu, before, after)
        rc = int(m[1])
        text = log.read_text() if log.exists() else ''
        skips = [line for line in text.splitlines() if 'SKIP' in line]
        oom = bool(re.search(r'device error|out of memory', text, re.I)) and rc != 0
        status = ('partial' if partial_stopped else 'failed: device OOM' if oom else
                  'skipped' if rc == 3 and skips else 'failed' if rc else
                  'done_with_skips' if skips else 'done')
        if metadata.get('watchdog_killed'):
            status = 'failed: RSS watchdog'
        result = {'job': name, 'rc': rc, 'wrapper_rc': wrapper_rc, 'status': status,
                  'skip': bool(skips), 'skip_lines': skips, 'command': list(map(str, command)),
                  'workdir': str(tree), 'log': str(log),
                  'evidence_lines': [line for line in text.splitlines() if re.search(r'SKIP|max.*W_gpu.*W_cpu|test result:', line)]}
        if output and rc == 0:
            path = tree / output
            try:
                data = json.loads(path.read_text())
                if not before <= path.stat().st_mtime <= after + 1:
                    raise QueueError('stale output file')
                validate_output(data, command, commit, gpu, before, after)
                result['output_sha256'] = digest(path)
                result['checks'] = data.get('checks')
                result['measured'] = data.get('measured')
                result['handoff_bootstraps'] = data.get('twin', {}).get('handoff_bootstraps')
            except (FileNotFoundError, json.JSONDecodeError, QueueError, TypeError, ValueError) as exc:
                result['status'] = 'failed: output checks/provenance'
                result['output_error'] = str(exc)
        self.records.append(result)
        atomic_json(self.state / (name + '.json'), result)
        return result

    def build(self, name, tree, args):
        self.disk(tree)
        result = self.job(name, tree, False, ['cargo', *args])
        if result['rc'] or result['status'] != 'done':
            raise QueueError(f'{name}: build failed; no dependent runs')

    def r47(self, tree):
        self.build('post-full-r47-build', tree, ['build', '-p', 'hekit-cuda', '--features', 'cuda', '--release', '--example', 'batch_streams'])
        self.job('R-47-3-example', tree, True, ['target/release/examples/batch_streams'], env_extra={'HEKIT_BATCH_REPS': '30'})
        nsys = Path('/usr/local/cuda-13.1/bin/nsys')
        if nsys.is_file() and os.access(nsys, os.X_OK):
            (tree / 'results/quadruped/raw/R-47-3').mkdir(parents=True, exist_ok=True)
            self.job('R-47-3-nsys', tree, True, [str(nsys), 'profile', '--trace=cuda', '--force-overwrite=true', '-o', 'results/quadruped/raw/R-47-3/r47_3_batch', 'target/release/examples/batch_streams'], env_extra={'HEKIT_BATCH_REPS': '5'})
        else:
            atomic_json(self.state / 'R-47-3-nsys.json', {'status': 'skipped: nsys unavailable'})
        self.job('R-47-3-tests', tree, True, ['cargo', 'test', '-p', 'hekit-cuda', '--features', 'cuda', '--release', '--', '--ignored', '--nocapture', '--test-threads=1'])

    def r48(self, tree):
        mnist = str(tree / 'data/mnist')
        self.job('R-48-1', tree, True, ['cargo', 'test', '-p', 'hekit-bench', '--features', 'cuda', '--release', '--test', 'mnist_cuda', '--test', 'iris_cuda', '--', '--ignored', '--nocapture', '--test-threads=1'], env_extra={'HEKIT_MNIST_DIR': mnist, 'HEKIT_TASK48_OUT': 'results/quadruped/raw/R-48-1'})
        def command(backend, opt, output, full=False):
            return ['target/release/ablation', '--backend', backend, '--params', 'n16_train', *([] if full else ['--top-level', '7']), '--optimizer', opt, '--iters', '3' if full else '5', '--warmup', '1', '--data', mnist, '--out', output]
        gpuout = 'results/quadruped/ablation_cuda.json'
        nag = self.job('R-48-2', tree, True, command('cuda', 'nag', gpuout), output=gpuout)
        opt = 'nag'
        if nag['rc'] == 3 and nag['skip'] and nag['status'] == 'skipped':
            sgdout = 'results/quadruped/ablation_cuda_sgd.json'
            sgd = self.job('R-48-2-sgd', tree, True, command('cuda', 'sgd', sgdout), output=sgdout)
            opt = 'nag' if sgd['rc'] == 3 and sgd['skip'] else 'sgd'
        atomic_json(self.state / 'R-48-optimizer.json', {'cpu_optimizer': opt})
        cpuout = 'results/quadruped/ablation_cpu_l7' + ('_sgd' if opt == 'sgd' else '') + '.json'
        self.job('R-48-3', tree, False, command('cpu', opt, cpuout), output=cpuout)
        cpuout = 'results/quadruped/ablation_cpu.json'
        self.job('R-48-4', tree, False, command('cpu', 'nag', cpuout, full=True), output=cpuout)

    def r52(self, tree):
        self.build('post-full-r52-build', tree, ['build', '--release', '-p', 'hekit-bench', '--features', 'parallel', '--bins'])
        data = 'results/quadruped/data/reboot_emlp1_full'
        copy_manifest_data(Path(self.cfg['full_tree']) / data, tree / data)
        for number, blocks, estimate in [(2, '64', 1.7), (3, '128,64', 3.2)]:
            raw = 'results/quadruped/raw/reboot_emlp23_full'
            output = f'results/quadruped/mnist196_emlp{number}_ni_b48_subset1008_seed0.json'
            self.job(f'R-52-{number - 1}', tree, False, ['target/release/mnist', '--mode', 'noninteractive', '--placement', 'when_needed', '--batch', '48', '--faithful', '--blocks', blocks, '--hidden', '32', '--dataset-npy', data, '--indices', f'results/reboot_emlp{number}_seed0_steps1-21_indices.txt', '--init-weights', f'{raw}/init_weights_emlp{number}_seed0', '--init-weights-manifest', f'{raw}/config_values.json', '--epochs', '1', '--transpose', 'reuse', '--out', output], output=output, partial=estimate * 2 * 3600)

    def execute(self):
        wait_gates(self.cfg)
        atomic_json(self.state / 'gate-released.json', {'at': time.time(), 'config': self.cfg})
        r47 = self.clone('r47')
        self.r47(r47)
        r48 = self.clone('r48')
        # Copy only the four raw IDX data files, and validate hashes from the pinned fetch script.
        for line in (r48 / 'scripts/fetch_mnist.sh').read_text().splitlines():
            fields = line.split()
            if len(fields) == 3 and re.fullmatch('[0-9a-f]{64}', fields[2]):
                src = Path(self.cfg['full_tree']) / 'data/mnist' / fields[0]
                dst = r48 / 'data/mnist' / fields[0]
                if src.is_symlink() or digest(src) != fields[2]:
                    raise QueueError(f'MNIST SHA256 mismatch: {src}')
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dst)
                if digest(dst) != fields[2]: raise QueueError('MNIST copy mismatch')
        self.build('post-full-r48-build', r48, ['build', '-p', 'hekit-bench', '--features', 'cuda', '--release', '--bins', '--tests'])
        self.r48(r48)
        r52 = self.clone('r52')
        self.r52(r52)
        failed = any(r['status'].startswith(('failed', 'partial')) for r in self.records)
        atomic_json(self.state / 'terminal.json', {'status': 'completed_with_failures' if failed else 'completed', 'jobs': self.records, 'ended': time.time()})
        return 1 if failed else 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    args = parser.parse_args()
    cfg = json.loads(Path(args.config).read_text())
    required = ('runs', 'full_tree', 'full_commit', 'not_before_epoch', 'chain_pid', 'chain_start_ticks', 'chain_log', 'bundle', 'bundle_sha256', 'clone_root', 'state')
    if any(key not in cfg for key in required):
        raise QueueError('missing required configuration')
    if sys.platform != 'linux':
        raise QueueError('production queue requires Linux /proc')
    full = Path(cfg['full_tree']).resolve()
    for key in ('runs', 'state', 'clone_root'):
        path = Path(cfg[key])
        if not path.is_absolute() or path.resolve().is_relative_to(full):
            raise QueueError(f'{key}: must be absolute and outside the FULL clone')
    # Atomic lock before waiting. Queue.__init__ must never create the state first.
    with queue_lock(cfg['state']):
        return Queue(cfg).execute()


if __name__ == '__main__':
    try:
        sys.exit(main())
    except (QueueError, OSError, ValueError, KeyError, subprocess.SubprocessError) as exc:
        print(f'post-full queue stopped: {exc}', file=sys.stderr)
        sys.exit(2)
