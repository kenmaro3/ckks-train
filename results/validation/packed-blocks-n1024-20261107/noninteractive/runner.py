"""Root-released single NI phase. No I dispatch or retry path."""
from pathlib import Path
import datetime
import hashlib
import json
import os
import shutil
import subprocess
import time

LEDGER = Path('.superpowers/sdd/2026-11-04-he-training-implementation-plan-m3-m5')
COMMIT = 'ad607daf3e53c3763c4c5db25701f2ede5115154'
REVIEW_SHA = '6bb24e383bfd8a82acb6e6335f41db2456b970f3daba6e8f3fb1e2dcbe682f3b'
MANIFEST_SHA = '4788e0af8128c514b36a11cb651fa018ec39db8b7e140425d08be16c4915796e'
EXE_SHA = '91dfd7ef7bd451cc8e423c1fc7cd4241c379c22e777dcf30c6e15c73b26ee5c8'
EXE = Path('target/debug/deps/packed_blocks-3f7d0726ee92034c')
RUN = LEDGER / 'task-54-d4c3-ni-primary-attempt-1'


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


manifest = LEDGER / 'task-54-d4c3-fix1-source-manifest.json'
assert digest(manifest) == MANIFEST_SHA
assert digest(LEDGER/'task-54-d4c3-fix1-rereview.md') == REVIEW_SHA
assert subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip() == COMMIT
frozen = json.loads(manifest.read_text())
for item in frozen['owned_source']:
    assert digest(item['path']) == item['sha256'], item['path']
assert digest(EXE) == EXE_SHA
assert digest(LEDGER/'task-54-d4c3-i-runtime-audit.md') == '454672effe66e62892f9793da0e8499489833a41f2fe5c82cd3dcf8085e98692'
assert json.loads((LEDGER/'task-54-d4c3-i-primary-attempt-1/runner.json').read_text())['returncode'] == 0
RUN.mkdir(exist_ok=False)
shutil.copyfile(EXE, RUN/'frozen-test-executable')
shutil.copyfile(manifest, RUN/'source-manifest.json')
command = ['cargo', 'test', '-p', 'ckks-train-nn', '--test', 'packed_blocks',
           'block_ni_original_fixture', '--', '--ignored', '--exact',
           '--nocapture', '--test-threads=1']
env = os.environ.copy()
env.update(CARGO_INCREMENTAL='0', CARGO_BUILD_JOBS='8', RAYON_NUM_THREADS='8')
meta = {'kind': 'external_runner_provenance_not_scientific_result',
        'phase': 'NI_only_single_attempt', 'source_commit': COMMIT,
        'source_manifest_sha256': MANIFEST_SHA, 'review_sha256': REVIEW_SHA,
        'runner_sha256': digest(__file__), 'executable_before_sha256': digest(EXE),
        'command': command, 'cwd': str(Path.cwd()),
        'environment': {k: env[k] for k in ('CARGO_INCREMENTAL', 'CARGO_BUILD_JOBS', 'RAYON_NUM_THREADS')},
        'host': dict(zip(('sysname','nodename','release','version','machine'), os.uname())),
        'host_scope': 'Mac correctness fixture, not quadruped timing or production security',
        'git_status_before': subprocess.check_output(['git', 'status', '--porcelain'], text=True),
        'started_unix': time.time(), 'started_local': datetime.datetime.now().astimezone().isoformat()}


def record():
    (RUN/'runner.json').write_text(json.dumps(meta, indent=2)+'\n')


record()
with (RUN/'stdout.txt').open('x') as out, (RUN/'stderr.txt').open('x') as err:
    child = subprocess.Popen(command, env=env, stdout=out, stderr=err)
    meta['child_pid'] = child.pid
    record()
    code = child.wait()
meta.update(returncode=code, finished_unix=time.time(),
            finished_local=datetime.datetime.now().astimezone().isoformat(),
            executable_after_sha256=digest(EXE), executable_unchanged=digest(EXE)==EXE_SHA,
            source_unchanged=all(digest(x['path'])==x['sha256'] for x in frozen['owned_source']))
for name in ('stdout.txt', 'stderr.txt'):
    meta[name+'_sha256'] = digest(RUN/name)
    meta[name+'_bytes'] = (RUN/name).stat().st_size
record()
print(json.dumps(meta, indent=2))
print('\n'.join(line for line in (RUN/'stdout.txt').read_text().splitlines()
                if line.startswith(('NI_END ', 'test result:', 'running ', 'test block_'))))
assert meta['executable_unchanged'] and meta['source_unchanged'], 'frozen identity changed'
raise SystemExit(code)
