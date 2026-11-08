"""Read back the fixed candidate, capture CI, and create distribution metadata."""
import gzip
import hashlib
import json
import pathlib
import shutil
import subprocess
import tarfile
import zipfile
from datetime import datetime, timezone

ROOT = pathlib.Path(__file__).resolve().parents[3]
RECORD = pathlib.Path(__file__).resolve().parent
SHA = 'e93cec434652a8c9f8c48becc6eb3045a3f78a7f'
RUN = '42484163158'
BASE = pathlib.Path(subprocess.check_output([
    'python3', str(ROOT / 'tools/release/_ext/storage_root.py'), 'archives'
], text=True).strip())
BUNDLE = BASE / 'research-preview2-e93cec4'


def digest(data):
    return hashlib.sha256(data).hexdigest()


def git(*args):
    return subprocess.check_output(['git', '-C', str(ROOT), *args])


def save(path, value):
    if path.exists():
        assert json.loads(path.read_text()) == value, path
        return
    with path.open('x') as stream:
        json.dump(value, stream, indent=2)
        stream.write('\n')


ci = json.loads(subprocess.check_output([
    'gh', 'run', 'view', RUN, '--repo', 'kenmaro3/ckks-train',
    '--json', 'status,conclusion,headSha,jobs,url'
]))
save(RECORD / 'ci-completed.json', ci)
assert ci['headSha'] == SHA and ci['status'] == 'completed'
assert ci['conclusion'] == 'success' and len(ci['jobs']) == 6
assert all(j['conclusion'] == 'success' for j in ci['jobs'])
if not (RECORD / 'ci-test.log.gz').exists():
    log = subprocess.check_output([
        'gh', 'run', 'view', RUN, '--repo', 'kenmaro3/ckks-train',
        '--job', str(next(j['databaseId'] for j in ci['jobs'] if j['name'] == 'test')),
        '--log'])
    with (RECORD / 'ci-test.log.gz').open('xb') as output:
        output.write(gzip.compress(log, mtime=0))

inventory = json.loads((BUNDLE / 'source-inventory.json').read_text())
entries = {}
for row in git('ls-tree', '-rz', SHA).split(b'\0'):
    if row:
        info, name = row.split(b'\t', 1)
        mode, kind, oid = info.decode().split()
        assert kind == 'blob'
        entries[name.decode()] = (mode, oid)
expected_inventory = {e['path']: e for e in inventory}
assert len(entries) == len(expected_inventory) == len(inventory) == 6113
with tarfile.open(BUNDLE / 'ckks-train-e93cec4-source.tar.gz') as tar:
    for member in tar:
        if member.isdir():
            continue
        assert member.isfile() and member.name.startswith('ckks-train-e93cec4/')
        path = member.name.removeprefix('ckks-train-e93cec4/')
        mode, oid = entries.pop(path)
        data = tar.extractfile(member).read()
        assert hashlib.sha1(b'blob ' + str(len(data)).encode() + b'\0' + data).hexdigest() == oid
        assert member.mode == int(mode, 8) & 0o777
        assert expected_inventory.pop(path) == dict(
            path=path, mode=mode, git_blob=oid, bytes=len(data), sha256=digest(data))
assert not entries and not expected_inventory

paper_record = ROOT / 'results/validation/paper-full2-r47-20261108'
inputs = json.loads((paper_record / 'direct-build-meta.json').read_text())['paper_inputs_sha256']
for path, expected in inputs.items():
    assert digest(git('show', SHA + ':' + path)) == expected, path
assert len(inputs) == 18
pdf = BUNDLE / 'ckks-train-paper-preview.pdf'
package = BUNDLE / 'ckks-train-arxiv-source.zip'
assert digest(pdf.read_bytes()) == '9d896a45296754a22de0ba4a4daa5f832a7900f689b62363588a02d2eb22699c'
assert digest(package.read_bytes()) == '2fa463bb471257b89d83414694de18f758bb40459e1b3e5e0f813297252a01c2'
package_manifest = json.loads((paper_record / 'package-manifest.json').read_text())
with zipfile.ZipFile(package) as archive:
    assert len(archive.namelist()) == 19
    assert set(archive.namelist()) == {m['name'] for m in package_manifest['members']}
    for member in package_manifest['members']:
        data = archive.read(member['name'])
        assert len(data) == member['bytes'] and digest(data) == member['sha256']
pdfinfo = subprocess.check_output(['pdfinfo', str(pdf)], text=True)
assert any(line.split() == ['Pages:', '10'] for line in pdfinfo.splitlines())
for name, source in [('PAPER_LICENSE.txt', 'paper/LICENSE'), ('SOFTWARE_LICENSE.txt', 'LICENSE')]:
    assert (BUNDLE / name).read_bytes() == git('show', SHA + ':' + source)
if (BUNDLE / 'PREVIEW.md').exists():
    assert (BUNDLE / 'PREVIEW.md').read_bytes() == (RECORD / 'release-notes.md').read_bytes()
else:
    with (BUNDLE / 'PREVIEW.md').open('xb') as output:
        output.write((RECORD / 'release-notes.md').read_bytes())
payload_names = ['PAPER_LICENSE.txt', 'SOFTWARE_LICENSE.txt', 'PREVIEW.md',
                 'ckks-train-e93cec4-source.tar.gz', 'ckks-train-paper-preview.pdf',
                 'ckks-train-arxiv-source.zip', 'source-inventory.json']
# macOS AppleDouble sidecars on the removable filesystem are not release assets.
assert {p.name for p in BUNDLE.iterdir() if not p.name.startswith('._')} == set(payload_names)
assets = []
for path in [BUNDLE / name for name in sorted(payload_names)]:
    assert path.is_file()
    data = path.read_bytes()
    assets.append(dict(name=path.name, bytes=len(data), sha256=digest(data)))
assert len(assets) == 7
manifest = dict(source_commit=SHA, tag='v1.0.0-preview.2',
                created_utc=datetime.now(timezone.utc).isoformat(),
                paper_pages=10, paper_inputs=18, zip_members=19, source_files=6113,
                ci_url=ci['url'], ci_conclusion='success',
                scope='research preview; archived-evidence reproduction; no new experiment',
                software_license='MIT', manuscript_notice='PAPER_LICENSE.txt',
                arxiv_submitted=False, assets=assets,
                manifest_scope='payload assets; SHA256SUMS also covers this manifest')
save(BUNDLE / 'manifest.json', manifest)
with (BUNDLE / 'SHA256SUMS').open('x') as output:
    for name in sorted(payload_names + ['manifest.json']):
        output.write(digest((BUNDLE / name).read_bytes()) + '  ' + name + '\n')
for line in (BUNDLE / 'SHA256SUMS').read_text().splitlines():
    expected, name = line.split('  ', 1)
    assert digest((BUNDLE / name).read_bytes()) == expected
for name in ['manifest.json', 'SHA256SUMS']:
    shutil.copyfile(BUNDLE / name, RECORD / name)
save(RECORD / 'qualification.json', dict(
    source_commit=SHA, verified_utc=datetime.now(timezone.utc).isoformat(),
    all_source_blobs_modes_and_inventory_match=True, source_files=6113,
    paper_input_hashes_match=True, zip_members_match=True, pdf_pages=10,
    asset_count=9, ci_jobs_passed=6,
    retained_reproduction='results/validation/research-preview2-e93cec4/reproduction.json',
    new_training_runs=False, pdf_rebuilt=False, arxiv_submitted=False))
print(json.dumps({'qualified': True, 'bundle': str(BUNDLE), 'assets': 9}))
