"""Package committed source and the byte-verified, already reviewed manuscript."""
import hashlib, json, pathlib, re, shutil, subprocess, tarfile, zipfile
from datetime import datetime, timezone
root = pathlib.Path('/Users/marconi/Documents/research/hekit')
sha = 'e93cec434652a8c9f8c48becc6eb3045a3f78a7f'
record = root / 'results/validation/research-preview2-e93cec4'
base = pathlib.Path(subprocess.check_output(['python3', str(root/'tools/release/_ext/storage_root.py'), 'archives'], text=True).strip())
out = base/'research-preview2-e93cec4'
out.mkdir(exist_ok=False)
def git(*args): return subprocess.check_output(['git', '-C', str(root), *args])
def digest(data): return hashlib.sha256(data).hexdigest()
def save(name, value): (record/name).write_text(json.dumps(value, indent=2)+'\n')
meta = json.loads((root/'results/validation/paper-full2-r47-20261108/direct-build-meta.json').read_text())
for path, expected in meta['paper_inputs_sha256'].items():
    assert digest(git('show', sha+':'+path)) == expected, path
pdf = base.parent/'exports/paper-full2-r47-20261108-final/arxiv-wrapper.pdf'
zip_path = base/'arxiv-source-full2-r47-20261108/ckks-train-arxiv-source.zip'
assert digest(pdf.read_bytes()) == '9d896a45296754a22de0ba4a4daa5f832a7900f689b62363588a02d2eb22699c'
assert digest(zip_path.read_bytes()) == '2fa463bb471257b89d83414694de18f758bb40459e1b3e5e0f813297252a01c2'
zm = json.loads((root/'results/validation/paper-full2-r47-20261108/package-manifest.json').read_text())
with zipfile.ZipFile(zip_path) as z:
    assert len(z.namelist()) == 19 and set(z.namelist()) == {m['name'] for m in zm['members']}
    for m in zm['members']:
        data = z.read(m['name'])
        assert len(data) == m['bytes'] and digest(data) == m['sha256'], m['name']
shutil.copyfile(pdf, out/'ckks-train-paper-preview.pdf')
shutil.copyfile(zip_path, out/'ckks-train-arxiv-source.zip')
(out/'PAPER_LICENSE.txt').write_bytes(git('show', sha+':paper/LICENSE'))
(out/'SOFTWARE_LICENSE.txt').write_bytes(git('show', sha+':LICENSE'))
archive = out/'ckks-train-e93cec4-source.tar.gz'
prefix = 'ckks-train-e93cec4/'
subprocess.run(['git', '-C', str(root), '-c', 'tar.umask=0022', 'archive', '--format=tar.gz', '--prefix='+prefix, '--output='+str(archive), sha], check=True)
entries = {}
for row in git('ls-tree', '-rz', sha).split(b'\0'):
    if not row: continue
    info, name = row.split(b'\t', 1)
    mode, kind, oid = info.decode().split()
    assert kind == 'blob'
    entries[name.decode()] = (mode, oid)
patterns = [rb'-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----', rb'gh[pousr]_[A-Za-z0-9]{36}', rb'github_pat_[A-Za-z0-9_]{50,}', rb'AKIA[0-9A-Z]{16}', rb'xox[baprs]-[0-9A-Za-z-]{20,}', rb'sk-proj-[A-Za-z0-9_-]{40,}']
inventory, findings = [], []
with tarfile.open(archive, 'r:gz') as tar:
    for m in tar:
        if m.isdir(): continue
        assert m.name.startswith(prefix) and m.isfile(), m.name
        path = m.name[len(prefix):]
        mode, oid = entries.pop(path)
        data = tar.extractfile(m).read()
        assert hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest() == oid, path
        assert m.mode == int(mode, 8) & 0o777, path
        assert '/_ext/' not in '/'+path, path
        for pattern in patterns:
            if re.search(pattern, data): findings.append({'path': path, 'pattern': pattern.decode()})
        inventory.append(dict(path=path, mode=mode, git_blob=oid, bytes=len(data), sha256=digest(data)))
assert not entries
save('scan.json', {'scope': 'Uncompressed tracked files in release archive; limited known credential patterns; not arbitrary secrets or compressed interiors', 'findings': findings})
assert not findings, 'inspect scan.json without printing matched values'
(out/'source-inventory.json').write_text(json.dumps(inventory, indent=2)+'\n')
proof = dict(kind='distribution_verification_not_new_scientific_measurement', source_commit=sha, prepared_utc=datetime.now(timezone.utc).isoformat(), bundle=str(out), source_files=len(inventory), all_source_blobs_and_modes_match=True, paper_inputs_matched=len(meta['paper_inputs_sha256']), zip_members_matched=19, paper_reused_from_verified_build=True, existing_visual_review='results/validation/paper-full2-r47-20261108/README.md', arxiv_submitted=False)
save('integrity.json', proof)
print(json.dumps(proof, indent=2))
