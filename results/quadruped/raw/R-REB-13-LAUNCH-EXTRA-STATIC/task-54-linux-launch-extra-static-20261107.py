"""Read only venv-root metadata and one fixed typing source identity."""
from pathlib import Path
import hashlib, json, os, stat, time
root = Path('/home/kmihara/reboot/exporter-venv-task54-v1')
source = Path('/home/kmihara/.local/share/uv/python/cpython-3.11.16-linux-x86_64-gnu/lib/python3.11/typing.py')
record = {'kind': 'read_only_launch_prerequisite_observation_not_scientific_result', 'started_unix': time.time(), 'nice': os.nice(0), 'uid': os.getuid()}
assert record['nice'] >= 10
assert root.resolve() == root and source.resolve() == source
r = root.lstat()
assert stat.S_ISDIR(r.st_mode)
record['venv_root'] = {'path': str(root), 'uid': r.st_uid, 'mode': stat.S_IMODE(r.st_mode), 'device': r.st_dev, 'inode': r.st_ino}
s = source.lstat()
assert stat.S_ISREG(s.st_mode) and s.st_size <= 524288
fd = os.open(source, os.O_RDONLY | os.O_NOFOLLOW)
try:
    before = os.fstat(fd)
    assert stat.S_ISREG(before.st_mode) and before.st_size <= 524288
    with os.fdopen(fd, 'rb', closefd=False) as stream:
        data = stream.read(524289)
    after = os.fstat(fd)
    assert len(data) <= 524288 and (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) == (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
finally:
    os.close(fd)
record['typing_source'] = {'path': str(source), 'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest(), 'uid': before.st_uid, 'mode': stat.S_IMODE(before.st_mode)}
record['scope'] = 'Metadata and source hash only; no target startup, source execution, semantic or expanded helper-closure review, context, capture, scientific import, or job/queue change.'
record['finished_unix'] = time.time()
print(json.dumps(record, indent=2))
