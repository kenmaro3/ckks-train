"""Verify public release and stream-download every asset without authentication."""
import hashlib
import json
import pathlib
import subprocess
import urllib.request
from datetime import datetime, timezone

record = pathlib.Path(__file__).resolve().parent
repo = 'kenmaro3/ckks-train'
tag = 'v1.0.0-preview.2'
sha = 'e93cec434652a8c9f8c48becc6eb3045a3f78a7f'
manifest = json.loads((record / 'manifest.json').read_text())
expected = {x['name']: (x['bytes'], x['sha256']) for x in manifest['assets']}
for name in ['manifest.json', 'SHA256SUMS']:
    data = (record / name).read_bytes()
    expected[name] = (len(data), hashlib.sha256(data).hexdigest())


def get(url):
    return urllib.request.urlopen(urllib.request.Request(
        url, headers={'User-Agent': 'ckks-train-public-artifact-verification'}), timeout=60)


with get(f'https://api.github.com/repos/{repo}/releases/tags/{tag}') as response:
    assert response.status == 200
    release = json.load(response)
assert not release['draft'] and release['prerelease'] and release['published_at']
assert len(release['assets']) == 9 and {a['name'] for a in release['assets']} == set(expected)
with get(release['html_url']) as response:
    assert response.status == 200
with get(f'https://api.github.com/repos/{repo}/git/ref/tags/{tag}') as response:
    ref = json.load(response)
assert ref['object']['type'] == 'commit' and ref['object']['sha'] == sha
verified = []
for asset in release['assets']:
    size, checksum = expected[asset['name']]
    assert asset['size'] == size and asset['digest'] == 'sha256:' + checksum
    actual = hashlib.sha256()
    count = 0
    with get(asset['browser_download_url']) as response:
        assert response.status == 200
        while chunk := response.read(1024 * 1024):
            count += len(chunk)
            actual.update(chunk)
    assert count == size and actual.hexdigest() == checksum, asset['name']
    verified.append(dict(name=asset['name'], bytes=count, sha256=actual.hexdigest(),
                         anonymous_http=200))
    print('verified:', asset['name'], flush=True)
for name, value in [('public-release.json', release), ('public-tag.json', ref),
                    ('public-verification.json', dict(
                        verified_utc=datetime.now(timezone.utc).isoformat(),
                        release_id=release['id'], published_at=release['published_at'],
                        url=release['html_url'], source_commit=sha,
                        release_page_anonymous_http=200, all_assets_downloaded=True,
                        assets=verified, arxiv_submitted=False))]:
    with (record / name).open('x') as output:
        json.dump(value, output, indent=2)
        output.write('\n')
