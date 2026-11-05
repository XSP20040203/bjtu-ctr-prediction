"""Vendor the official Netron browser assets without installing or running it."""
from pathlib import Path
from urllib.request import urlopen
import hashlib
import io
import json
import zipfile

ROOT = Path(__file__).resolve().parent
VERSION = '9.3.4'
metadata = json.load(urlopen(f'https://pypi.org/pypi/netron/{VERSION}/json', timeout=40))
artifact = next(x for x in metadata['urls'] if x['filename'].endswith('.whl'))
raw = urlopen(artifact['url'], timeout=60).read()
assert hashlib.sha256(raw).hexdigest() == artifact['digests']['sha256']
destination = ROOT / 'vendor' / 'netron'
destination.mkdir(parents=True, exist_ok=True)
count = 0
with zipfile.ZipFile(io.BytesIO(raw)) as package:
    for entry in package.infolist():
        if entry.is_dir():
            continue
        if entry.filename.startswith('netron/') and Path(entry.filename).suffix.lower() in {'.js', '.css', '.html', '.json', '.ico', '.png', '.svg', '.woff', '.woff2'}:
            rel = Path(entry.filename).relative_to('netron')
            target = (destination / rel).resolve()
            assert target.is_relative_to(destination.resolve())
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(package.read(entry))
            count += 1
        if entry.filename.endswith('/licenses/LICENSE'):
            (destination / 'LICENSE').write_bytes(package.read(entry))
manifest = dict(name='Netron', version=VERSION, source='https://github.com/lutzroeder/netron', package_url=artifact['url'],
                wheel_sha256=artifact['digests']['sha256'], browser_asset_count=count, license='MIT')
(destination / 'vendor_manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
print(json.dumps(manifest))
