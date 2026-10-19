"""Download verified model assets declared in artifacts/model_manifest.json."""
import hashlib
import json
from pathlib import Path
import sys
import urllib.request

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    out = Path(sys.argv[1] if len(sys.argv) > 1 else ROOT / 'artifacts')
    manifest = json.loads((ROOT / 'artifacts/model_manifest.json').read_text())
    for asset in manifest['models']:
        dest = out / asset['path']; dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.exists() and digest(dest) == asset['sha256']:
            print('verified local:', asset['path']); continue
        tmp = dest.with_suffix('.download')
        try:
            with tmp.open('wb') as f:
                for part in asset['parts']:
                    req = urllib.request.Request(part['url'], headers={'User-Agent':'antispam-coursework/2.0'})
                    with urllib.request.urlopen(req, timeout=120) as r:
                        data = r.read()
                    if len(data) != part['size'] or hashlib.sha256(data).hexdigest() != part['sha256']:
                        raise RuntimeError('model part checksum mismatch: ' + part['name'])
                    f.write(data)
            if digest(tmp) != asset['sha256']:
                raise RuntimeError('assembled model checksum mismatch: ' + asset['path'])
            tmp.replace(dest)
        finally:
            tmp.unlink(missing_ok=True)
        print('downloaded and verified:', asset['path'])


if __name__ == '__main__':
    main()
