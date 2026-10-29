#!/usr/bin/env python3
"""Verify packaged file identities and recoverable trace hashes; no GPU needed."""
import gzip
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def digest(path, compressed=False):
    h = hashlib.sha256()
    opener = gzip.open if compressed else open
    with opener(path, 'rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def main():
    errors = []
    manifest = json.loads((ROOT / 'release/export_manifest.json').read_text())
    for item in manifest['files']:
        path = ROOT / item['path']
        if not path.is_file() or digest(path) != item['sha256']:
            errors.append(item['path'])
    traces = json.loads((ROOT / 'release/trace_index.json').read_text())
    for item in traces:
        path = ROOT / item['compressed_path']
        if not path.is_file() or digest(path, True) != item['sha256']:
            errors.append(item['compressed_path'] + ': original trace identity')
    if errors:
        raise SystemExit('FAILED\n' + '\n'.join(errors))
    print(f"PASS: {len(manifest['files'])} packaged files; {len(traces)} original trace hashes.")
    print('This checks release integrity, not GPU measurements or redistribution rights.')

if __name__ == '__main__':
    main()
