#!/usr/bin/env python3
"""Verify packaged file identities and recoverable trace hashes; no GPU needed."""
import hashlib
import json
from pathlib import Path
from evidence_store import EvidenceStore

ROOT = Path(__file__).resolve().parents[1]

def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def main():
    errors = []
    manifest = json.loads((ROOT / 'companion.json').read_text())
    for item in manifest['files']:
        if item['path'] == 'companion.json':
            continue  # A manifest cannot include its own byte hash; Git records it.
        path = ROOT / item['path']
        if not path.is_file() or digest(path) != item['sha256']:
            errors.append(item['path'])
    archive = EvidenceStore()
    for item in manifest['results']:
        raw = archive.read(item['archived_source'])
        if hashlib.sha256(raw).hexdigest() != item['sha256'] or (ROOT/item['path']).read_bytes() != raw:
            errors.append(item['path'] + ': original result identity')
    if errors:
        raise SystemExit('FAILED\n' + '\n'.join(errors))
    print(f"PASS: {len(manifest['files'])} reader files; {len(manifest['results'])} unchanged original result records; pinned archive manifest.")
    print('This checks release integrity, not GPU measurements or redistribution rights.')

if __name__ == '__main__':
    main()
