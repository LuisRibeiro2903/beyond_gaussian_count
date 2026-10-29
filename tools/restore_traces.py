#!/usr/bin/env python3
"""Restore exact numerical traces for historical aggregators; refuse overwrites."""
import gzip
import hashlib
import json
import os
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def main():
    restored = unchanged = 0
    for item in json.loads((ROOT / 'release/trace_index.json').read_text()):
        target = ROOT / item['path']
        raw = gzip.decompress((ROOT / item['compressed_path']).read_bytes())
        if len(raw) != item['bytes'] or hashlib.sha256(raw).hexdigest() != item['sha256']:
            raise SystemExit(f"Corrupt archived trace: {item['compressed_path']}")
        if target.exists():
            if hashlib.sha256(target.read_bytes()).hexdigest() != item['sha256']:
                raise SystemExit(f'Refusing to overwrite changed trace: {target}')
            unchanged += 1
            continue
        fd, temporary = tempfile.mkstemp(dir=target.parent, prefix='.restore-')
        try:
            with os.fdopen(fd, 'wb') as handle:
                handle.write(raw)
            os.replace(temporary, target)
        finally:
            if Path(temporary).exists():
                Path(temporary).unlink()
        restored += 1
    print(f'Restored {restored} traces; {unchanged} already matched.')

if __name__ == '__main__':
    main()
