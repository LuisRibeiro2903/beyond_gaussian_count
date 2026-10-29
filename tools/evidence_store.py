"""Read hash-checked evidence from the pinned full-record Git revision or ZIP."""
import gzip
import hashlib
import json
import os
import subprocess
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[1]

class EvidenceStore:
    def __init__(self):
        self.reference = json.loads((ROOT/'companion.json').read_text())['evidence']
        self.directory = Path(os.environ['BEYOND_EVIDENCE_DIR']).resolve() if os.environ.get('BEYOND_EVIDENCE_DIR') else None
        self.cache = {}
        manifest = self._load('release/export_manifest.json')
        if hashlib.sha256(manifest).hexdigest() != self.reference['manifest_sha256']:
            raise ValueError('The full-record manifest does not match the pinned archive.')
        self.identities = {item['path']:item['sha256'] for item in json.loads(manifest)['files']}

    def _load(self, name):
        relative = PurePosixPath(name)
        if relative.is_absolute() or '..' in relative.parts:
            raise ValueError('Evidence paths must be relative to the full record.')
        if self.directory:
            return (self.directory/str(relative)).read_bytes()
        result = subprocess.run(['git','show',f"{self.reference['commit']}:{relative}"],cwd=ROOT,capture_output=True)
        if result.returncode:
            raise FileNotFoundError(f'Pinned evidence unavailable: {relative}. Run "git fetch origin tag v1.0.0", or set BEYOND_EVIDENCE_DIR to the extracted v1.0.0 archive.')
        return result.stdout

    def read(self, name):
        name = str(name)
        if name not in self.cache:
            raw = self._load(name)
            if name not in self.identities or hashlib.sha256(raw).hexdigest() != self.identities[name]:
                raise ValueError('Archived evidence identity mismatch: '+name)
            self.cache[name] = raw
        return self.cache[name]

    def json(self, name):
        return json.loads(self.read(name))

    def jsonl_bytes(self, name):
        return gzip.decompress(self.read(str(name)+'.gz'))
