#!/usr/bin/env python3
"""Check the public file inventory, licence coverage, exclusions and credential patterns.

This mechanical gate supplements the documented source/licence review. It does
not infer legal ownership or replace the original upstream terms.
"""
import argparse
import gzip
import json
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PATTERNS = {
    'private key': r'-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----',
    'GitHub token': r'\b(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{30,})\b',
    'AWS access key': r'\bAKIA[A-Z0-9]{16}\b',
    'Hugging Face token': r'\bhf_[A-Za-z0-9]{30,}\b',
    'Google API key': r'\bAIza[A-Za-z0-9_-]{35}\b',
}
CAMERA_FIELDS = {'camera_to_world', 'c2w', 'camera_center', 'rotation_world_to_camera',
                 'qvec_world_to_camera', 'tvec_world_to_camera', 'world_view_transform',
                 'full_proj_transform', 'full_projection_transform'}

def contains_camera_payload(value):
    if isinstance(value, dict):
        return any((key in CAMERA_FIELDS and isinstance(item, (list, dict))) or
                   contains_camera_payload(item) for key, item in value.items())
    if isinstance(value, list):
        return any(contains_camera_payload(item) for item in value)
    return False

def tracked_and_new():
    result = subprocess.check_output(['git', 'ls-files', '--cached', '--others', '--exclude-standard', '-z'], cwd=ROOT)
    return sorted({p.decode() for p in result.split(b'\0') if p and (ROOT / p.decode()).is_file()})

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--history', action='store_true', help='Also scan all reachable Git commits')
    args = parser.parse_args()
    files = tracked_and_new()
    mapping = {x['path']: x for x in json.loads((ROOT/'release/licensing.json').read_text())['files']}
    excluded = {x['path'] for x in json.loads((ROOT/'release/public_exclusions.json').read_text())['files']}
    errors = []
    for name in files:
        if name not in mapping:
            errors.append('Missing licence classification: '+name)
        if name in excluded:
            errors.append('Excluded dataset payload present: '+name)
        raw = (ROOT/name).read_bytes()
        if name.endswith('.gz'):
            raw = gzip.decompress(raw)
        text = raw.decode('utf-8', errors='replace')
        if name.endswith('.json') and any(word in (name+text).lower() for word in ['campus','sci-art','sciart']):
            if contains_camera_payload(json.loads(text)):
                errors.append('Raw Campus/Sci-Art camera payload present: '+name)
        for kind, pattern in PATTERNS.items():
            if re.search(pattern, text):
                errors.append('Potential '+kind+': '+name)
    history_count = 0
    if args.history:
        commits = subprocess.check_output(['git','rev-list','--all'],cwd=ROOT,text=True).splitlines()
        seen = set()
        for commit in commits:
            history_count += 1
            tree = subprocess.check_output(['git','ls-tree','-r',commit],cwd=ROOT,text=True)
            for row in tree.splitlines():
                meta,name = row.split('\t',1)
                if name in excluded:
                    errors.append('Excluded payload in reachable history: '+name)
                oid = meta.split()[2]
                if oid in seen: continue
                seen.add(oid)
                data = subprocess.check_output(['git','cat-file','blob',oid],cwd=ROOT)
                if name.endswith('.gz'):
                    data = gzip.decompress(data)
                text = data.decode('utf-8',errors='replace')
                if name.endswith('.json') and any(word in (name+text).lower() for word in ['campus','sci-art','sciart']):
                    if contains_camera_payload(json.loads(text)):
                        errors.append('Raw Campus/Sci-Art camera payload in history: '+name)
                for kind, pattern in PATTERNS.items():
                    if re.search(pattern,text): errors.append('Potential '+kind+' in history: '+name)
    if errors:
        raise SystemExit('\n'.join(sorted(set(errors))))
    print(json.dumps({'status':'PASS','classified_files':len(files),'reachable_commits_scanned':history_count,
                      'excluded_camera_payloads':len(excluded),'credential_patterns':list(PATTERNS)},indent=2))

if __name__ == '__main__': main()
