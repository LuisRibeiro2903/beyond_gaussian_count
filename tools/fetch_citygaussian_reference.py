#!/usr/bin/env python3
"""Obtain one pinned official source file for optional CPU oracle tests.

The upstream file is downloaded into ignored local storage, never bundled into
the companion. Its upstream licence and attribution remain applicable.
"""
import argparse
import hashlib
import json
import urllib.request
from pathlib import Path

REVISION = 'e16c526f78ab01d377b13b3e6862dcc93ddbe953'
URL = f'https://raw.githubusercontent.com/Linketic/CityGaussian/{REVISION}/utils/large_utils.py'
SHA256 = '3fcdd9d97304cf0f2447be016ff16790460396732b7d575c6e214297fec7d75c'

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('local/reference-citygaussian'))
    args = parser.parse_args()
    target = args.output/'utils/large_utils.py'
    if target.exists():
        data = target.read_bytes()
    else:
        with urllib.request.urlopen(URL, timeout=60) as response:
            data = response.read()
    if hashlib.sha256(data).hexdigest() != SHA256:
        raise SystemExit('Official reference hash mismatch; refusing to use or overwrite the source.')
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(data)
    print(json.dumps({'status':'PASS','revision':REVISION,'sha256':SHA256,'source':str(target),
                      'terms':'CityGaussian CC BY-NC-SA 4.0; obtain full upstream licence from the official repository'}))

if __name__ == '__main__': main()
