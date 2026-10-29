#!/usr/bin/env python3
"""List canonical result tables or print a selected original JSON table."""
import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--table', help='Repository-relative path printed in the index')
    args = parser.parse_args()
    tables = sorted((ROOT / 'benchmark_artifacts').glob('*/tables/*.json'))
    if args.table:
        path = (ROOT / args.table).resolve()
        if path not in tables:
            parser.error('Choose a table from the index produced without --table')
        print(json.dumps(json.loads(path.read_text()), indent=2))
    else:
        for path in tables:
            print(path.relative_to(ROOT))
        print(f'\n{len(tables)} original result tables. Interpret each within its own protocol.')

if __name__ == '__main__':
    main()
