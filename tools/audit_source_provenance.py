#!/usr/bin/env python3
"""Find substantial identical Python function bodies in supplied upstream trees.

This is a review aid. No match does not prove independent authorship or licence
compatibility. The report contains locations/hashes, not copied upstream code.
"""
import argparse
import ast
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def functions(base):
    for path in sorted(base.rglob('*.py')):
        if any(part in {'.git', '__pycache__', 'submodules', 'third_party'} for part in path.relative_to(base).parts):
            continue
        try:
            tree = ast.parse(path.read_text())
        except (UnicodeError, SyntaxError):
            continue
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            body = node.body
            if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant) and isinstance(body[0].value.value, str):
                body = body[1:]
            module = ast.Module(body=body, type_ignores=[])
            if sum(1 for _ in ast.walk(module)) < 100:
                continue
            identity = hashlib.sha256(ast.dump(module, include_attributes=False).encode()).hexdigest()
            yield identity, {'path':str(path.relative_to(base)), 'line':node.lineno, 'function':node.name}

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--upstream', action='append', type=Path, required=True)
    args = parser.parse_args()
    index = {}
    for base in args.upstream:
        if not base.is_dir():
            parser.error(f'Not a directory: {base}')
        for identity, record in functions(base):
            index.setdefault(identity, []).append({'upstream':base.name, **record})
    matches = []
    for identity, record in functions(ROOT / 'gs_framework'):
        if identity in index:
            matches.append({'project':record, 'upstream':index[identity], 'body_ast_sha256':identity})
    print(json.dumps({'method':'Identical AST function bodies with >=100 AST nodes; docstrings excluded; submodules excluded', 'limitation':'Candidate provenance matches only; absence of matches is not licensing clearance.', 'matches':matches}, indent=2))

if __name__ == '__main__':
    main()
