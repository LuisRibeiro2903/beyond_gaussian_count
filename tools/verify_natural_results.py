#!/usr/bin/env python3
"""Recompute the final natural ALoD-Exec table statistics from included traces."""
import json
import math
import statistics
from pathlib import Path
from evidence_store import EvidenceStore

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = Path('benchmark_artifacts/alod_ext_natural_previous_completion_001')
ARCHIVE = None

def rows(path):
    raw = ARCHIVE.jsonl_bytes(path.as_posix())
    return [json.loads(line) for line in raw.decode().splitlines() if line]

def require(actual, expected, label):
    if actual is None or expected is None:
        if actual is not expected: raise ValueError(label)
    elif not math.isclose(actual, expected, rel_tol=1e-10, abs_tol=1e-7):
        raise ValueError(f'{label}: {actual} != {expected}')

def main():
    global ARCHIVE
    ARCHIVE = EvidenceStore()
    summary = json.loads((ROOT/'results/alod_exec/natural_completion.json').read_text())
    checks = processes = frames = 0
    for dataset, content in summary['datasets'].items():
        for mode, table in content['modes'].items():
            measured, warmup = [], []
            for rep in range(1,4):
                folder = PACKAGE/'runs'/dataset/mode/f'replicate-{rep:02d}'
                sequence = rows(folder/'frames.jsonl')
                warm = rows(folder/'warmup_frames.jsonl')
                if len(sequence) != content['camera_count'] or len(warm) != 10:
                    raise ValueError(f'Wrong frame schedule: {folder}')
                if [row['camera_id'] for row in sequence] != table['camera_ids']:
                    raise ValueError(f'Wrong camera order: {folder}')
                measured.extend(sequence); warmup.extend(warm)
                processes += 1; frames += len(sequence)
            for section in ['timings_ms','counts']:
                for key, expected in table[section].items():
                    values = sorted(float(row[key]) for row in measured if row.get(key) is not None)
                    actual = dict.fromkeys(['minimum','mean','median','p95','maximum'])
                    if values:
                        offset = .95*(len(values)-1)
                        lo,hi = math.floor(offset),math.ceil(offset)
                        actual.update(minimum=values[0],maximum=values[-1],mean=statistics.mean(values),
                                      median=statistics.median(values),p95=values[lo]+(offset-lo)*(values[hi]-values[lo]))
                    for statistic,value in expected.items():
                        require(actual[statistic],value,f'{dataset}/{mode}/{key}/{statistic}');checks += 1
            for population,label in [(measured,'steady'),(warmup,'warmup'),(measured+warmup,'cold_inclusive')]:
                for unit in ['rows','bytes']:
                    actual=sum(row.get('alod_h2d_'+unit,0) or 0 for row in population)
                    require(actual,table[f'{label}_dynamic_property_h2d_{unit}'],f'{dataset}/{mode}/{label}/{unit}');checks += 1
    print(json.dumps({'status':'PASS','fresh_process_records':processes,'measured_frames':frames,
                      'table_statistics_checked':checks,'scope':'saved numerical evidence; no GPU, camera payload or scene required'},indent=2))

if __name__ == '__main__': main()
