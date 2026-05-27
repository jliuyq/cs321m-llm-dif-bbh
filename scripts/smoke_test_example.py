#!/usr/bin/env python3
"""Run the small example fixture and check that core outputs are produced."""
from __future__ import annotations
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/example_smoke_test'
CMD = [
    sys.executable, str(ROOT / 'scripts/run_harmonized_analysis.py'),
    '--data-dir', str(ROOT / 'data/example_bbh'),
    '--out-dir', str(OUT),
    '--seed', '20260526',
    '--score-bins', '5',
    '--min-group-per-stratum', '2',
    '--anchor-q', '0.20',
    '--madness-min-valid-items-floor', '5',
    '--min-cluster-size', '5',
]
subprocess.check_call(CMD, cwd=ROOT)
required = [
    'dif_abc_summary_common_dif_items.csv',
    'heldout_evaluation_consequence_table.csv',
    'response_derived_group_sizes.csv',
    'split_diagnostics.json',
]
missing = [f for f in required if not (OUT / f).exists()]
if missing:
    raise SystemExit(f'Smoke test failed; missing outputs: {missing}')
print('SMOKE TEST PASSED: example fixture ran end-to-end and produced core outputs.')
