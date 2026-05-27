# PACKAGING_REPORT.md

## Included files

- Final core analysis script: `scripts/run_harmonized_analysis.py`
- Figure regeneration script and final figure: `scripts/plot_consequence_gap.py`, `figures/consequence_gap.{png,pdf}`
- Example smoke fixture: `data/example_bbh/`
- Example fixture builder: `scripts/make_example_data.py`
- Smoke test: `scripts/smoke_test_example.py`
- Preserved small manuscript outputs: `outputs/final/`
- Documentation: `README.md`, `data/README.md`, `docs/SOURCE_FACT_MAP.md`, `docs/REPRODUCIBILITY_AUDIT.md`
- Manuscript PDF only: `manuscript/final_report.pdf`

## Excluded files

- Raw HuggingFace / Open LLM Leaderboard caches
- Full processed BBH inputs
- Huge item-level DIF files
- Exploratory scripts and abandoned analyses
- Old reports and old PDFs
- OpenClaw logs
- `__pycache__`, `.DS_Store`, local environment files

## Smoke test

Passed. Command run:

```bash
python scripts/smoke_test_example.py
```

Output:

```
[18:25:40] models=60; group_items=80; dif_items=40
[18:25:40] response chunks=1 total=7,200 kept=7,200
[18:25:40] sidecar chunks=1 total=7,200 kept=4,252
[18:25:40] MADNESS matrix multiply for n=42, items=80
[18:25:40] MADNESS matrix multiply for n=60, items=80
[18:25:40] done: outputs/example_smoke_test; runtime=0.1s
SMOKE TEST PASSED: example fixture ran end-to-end and produced core outputs.
```

## Full rerun

Not rerun inside this minimal package after pruning, because full processed BBH inputs are intentionally not included. The same core script was previously used to generate the preserved manuscript summaries.

## Figure regeneration

Passed. `scripts/plot_consequence_gap.py` now regenerates the manuscript-style three-panel point plot with error bars from `gap_pp_draw_mean` and `gap_pp_draw_sd`. Command run:

```bash
python scripts/plot_consequence_gap.py
```

Output:

```
wrote figures/consequence_gap.png and figures/consequence_gap.pdf
```

## Expected runtime

- Example smoke test: under 1 minute on the packaging machine.
- Figure regeneration: under 1 minute on the packaging machine.
- Full processed-data rerun: about 1--2 minutes on the original Mac mini after processed BBH CSVs already exist; varies by disk/CPU.

## Known limitations

- Example fixture verifies the code path only and does not reproduce manuscript numbers.
- Full preprocessing from OLL/HuggingFace is not bundled.
- Family baseline uses simplified MH/ETS inference, not cluster-robust Phillips-Holland/Rao-Scott SE.
- Difficulty-matched consequence baselines are not included in the harmonized final run.

## Security cleanup

No token/secret patterns or local absolute paths were found in the final scan.

## Final security scan output
# Security scan

## Final absolute path scan output
# Absolute path scan
