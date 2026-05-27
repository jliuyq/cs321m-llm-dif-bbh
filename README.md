# CS321M LLM DIF / BBH

Minimal public reproduction package for the CS321M harmonized LLM DIF / BBH project.

## Project overview

This project treats BBH benchmark items as measurement items and models as respondents. It compares three harmonized DIF diagnostics on the same model split and same item split:

1. **Family baseline**: external Llama/Qwen labels.
2. **Strict MADNESS**: response-derived same-wrong grouping.
3. **SRRC**: residualized response-regime clustering with frozen centroid assignment (called EP-1b in some preserved CSV filenames).

The full BBH preprocessing is large, so this repository includes a small runnable fixture plus final summary outputs used by the manuscript.

## Repository structure

```text
cs321m-llm-dif-bbh/
├── README.md
├── requirements.txt
├── scripts/
│   ├── run_harmonized_analysis.py
│   ├── smoke_test_example.py
│   ├── make_example_data.py
│   └── plot_consequence_gap.py
├── data/
│   ├── README.md
│   ├── processed_data_manifest.json
│   └── example_bbh/
│       ├── BBH_responses.csv.gz
│       ├── BBH_model_metadata.csv.gz
│       ├── BBH_item_metadata.csv.gz
│       └── BBH_decoded_predictions_main_decodable.csv.gz
├── figures/
│   ├── consequence_gap.png
│   └── consequence_gap.pdf
├── outputs/
│   └── final/
│       ├── dif_abc_summary_common_dif_items.csv
│       ├── heldout_evaluation_consequence_table.csv
│       ├── response_derived_group_sizes.csv
│       ├── split_diagnostics.json
│       └── strict_madness_ward_mh_strata_diagnostics.csv
├── docs/
│   ├── SOURCE_FACT_MAP.md
│   └── REPRODUCIBILITY_AUDIT.md
└── manuscript/
    └── final_report.pdf
```

## Environment setup

Tested with Python 3.14.3. `requirements.txt` pins the exact package versions used for smoke testing and figure regeneration.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Quick smoke test

```bash
python scripts/smoke_test_example.py
```

Equivalent direct command:

```bash
python scripts/run_harmonized_analysis.py \
  --data-dir data/example_bbh \
  --out-dir outputs/example_run \
  --seed 20260526 \
  --score-bins 5 \
  --min-group-per-stratum 2 \
  --madness-min-valid-items-floor 5 \
  --min-cluster-size 5
```

This runs on the small fixture and verifies that the full code path executes: family baseline, strict MADNESS, SRRC, MH-DIF, and consequence analysis. It does **not** reproduce manuscript numbers.

## Full rerun

Place full processed BBH inputs under `data/processed_bbh/` (see `data/README.md`), then run:

```bash
python scripts/run_harmonized_analysis.py \
  --data-dir data/processed_bbh \
  --out-dir outputs/full_rerun \
  --seed 20260526
```

Expected runtime on the original Mac mini environment was about 1--2 minutes after processed BBH files already existed. Memory use was a few GB. Full preprocessing from OLL/HuggingFace was time-consuming and is not bundled.

## Figure regeneration

Figure 1 can be regenerated from the preserved consequence table:

```bash
python scripts/plot_consequence_gap.py
```

This writes `figures/consequence_gap.png` and `figures/consequence_gap.pdf`.

## Manuscript result checking

`outputs/final/` contains preserved final summaries used in the manuscript:

- model/item split counts: `split_diagnostics.json`
- A/B/C DIF counts: `dif_abc_summary_common_dif_items.csv`
- response-derived group sizes: `response_derived_group_sizes.csv`
- strict MADNESS usable strata: `strict_madness_ward_mh_strata_diagnostics.csv`
- held-out consequence gaps: `heldout_evaluation_consequence_table.csv`

## Limitations

- The family baseline uses simplified MH/ETS inference and does not implement cluster-robust Phillips-Holland/Rao-Scott standard errors.
- The example data are only a runnable fixture.
- Full raw OLL/HuggingFace caches and preprocessing are not bundled because of size.
- Difficulty-matched consequence baselines are not included in the harmonized run.
