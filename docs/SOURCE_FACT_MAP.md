# SOURCE_FACT_MAP.md

| Manuscript claim/table | Exact source file | Columns / JSON keys | Script | Computation status |
|---|---|---|---|---|
| Model/item split counts | `outputs/final/split_diagnostics.json` | `total_models`, `discovery_models`, `evaluation_models`, `group_discovery_items`, `dif_test_items` | `scripts/run_harmonized_analysis.py` | Directly computed |
| A/B/C counts and percentages | `outputs/final/dif_abc_summary_common_dif_items.csv` | `analysis`, `A_count`, `B_count`, `C_count`, `A_pct`, `B_pct`, `C_pct`, `usable_strata` | `scripts/run_harmonized_analysis.py` | Directly computed |
| Strict MADNESS usable strata | `outputs/final/strict_madness_ward_mh_strata_diagnostics.csv` | `pass`, `stratum`, `n_G0`, `n_G1`, `usable` | `scripts/run_harmonized_analysis.py` | Directly computed |
| Response-derived group sizes | `outputs/final/response_derived_group_sizes.csv` | `analysis_split`, `group`, `models` | `scripts/run_harmonized_analysis.py` | Directly computed |
| Consequence gaps | `outputs/final/heldout_evaluation_consequence_table.csv` | `analysis`, `pool`, `pool_items`, `eval_models`, `gap_pp_full_pool`, `gap_pp_draw_mean`, `gap_pp_draw_sd` | `scripts/run_harmonized_analysis.py` | Directly computed |
