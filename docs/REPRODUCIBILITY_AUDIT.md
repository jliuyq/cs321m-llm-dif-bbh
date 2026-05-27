# REPRODUCIBILITY_AUDIT.md

- Are all random seeds fixed? **Yes.** Use `--seed`; manuscript run used seed `20260526`.
- Are group-discovery items disjoint from DIF-test items? **Yes.** The script creates a single item split and stores it in `item_split_assignments.csv` for full reruns.
- Are DIF-test items excluded from strict MADNESS and SRRC grouping/tuning/centroid construction/evaluation assignment? **Yes.** Group learning and evaluation assignment use only group-discovery items.
- Are discovery and evaluation models disjoint? **Yes.** The script creates one model split and stores it in `model_split_assignments.csv` for full reruns.
- How are evaluation models assigned to strict MADNESS groups? Using a frozen discovery grouping and mean same-wrong z-similarity to G0/G1 discovery members, computed only on retained group-discovery items.
- How are evaluation models assigned to SRRC groups? Using frozen residualization/standardization and nearest frozen k-means centroid, computed only on group-discovery items.
- What is the gap sign convention? Family: Qwen − Llama; strict MADNESS: G1 − G0; SRRC: EPn1 − EPn2 in preserved CSV names.
- Is the family baseline cluster-robust? **No.** It is a simplified Hansol-style MH/ETS implementation, not Phillips-Holland/Rao-Scott cluster-robust inference.
- Are difficulty-matched baselines included? **No.** This is a limitation of the harmonized run.
- What does the example data reproduce? The example fixture reproduces the code path only, not manuscript numbers.
- What data are needed for full manuscript reproduction? Full processed `BBH_responses.csv.gz`, `BBH_model_metadata.csv.gz`, `BBH_item_metadata.csv.gz`, and `BBH_decoded_predictions_main_decodable.csv.gz` under `data/processed_bbh/`.
