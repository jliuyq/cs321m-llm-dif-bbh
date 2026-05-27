# Data

`data/example_bbh/` is a small real processed BBH fixture sampled from the final processed data. It is for smoke testing only and does **not** reproduce manuscript numbers.

Full manuscript reproduction requires these processed files under `data/processed_bbh/`:

- `BBH_responses.csv.gz`
- `BBH_model_metadata.csv.gz`
- `BBH_item_metadata.csv.gz`
- `BBH_decoded_predictions_main_decodable.csv.gz`

Raw Open LLM Leaderboard / HuggingFace caches are not included because they are large and slow to rebuild. No HuggingFace tokens or credentials are included.

See `processed_data_manifest.json` for expected filenames, sizes, hashes, and row counts of the full processed files used by the manuscript run.
