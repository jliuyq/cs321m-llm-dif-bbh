#!/usr/bin/env python3
"""Create a small processed BBH fixture from full processed BBH files.

The fixture is for smoke testing only. It samples real processed rows from the
same schema expected by run_harmonized_analysis.py, but it is intentionally too
small to reproduce manuscript numbers.
"""
from __future__ import annotations
from pathlib import Path
import argparse
import pandas as pd

DOMAINS = [
    'boolean_expressions', 'date_understanding', 'disambiguation_qa', 'geometric_shapes',
    'hyperbaton', 'logical_deduction_five_objects', 'logical_deduction_three_objects',
    'temporal_sequences'
]

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--full-data-dir', type=Path, required=True, help='Directory with full processed BBH files.')
    ap.add_argument('--sidecar', type=Path, default=None, help='Path to full decoded sidecar CSV.GZ; defaults to full-data-dir/BBH_decoded_predictions_main_decodable.csv.gz.')
    ap.add_argument('--out-dir', type=Path, default=Path('data/example_bbh'))
    ap.add_argument('--items-per-domain', type=int, default=15)
    ap.add_argument('--models-per-family', type=int, default=20)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    sidecar_path = args.sidecar or (args.full_data_dir / 'BBH_decoded_predictions_main_decodable.csv.gz')
    item = pd.read_csv(args.full_data_dir / 'BBH_item_metadata.csv.gz')
    selected_items = []
    for d in DOMAINS:
        selected_items += item[item.domain == d].sort_values('item').head(args.items_per_domain).item.tolist()
    item_ex = item[item.item.isin(selected_items)].copy()

    meta = pd.read_csv(args.full_data_dir / 'BBH_model_metadata.csv.gz')
    # Find models with decoded sidecar coverage for all selected items.
    usecols = ['id','item','domain','target_idx','pred_idx','option_count','resp','decode_status']
    chunks = []
    for ch in pd.read_csv(sidecar_path, usecols=usecols, chunksize=1_000_000):
        ch = ch[ch.item.isin(selected_items) & ch.decode_status.eq('ok')].copy()
        if len(ch): chunks.append(ch)
    side = pd.concat(chunks, ignore_index=True)
    cov = side.groupby('id').item.nunique()
    complete_ids = set(cov[cov == len(selected_items)].index)
    cand = meta[meta.id.isin(complete_ids)].copy()
    selected_models = []
    for fam in ['Llama','Qwen','Other']:
        pool = cand[cand.architecture_family.eq(fam) if fam != 'Other' else ~cand.architecture_family.isin(['Llama','Qwen'])]
        selected_models += pool.sort_values('id').head(args.models_per_family).id.tolist()
    meta_ex = meta[meta.id.isin(selected_models)].copy()
    side_ex = side[side.id.isin(selected_models) & side.item.isin(selected_items)].copy()

    resp_chunks=[]
    for ch in pd.read_csv(args.full_data_dir / 'BBH_responses.csv.gz', usecols=['id','item','resp','model_resp'], chunksize=1_000_000):
        ch=ch[ch.id.isin(selected_models)&ch.item.isin(selected_items)].copy()
        if len(ch): resp_chunks.append(ch)
    resp_ex=pd.concat(resp_chunks, ignore_index=True)

    meta_ex.to_csv(args.out_dir / 'BBH_model_metadata.csv.gz', index=False, compression='gzip')
    item_ex.to_csv(args.out_dir / 'BBH_item_metadata.csv.gz', index=False, compression='gzip')
    resp_ex.to_csv(args.out_dir / 'BBH_responses.csv.gz', index=False, compression='gzip')
    side_ex.to_csv(args.out_dir / 'BBH_decoded_predictions_main_decodable.csv.gz', index=False, compression='gzip')
    print(f'wrote {args.out_dir}: models={meta_ex.id.nunique()}, items={item_ex.item.nunique()}, responses={len(resp_ex)}, sidecar_rows={len(side_ex)}')

if __name__ == '__main__':
    main()
